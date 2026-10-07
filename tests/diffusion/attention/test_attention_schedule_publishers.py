# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contracts shared by every denoise progress publisher.

These tests do not load a model. A toy publisher with no model code drives a
real Attention layer with prepared fake candidates through the shared helpers:
begin_scheduled_denoise checks the bound schedule against the sequence the loop
builds, record_denoise_step publishes the integer step with that total, and
request_denoise_progress publishes one request's progress inside a shared
denoise_step call and restores the previous values. The file also covers the
step-mode request check, model-owned candidate checks at startup, and a
pipeline that stays rejected because it publishes no total.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.attention.layer as layer_mod
import vllm_omni.diffusion.forward_context as forward_context_mod
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
    InvalidAttentionScheduleError,
    require_attention_schedule_fits,
    require_denoise_progress_publisher,
    require_request_attention_schedule_fits,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionScheduleConfig, AttentionSpec, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import (
    DenoiseProgressMixin,
    begin_scheduled_denoise,
    bind_attention_schedule,
    get_forward_context,
    is_forward_context_available,
    request_denoise_progress,
    set_forward_context,
    set_forward_context_denoise_step_idx,
    set_forward_context_denoise_timestep,
    set_forward_context_denoise_total_steps,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.errors import OmniClientError, client_error_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeImpl:
    """Accepts any construction kwargs and returns the query."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def forward(self, query, key, value, attn_metadata=None):
        return query


_BACKEND_CACHE: dict[str, type] = {}


def _fake_backend(name: str) -> type:
    """A distinct fake backend (and impl subclass) per backend name."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    impl_cls = type(f"Fake{name}Impl", (_FakeImpl,), {})

    class _Backend:
        supports_paged_kv = False
        supports_piecewise_spans = False

        @staticmethod
        def get_name() -> str:
            return name

        @staticmethod
        def get_impl_cls():
            return impl_cls

        @staticmethod
        def supports_attention_mask(spec) -> bool:
            return True

    _Backend.__name__ = f"Fake{name}Backend"
    _BACKEND_CACHE[name] = _Backend
    return _Backend


def _fake_resolve(*, role, head_size, attention_config=None, role_category=None, allow_trtllm_default=True):
    """Resolve a fake backend from the supplied AttentionConfig, like the real selector."""
    spec = None
    if attention_config is not None:
        spec, _source = attention_config.resolve_with_source(role=role, role_category=role_category)
    if spec is None:
        return _fake_backend("PLATFORM_DEFAULT"), None
    return _fake_backend(spec.backend.upper()), spec


def _make_config(*, schedule: AttentionScheduleConfig | None = None) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(diffusion_attention_schedule=schedule)


@pytest.fixture
def attention_env(monkeypatch):
    monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", lambda **kwargs: _fake_resolve(**kwargs))

    def build(config, **kwargs):
        with set_current_diffusion_config(config):
            return Attention(num_heads=4, head_size=64, causal=False, softmax_scale=1.0, **kwargs)

    return SimpleNamespace(build=build)


def _service():
    return AttentionScheduleConfig(
        profiles={
            "sparse": AttentionConfig(default=AttentionSpec(backend="SDPA")),
            "dense": AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")),
        },
        default=[{"start": 0, "end": 2, "profile": "sparse"}],
    )


def _ranges(*entries):
    """A normalized schedule from (start, end, profile) triples."""
    return tuple(AttentionScheduleRange(start=start, end=end, profile=profile) for start, end, profile in entries)


def _timesteps(count):
    """A descending timestep sequence on a 1000-step training scale."""
    return [1000.0 * (count - index) / count for index in range(count)]


@contextmanager
def _runner_scope(config, schedule):
    """The forward context and bind the runner sets around pipeline.forward or denoise_step.

    The bind uses denoise=False; the publisher's first step publish activates the schedule.
    """
    with set_forward_context(omni_diffusion_config=config), bind_attention_schedule(schedule, denoise=False):
        yield get_forward_context()


def _record_forwards(layer, monkeypatch):
    """Record, for each attention forward, the implementation that ran and the progress it saw.

    Each entry is (profile name or "base", denoise_step_idx, total_denoise_steps, denoise_timestep).
    """
    calls: list[tuple[Any, ...]] = []

    def recorder(name):
        def forward(query, key, value, attn_metadata=None):
            ctx = get_forward_context()
            calls.append((name, ctx.denoise_step_idx, ctx.total_denoise_steps, ctx.denoise_timestep))
            return query

        return forward

    monkeypatch.setattr(layer.attention, "forward", recorder("base"))
    for name, record in layer._schedule_candidates.items():
        monkeypatch.setattr(record.impl, "forward", recorder(name))
    return calls


def _steps(names, total):
    """Expected (name, step, total) entries for one sequence: two forwards per step, then one after the loop."""
    return [(name, step, total) for step, name in enumerate(names) for _branch in range(2)] + [("base", None, None)]


def _names(layer, selected):
    """Replace each recorded implementation with its profile name, or "base"."""
    names = {id(layer.attention): "base"}
    names.update({id(record.impl): name for name, record in layer._schedule_candidates.items()})
    return [(*entry[:-1], names[id(entry[-1])]) for entry in selected]


def _progress(ctx):
    return ctx.denoise_step_idx, ctx.denoise_timestep, ctx.total_denoise_steps, ctx.attention_schedule_denoise_active


class _ToyPublisher(DenoiseProgressMixin):
    """A denoise loop with no model code.

    It builds its own timestep sequence, checks the bound schedule against the length of that
    sequence, publishes each step once and runs two attention forwards per step, as a CFG loop does.
    """

    def __init__(self, layer):
        self.layer = layer
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def _evaluate(self):
        query = torch.zeros(1, 2, 4, 64)
        self.layer(query, query, query)

    def run(self, timesteps):
        # Each call is one denoise sequence that starts at step 0, so the schedule is checked
        # against this sequence's own length.
        total = begin_scheduled_denoise(len(timesteps))
        for step_idx, timestep in enumerate(timesteps):
            self.record_denoise_step(step_idx, timestep, total_steps=total)
            self._evaluate()  # conditional branch
            self._evaluate()  # unconditional branch, same step
        self.record_denoise_step(None)
        self._evaluate()  # attention after the loop, e.g. during decode


# Toy publisher: selection follows the integer step and the actual total.


def test_toy_publisher_selects_by_integer_step_against_the_actual_total(attention_env, monkeypatch):
    # 8 steps with [3, 6) on "sparse"; the open range [7, None) ends at the actual total.
    config = _make_config(schedule=_service())
    layer = attention_env.build(config)
    calls = _record_forwards(layer, monkeypatch)
    timesteps = _timesteps(8)

    with _runner_scope(config, _ranges((3, 6, "sparse"), (7, None, "dense"))):
        _ToyPublisher(layer).run(timesteps)

    # Both forwards of a step share one publish, so they select the same profile.
    assert [entry[:3] for entry in calls] == _steps(["base"] * 3 + ["sparse"] * 3 + ["base", "dense"], 8)
    assert [entry[3] for entry in calls[:-1]] == [timestep / 1000 for timestep in timesteps for _branch in range(2)]


def test_toy_publisher_selection_ignores_numeric_timesteps(attention_env, monkeypatch):
    # Two sequences of the same length with different numeric timesteps. The second one's values are
    # small integers in reverse order, so selecting by timestep value would pick different profiles.
    config = _make_config(schedule=_service())
    layer = attention_env.build(config)
    calls = _record_forwards(layer, monkeypatch)
    schedule = _ranges((3, 6, "sparse"), (7, None, "dense"))
    published: list[list[float]] = []
    for timesteps in (_timesteps(8), [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]):
        calls.clear()
        with _runner_scope(config, schedule):
            _ToyPublisher(layer).run(timesteps)
        assert [entry[:3] for entry in calls] == _steps(["base"] * 3 + ["sparse"] * 3 + ["base", "dense"], 8)
        published.append([entry[3] for entry in calls[:-1]])

    assert published[0] != published[1]


@pytest.mark.parametrize(
    ("configured", "schedule"),
    [(False, None), (True, None), (True, ())],
    ids=["no-service-schedule", "unbound", "request-disables"],
)
def test_toy_publisher_without_a_schedule_publishes_no_total_and_runs_the_baseline(
    attention_env, monkeypatch, configured, schedule
):
    # Without a schedule the publisher publishes no total.
    config = _make_config(schedule=_service() if configured else None)
    layer = attention_env.build(config)
    calls = _record_forwards(layer, monkeypatch)

    with _runner_scope(config, schedule) as ctx:
        _ToyPublisher(layer).run(_timesteps(4))
        assert ctx.attention_schedule_denoise_active is False

    assert [entry[:3] for entry in calls] == _steps(["base"] * 4, None)


def test_toy_publisher_checks_and_selects_each_restarted_sequence_on_its_own(attention_env, monkeypatch):
    # A pipeline that restarts at step 0 (per output seed, continuation window or clip) runs one
    # sequence per restart. Each one is checked against its own length and selects from step 0 again.
    config = _make_config(schedule=_service())
    layer = attention_env.build(config)
    calls = _record_forwards(layer, monkeypatch)
    publisher = _ToyPublisher(layer)

    with _runner_scope(config, _ranges((1, None, "sparse"))):
        publisher.run(_timesteps(3))
        publisher.run(_timesteps(2))
    assert [entry[:3] for entry in calls] == _steps(["base", "sparse", "sparse"], 3) + _steps(["base", "sparse"], 2)

    calls.clear()
    with _runner_scope(config, _ranges((2, 3, "sparse"))):
        publisher.run(_timesteps(3))
        with pytest.raises(InvalidAttentionScheduleError, match="exceeds total_steps=2"):
            publisher.run(_timesteps(2))
    # The second sequence failed before its first forward.
    assert [entry[:3] for entry in calls] == _steps(["base", "base", "sparse"], 3)


def test_toy_publisher_rejects_a_short_sequence_before_any_attention_call(attention_env, monkeypatch):
    # The request asked for 8 steps, but the loop builds 5, as with a fixed DMD table.
    config = _make_config(schedule=_service())
    layer = attention_env.build(config)
    calls = _record_forwards(layer, monkeypatch)

    with _runner_scope(config, _ranges((3, 6, "sparse"))) as ctx:
        with pytest.raises(InvalidAttentionScheduleError, match="exceeds total_steps=5"):
            _ToyPublisher(layer).run(_timesteps(5))
        assert _progress(ctx) == (None, None, None, False)

    assert calls == []


# begin_scheduled_denoise


def test_begin_scheduled_denoise_returns_the_total_for_a_bound_schedule():
    with set_forward_context(), bind_attention_schedule(_ranges((0, 2, "sparse")), denoise=False):
        assert begin_scheduled_denoise(2) == 2
        assert begin_scheduled_denoise(8) == 8
        # The check publishes nothing; the publisher does.
        assert _progress(get_forward_context()) == (None, None, None, False)


@pytest.mark.parametrize("scope", ["no-context", "unbound", "disabled", "context-without-field"])
def test_begin_scheduled_denoise_returns_none_without_checking_when_no_schedule_is_bound(monkeypatch, scope):
    # total_steps=0 fails the range check, so a None result shows that the check did not run.
    if scope == "no-context":
        assert is_forward_context_available() is False
        assert begin_scheduled_denoise(0) is None
    elif scope == "context-without-field":
        # Some model tests install a recorder object that has no attention_schedule attribute.
        monkeypatch.setattr(forward_context_mod, "_forward_context", SimpleNamespace(denoise_step_idx=None))
        assert begin_scheduled_denoise(0) is None
    else:
        with set_forward_context(), bind_attention_schedule(None if scope == "unbound" else (), denoise=False):
            assert begin_scheduled_denoise(0) is None


@pytest.mark.parametrize(
    "schedule",
    [_ranges((3, 6, "sparse")), _ranges((5, None, "sparse"))],
    ids=["end-past-total", "open-range-starts-at-total"],
)
def test_begin_scheduled_denoise_rejects_a_range_past_the_actual_total(schedule):
    with set_forward_context(), bind_attention_schedule(schedule, denoise=False):
        with pytest.raises(InvalidAttentionScheduleError, match="exceeds total_steps=5") as excinfo:
            begin_scheduled_denoise(5)

    assert isinstance(excinfo.value, OmniClientError)
    assert client_error_metadata(excinfo.value) == (400, "invalid_attention_schedule")


# require_attention_schedule_fits and require_request_attention_schedule_fits


def test_require_attention_schedule_fits_accepts_ranges_inside_the_sequence():
    require_attention_schedule_fits((), 1)
    require_attention_schedule_fits(_ranges((0, 2, "sparse"), (3, None, "dense")), 4)
    require_attention_schedule_fits(_ranges((3, 6, "sparse")), 6)


@pytest.mark.parametrize(
    ("schedule", "total_steps", "message"),
    [
        (_ranges((3, 6, "sparse")), 5, "exceeds total_steps=5"),
        (_ranges((5, None, "sparse")), 5, "exceeds total_steps=5"),
        (_ranges((0, 1, "sparse")), 0, "total_steps must be >= 1"),
    ],
    ids=["end-past-total", "open-range-starts-at-total", "empty-sequence"],
)
def test_require_attention_schedule_fits_rejects_as_client_error(schedule, total_steps, message):
    with pytest.raises(InvalidAttentionScheduleError, match=message) as excinfo:
        require_attention_schedule_fits(schedule, total_steps)

    assert isinstance(excinfo.value, OmniClientError)
    assert isinstance(excinfo.value, ValueError)
    assert client_error_metadata(excinfo.value) == (400, "invalid_attention_schedule")


def _carrier(holder: str, params: OmniDiffusionSamplingParams) -> OmniDiffusionRequest | StepRequestState:
    """A request exposes sampling_params; a runner step state exposes sampling."""
    if holder == "request":
        return OmniDiffusionRequest(prompt="scheduled request", request_id="carrier", sampling_params=params)
    return StepRequestState(request_id="carrier", sampling=params)


@pytest.mark.parametrize("holder", ["request", "state"])
@pytest.mark.parametrize(
    ("request_schedule", "expected"),
    [
        (None, _ranges((0, 2, "sparse"))),
        ([{"start": 1, "end": 3, "profile": "dense"}], _ranges((1, 3, "dense"))),
    ],
    ids=["inherit", "replace"],
)
def test_require_request_attention_schedule_fits_checks_the_resolved_request_schedule(
    holder, request_schedule, expected
):
    carrier = _carrier(holder, OmniDiffusionSamplingParams(attention_schedule=request_schedule))
    od_config = OmniDiffusionConfig(diffusion_attention_schedule=_service())

    assert require_request_attention_schedule_fits(carrier, od_config, 3) == expected
    with pytest.raises(InvalidAttentionScheduleError, match="exceeds total_steps=1"):
        require_request_attention_schedule_fits(carrier, od_config, 1)


@pytest.mark.parametrize("holder", ["request", "state"])
@pytest.mark.parametrize(
    ("request_schedule", "service"),
    [([], _service()), (None, None)],
    ids=["request-disables", "no-service-schedule"],
)
def test_require_request_attention_schedule_fits_skips_an_unscheduled_request(holder, request_schedule, service):
    # The service default [0, 2) does not fit a 1-step sequence; an unscheduled request is not checked.
    carrier = _carrier(holder, OmniDiffusionSamplingParams(attention_schedule=request_schedule))
    od_config = OmniDiffusionConfig(diffusion_attention_schedule=service)

    assert require_request_attention_schedule_fits(carrier, od_config, 1) == ()


# request_denoise_progress


class _RecordingPagedRuntime:
    """Stands in for the runner-owned paged runtime and records each ensure_active call."""

    def __init__(self):
        self.activated: list[int] = []

    def ensure_active(self, step_idx):
        self.activated.append(step_idx)


@pytest.mark.parametrize("schedule", [None, ()], ids=["unbound", "disabled"])
def test_request_denoise_progress_publishes_without_activating_an_unscheduled_context(schedule):
    with set_forward_context(), bind_attention_schedule(schedule, denoise=False):
        ctx = get_forward_context()
        with request_denoise_progress(2, 6, timestep=torch.tensor(0.5)):
            assert _progress(ctx) == (2, 0.5, 6, False)
            assert type(ctx.denoise_timestep) is float
        assert _progress(ctx) == (None, None, None, False)


@pytest.mark.parametrize(
    ("outer_published", "fail"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["outer-unpublished-exit", "outer-unpublished-exception", "outer-published-exit", "outer-published-exception"],
)
def test_request_denoise_progress_restores_the_previous_progress(outer_published, fail):
    runtime = _RecordingPagedRuntime()
    with (
        set_forward_context(paged_kv_runtime=runtime),
        bind_attention_schedule(_ranges((0, None, "sparse")), denoise=False),
    ):
        ctx = get_forward_context()
        if outer_published:
            # A batch-level publish made before the per-request scope.
            set_forward_context_denoise_step_idx(1)
            set_forward_context_denoise_timestep(0.9)
            set_forward_context_denoise_total_steps(7)
        outer = _progress(ctx)
        activated_before = list(runtime.activated)
        raises = pytest.raises(RuntimeError, match="request forward failed") if fail else nullcontext()

        with raises:
            with request_denoise_progress(4, 5, timestep=0.25):
                assert _progress(ctx) == (4, 0.25, 5, True)
                if fail:
                    raise RuntimeError("request forward failed")

        assert _progress(ctx) == outer
    # Publishing the request's step activates its pages. The restore assigns the saved fields
    # directly and does not call ensure_active again.
    assert runtime.activated == activated_before + [4]


def test_request_denoise_progress_requires_a_forward_context():
    assert is_forward_context_available() is False
    with pytest.raises(RuntimeError, match="requires an active forward context"):
        with request_denoise_progress(0, 1):
            pass


def test_sequential_request_scopes_select_their_own_profiles_without_leaking(attention_env):
    # Step mode: requests at different steps share one denoise_step call and are
    # evaluated one at a time, each under its own step, total and timestep.
    config = _make_config(schedule=_service())
    layer = attention_env.build(config)
    requests = [("req-a", 2, 6, 0.5), ("req-b", 5, 8, None), ("req-c", 4, 8, 0.1)]
    seen: list[tuple[Any, ...]] = []

    with _runner_scope(config, _ranges((2, 4, "sparse"), (5, None, "dense"))) as ctx:
        for request_id, step, total, timestep in requests:
            with request_denoise_progress(step, total, timestep=timestep):
                seen.append((request_id, *_progress(ctx), layer.effective_attention()[0]))
            # Between requests nothing is published, so attention runs the baseline.
            assert _progress(ctx) == (None, None, None, False)
            assert layer.effective_attention()[0] is layer.attention

    assert _names(layer, seen) == [
        ("req-a", 2, 0.5, 6, True, "sparse"),
        ("req-b", 5, None, 8, True, "dense"),
        ("req-c", 4, 0.1, 8, True, "base"),
    ]


# add_schedule_candidate_check: startup validation applies model-owned checks to every candidate.


def _loaded_model(layer):
    """A loaded-pipeline stand-in whose named_modules() reports the layer as transformer.attn1."""
    transformer = nn.Module()
    transformer.attn1 = layer
    root = nn.Module()
    root.transformer = transformer
    return root


def test_candidate_check_reason_fails_startup_validation(attention_env):
    config = _make_config(schedule=_service())
    layer = attention_env.build(config)
    reason = "the model always passes a padding mask"
    seen: list[Any] = []

    def check(record):
        seen.append(record)
        return reason if record.backend_cls.get_name() == "SDPA" else None

    layer.add_schedule_candidate_check(check)

    with pytest.raises(ValueError, match=r"profile 'sparse' selects backend SDPA") as excinfo:
        layer_mod.validate_attention_schedule_candidates(_loaded_model(layer), config)
    assert str(excinfo.value).endswith(f"cannot use on layer 'transformer.attn1': {reason}")
    # Profiles are validated in sorted order, so "dense", which the default never references, came first.
    assert len(seen) == 2
    assert seen[0] is layer._schedule_candidates["dense"]
    assert seen[1] is layer._schedule_candidates["sparse"]


def test_candidate_check_returning_none_passes_startup_validation(attention_env):
    config = _make_config(schedule=_service())
    layer = attention_env.build(config)
    seen: list[str] = []

    def check(record):
        seen.append(record.backend_cls.get_name())
        return None

    layer.add_schedule_candidate_check(check)

    assert layer_mod.validate_attention_schedule_candidates(_loaded_model(layer), config) == 2
    assert seen == ["FLASH_ATTN", "SDPA"]


def test_candidate_check_is_not_consulted_without_a_schedule(attention_env):
    config = _make_config(schedule=None)
    layer = attention_env.build(config)
    seen: list[Any] = []

    def check(record):
        seen.append(record)
        return "must not be consulted"

    layer.add_schedule_candidate_check(check)

    assert layer_mod.validate_attention_schedule_candidates(_loaded_model(layer), config) == 0
    assert seen == []


# A pipeline that publishes the step without a total stays rejected.


def test_lingbot_pipeline_stays_rejected_for_a_non_empty_schedule():
    # LingBot's DMD block publishes only the step index, with no total, so the pipeline does not
    # provide record_denoise_step and a scheduled request is rejected before denoise.
    from vllm_omni.diffusion.models.lingbot_world.pipeline import LingBotWorldCausalDMDPipeline

    pipeline = object.__new__(LingBotWorldCausalDMDPipeline)
    schedule = _ranges((0, 1, "sparse"))

    assert not issubclass(LingBotWorldCausalDMDPipeline, DenoiseProgressMixin)
    with pytest.raises(ValueError, match="publishes denoise progress"):
        require_denoise_progress_publisher(pipeline, schedule)
    # A disabled schedule passes on the same pipeline, and a generic publisher passes the same check.
    require_denoise_progress_publisher(pipeline, ())
    require_denoise_progress_publisher(_ToyPublisher(layer=None), schedule)
