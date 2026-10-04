# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contracts for request schedule binding, context restore, and layer selection.

These tests do not load a model. They pin the shared selection interface that
entry, scheduler, and runner code must use, and drive the real runner request
and step paths with fake pipelines that publish denoise progress to a layer
with prepared candidates.
"""

import dataclasses
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_omni.diffusion.attention.layer as layer_mod
import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from tests.diffusion.attention.test_attention_schedule_candidates import (
    _fake_backend,
    _fake_resolve,
    _FakeImpl,
    _make_config,
)
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
    InvalidAttentionScheduleError,
    require_denoise_progress_publisher,
    resolve_attention_schedule,
    resolve_batch_attention_schedule,
    validate_request_attention_schedule,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionScheduleConfig,
    AttentionSpec,
    DiffusionOutput,
    DiffusionRequestAbortedError,
)
from vllm_omni.diffusion.forward_context import (
    DenoiseProgressMixin,
    bind_attention_schedule,
    get_forward_context,
    is_forward_context_available,
    override_paged_kv_adapter,
    set_forward_context,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler, build_request_batch_sampling_params_key
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.errors import client_error_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, absorb_attention_schedule_extra_args

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


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


def test_resolve_does_not_mutate_service_default_or_sibling():
    service = _service()
    request_list = [{"start": 1, "end": 3, "profile": "dense"}]
    sibling = [{"start": 2, "end": 4, "profile": "sparse"}]

    resolved = resolve_attention_schedule(request_list, service.default, profiles=service.profiles)
    resolved_sibling = resolve_attention_schedule(sibling, service.default, profiles=service.profiles)
    inherited = resolve_attention_schedule(None, service.default, profiles=service.profiles)
    # Mutating the caller's list after resolution must not reach the resolved schedule.
    request_list[0]["profile"] = "sparse"
    request_list.append({"start": 3, "end": 4, "profile": "sparse"})

    assert resolved == (AttentionScheduleRange(start=1, end=3, profile="dense"),)
    assert resolved_sibling == (AttentionScheduleRange(start=2, end=4, profile="sparse"),)
    assert inherited == (AttentionScheduleRange(start=0, end=2, profile="sparse"),)
    # Compared by value: the default is still the configured range after three resolutions.
    assert service.default == (AttentionScheduleRange(start=0, end=2, profile="sparse"),)


def test_bind_restores_context_after_exception():
    with set_forward_context():
        outer = get_forward_context()
        assert outer.attention_schedule is None
        try:
            with bind_attention_schedule((("x",),), denoise=True):
                assert get_forward_context().attention_schedule_denoise_active is True
                raise RuntimeError("cancelled")
        except RuntimeError:
            pass
        assert get_forward_context() is outer
        assert outer.attention_schedule is None
        assert outer.attention_schedule_denoise_active is False


def test_layer_uses_baseline_outside_denoise(attention_env):
    service = _service()
    layer = attention_env.build(_make_config(schedule=service))
    with set_forward_context(), bind_attention_schedule(service.default, denoise=False):
        impl, _backend, _spec = layer.effective_attention()
    assert impl is layer.attention


def test_layer_fails_when_denoise_step_is_missing(attention_env):
    service = _service()
    layer = attention_env.build(_make_config(schedule=service))
    with set_forward_context(), bind_attention_schedule(service.default, denoise=True):
        with pytest.raises(RuntimeError, match="denoise_step_idx"):
            layer.effective_attention()


def test_layer_selects_prepared_candidate_and_gap_uses_baseline(attention_env):
    service = _service()
    layer = attention_env.build(_make_config(schedule=service))
    with set_forward_context(denoise_step_idx=0), bind_attention_schedule(service.default, denoise=True):
        get_forward_context().total_denoise_steps = 4
        impl, backend, _spec = layer.effective_attention()
    assert impl is layer._schedule_candidates["sparse"].impl
    assert backend is layer._schedule_candidates["sparse"].backend_cls

    with set_forward_context(denoise_step_idx=3), bind_attention_schedule(service.default, denoise=True):
        get_forward_context().total_denoise_steps = 4
        impl, _backend, _spec = layer.effective_attention()
    assert impl is layer.attention


def _recording_forward(calls, name):
    def forward(query, key, value, attn_metadata=None):
        calls.append(name)
        return query

    return forward


def test_layer_forward_runs_the_selected_implementation(attention_env, monkeypatch):
    service = _service()
    config = _make_config(schedule=service)
    layer = attention_env.build(config)
    calls: list[str] = []
    monkeypatch.setattr(layer.attention, "forward", _recording_forward(calls, "base"))
    for name, record in layer._schedule_candidates.items():
        monkeypatch.setattr(record.impl, "forward", _recording_forward(calls, name))
    query = torch.zeros(1, 2, 4, 64)

    for step in (0, 3):
        with (
            set_forward_context(omni_diffusion_config=config, denoise_step_idx=step),
            bind_attention_schedule(service.default, denoise=True),
        ):
            get_forward_context().total_denoise_steps = 4
            layer(query, query, query)

    assert calls == ["sparse", "base"]


@contextmanager
def _denoise_context(config, schedule, *, step, total_steps):
    """Forward context for one denoise step with ``schedule`` bound, as the runner and pipeline set it."""
    with (
        set_forward_context(omni_diffusion_config=config, denoise_step_idx=step),
        bind_attention_schedule(schedule, denoise=True),
    ):
        get_forward_context().total_denoise_steps = total_steps
        yield


def _automatic_flash_resolve(**kwargs):
    """Like _fake_resolve, but an automatic selection resolves to FLASH_ATTN, as on CUDA."""
    backend, spec = _fake_resolve(**kwargs)
    return (_fake_backend("FLASH_ATTN"), None) if spec is None else (backend, spec)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (AttentionConfig(), "fallback"),
        (AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")), "candidate"),
        (AttentionConfig(default=AttentionSpec(backend="SDPA")), "candidate"),
    ],
    ids=["automatic-flash", "explicit-flash", "explicit-other"],
)
def test_fp32_fallback_follows_the_selected_candidate(attention_env, monkeypatch, profile, expected):
    # The FP32 SDPA fallback is decided from the selected candidate, not from the baseline. On an automatic
    # FLASH_ATTN baseline, an explicitly selected candidate runs its own implementation on its scheduled
    # step, and a candidate that is itself an automatic FLASH_ATTN choice falls back to SDPA.
    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", _automatic_flash_resolve)
    service = AttentionScheduleConfig(
        profiles={"picked": profile}, default=[{"start": 0, "end": 1, "profile": "picked"}]
    )
    config = _make_config(schedule=service)
    layer = attention_env.build(config, allow_fp32_fallback=True)
    calls: list[str] = []
    monkeypatch.setattr(layer.attention, "forward", _recording_forward(calls, "base"))
    monkeypatch.setattr(layer.sdpa_fallback, "forward", _recording_forward(calls, "fallback"))
    monkeypatch.setattr(layer._schedule_candidates["picked"].impl, "forward", _recording_forward(calls, "candidate"))
    # Only the dispatch predicate is simulated; every tensor stays on CPU.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    query = torch.zeros(1, 2, 4, 64)

    for step in (0, 1):
        with _denoise_context(config, service.default, step=step, total_steps=2):
            layer(query, query, query)

    # Step 0 runs the picked profile. Step 1 is a gap, where the automatic FLASH_ATTN baseline falls back.
    assert calls == [expected, "fallback"]


def test_mask_check_reads_the_selected_candidate_spec_even_when_it_is_none(attention_env, monkeypatch):
    # The fake backend accepts a mask only without a spec, so the check shows which spec it read.
    # The baseline is explicit (spec set); the implicit candidate has spec None and must not fall
    # back to the baseline spec.
    monkeypatch.setattr(
        _fake_backend("PLATFORM_DEFAULT"), "supports_attention_mask", staticmethod(lambda spec: spec is None)
    )
    schedule = AttentionScheduleConfig(
        profiles={"implicit": AttentionConfig()},
        default=[{"start": 0, "end": 1, "profile": "implicit"}],
    )
    baseline = AttentionConfig(default=AttentionSpec(backend="PLATFORM_DEFAULT"))
    config = _make_config(baseline=baseline, schedule=schedule)
    layer = attention_env.build(config)
    assert layer.attn_spec is not None
    assert layer._schedule_candidates["implicit"].spec is None
    query = torch.zeros(1, 2, 4, 64)
    metadata = AttentionMetadata(attn_mask=torch.ones(1, 2, dtype=torch.bool))

    with _denoise_context(config, schedule.default, step=0, total_steps=2):
        layer(query, query, query, metadata)

    # Step 1 is a gap, so the explicit baseline runs and its spec rejects the mask.
    with _denoise_context(config, schedule.default, step=1, total_steps=2):
        with pytest.raises(ValueError, match="does not support attn_mask"):
            layer(query, query, query, metadata)


class _PagedAdapter:
    """Records page-table preparation; the layer then calls forward_paged with the returned context."""

    def __init__(self):
        self.prepared: list[str] = []

    def prepare_layer_context(self, prefix, query, key, value, *, omni_attn_metadata=None):
        del query, key, value, omni_attn_metadata
        self.prepared.append(prefix)
        return ("paged-context", prefix)


def test_paged_forward_applies_the_step_check_before_preparing_the_page_table(attention_env, monkeypatch):
    # forward_paged runs the baseline native implementation, so startup keeps paged candidates
    # identical to the baseline. The step, range and profile checks must still run on this path.
    monkeypatch.setattr(FlashAttentionBackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    schedule = AttentionScheduleConfig(
        profiles={"same": AttentionConfig()},
        default=[{"start": 0, "end": None, "profile": "same"}],
    )
    config = _make_config(schedule=schedule, kv_mode=layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER)
    layer = attention_env.build(config, paged_kv_cache_role="primary")
    paged_calls: list[Any] = []

    def forward_paged(context):
        paged_calls.append(context)
        return torch.zeros(1, 2, 4, 64)

    monkeypatch.setattr(layer.attention, "forward_paged", forward_paged, raising=False)
    adapter = _PagedAdapter()
    query = torch.zeros(1, 2, 4, 64)

    with (
        set_forward_context(omni_diffusion_config=config),
        bind_attention_schedule(schedule.default, denoise=True),
        override_paged_kv_adapter(adapter),
    ):
        with pytest.raises(RuntimeError, match="denoise_step_idx"):
            layer(query, query, query)
    assert adapter.prepared == []
    assert paged_calls == []

    with _denoise_context(config, schedule.default, step=0, total_steps=2), override_paged_kv_adapter(adapter):
        layer(query, query, query)
    assert adapter.prepared == [layer.prefix]
    assert paged_calls == [("paged-context", layer.prefix)]


class _RingRunner:
    def __init__(self):
        self.calls = 0

    def run_attention(self, query, key, value, attn_metadata, *, softmax_scale, causal):
        del key, value, attn_metadata, softmax_scale, causal
        self.calls += 1
        return query


def test_ring_attention_applies_the_step_check_before_delegating(attention_env):
    # The ring runner is bound to the baseline backend, so startup keeps ring candidates identical
    # to the baseline. The step, range and profile checks must still run before delegating.
    schedule = AttentionScheduleConfig(
        profiles={"same": AttentionConfig()},
        default=[{"start": 0, "end": 2, "profile": "same"}],
    )
    config = _make_config(schedule=schedule)
    # Built with ring_degree=1 because a real ring runner needs a distributed group; the runner
    # is replaced by a fake after construction.
    layer = attention_env.build(config)
    runner = _RingRunner()
    layer.use_ring = True
    layer.ring_runner = runner
    query = torch.zeros(1, 2, 4, 64)

    with set_forward_context(omni_diffusion_config=config), bind_attention_schedule(schedule.default, denoise=True):
        with pytest.raises(RuntimeError, match="denoise_step_idx"):
            layer._run_ring_attention(query, query, query, None)
    assert runner.calls == 0

    with _denoise_context(config, schedule.default, step=3, total_steps=2):
        with pytest.raises(ValueError, match="must be less than total_steps"):
            layer._run_ring_attention(query, query, query, None)
    assert runner.calls == 0

    with _denoise_context(config, schedule.default, step=1, total_steps=2):
        layer._run_ring_attention(query, query, query, None)
    assert runner.calls == 1


def test_batch_key_separates_ranges_and_matches_equal_ranges():
    left = SimpleNamespace(
        sampling_params=OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": 2, "profile": "sparse"}]),
        batch_compatibility_key=None,
    )
    right = SimpleNamespace(
        sampling_params=OmniDiffusionSamplingParams(attention_schedule=[{"start": 2, "end": 4, "profile": "sparse"}]),
        batch_compatibility_key=None,
    )
    same = SimpleNamespace(
        sampling_params=OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": 2, "profile": "sparse"}]),
        batch_compatibility_key=None,
    )

    assert build_request_batch_sampling_params_key(left) != build_request_batch_sampling_params_key(right)
    assert build_request_batch_sampling_params_key(left) == build_request_batch_sampling_params_key(same)


def test_validation_rejects_unknown_profile():
    service = _service()
    request = SimpleNamespace(
        sampling_params=OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": 1, "profile": "missing"}])
    )
    od_config = SimpleNamespace(diffusion_attention_schedule=service)

    with pytest.raises(ValueError, match="unknown profile"):
        validate_request_attention_schedule(request, od_config)


def test_video_extra_params_match_typed_schedule():
    typed = OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": 2, "profile": "sparse"}])
    via_extra = OmniDiffusionSamplingParams()
    via_extra.extra_args = {"attention_schedule": [{"start": 0, "end": 2, "profile": "sparse"}], "other": 1}
    absorb_attention_schedule_extra_args(via_extra)

    assert via_extra.attention_schedule == typed.attention_schedule
    assert "attention_schedule" not in via_extra.extra_args
    assert via_extra.extra_args["other"] == 1


def test_pipeline_without_progress_publisher_is_rejected_before_denoise():
    ran: list[str] = []

    def denoise():
        ran.append("denoise")

    pipeline = SimpleNamespace(denoise_step=denoise)
    with pytest.raises(ValueError, match="publishes denoise progress"):
        require_denoise_progress_publisher(
            pipeline,
            resolve_attention_schedule(
                [{"start": 0, "end": 1, "profile": "sparse"}],
                None,
                profiles={"sparse"},
            ),
        )
    assert ran == []


def test_layer_rejects_step_past_total(attention_env):
    service = _service()
    layer = attention_env.build(_make_config(schedule=service))
    with set_forward_context(denoise_step_idx=4), bind_attention_schedule(service.default, denoise=True):
        get_forward_context().total_denoise_steps = 4
        with pytest.raises(ValueError, match="step_index=4"):
            layer.effective_attention()


def test_nested_bind_restores_outer_schedule():
    outer = (("outer",),)
    inner = (("inner",),)
    with set_forward_context(), bind_attention_schedule(outer, denoise=False):
        with bind_attention_schedule(inner, denoise=True):
            assert get_forward_context().attention_schedule == inner
        ctx = get_forward_context()
        assert ctx.attention_schedule == outer
        assert ctx.attention_schedule_denoise_active is False
    assert is_forward_context_available() is False


def test_sampling_clone_keeps_schedule_and_does_not_alias_source():
    source = OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": 2, "profile": "sparse"}])
    cloned = source.clone()
    assert cloned.attention_schedule == source.attention_schedule
    assert cloned.attention_schedule is not source.attention_schedule


def test_scheduler_admits_valid_request_and_rejects_invalid():
    scheduler = RequestScheduler()
    scheduler.od_config = SimpleNamespace(diffusion_attention_schedule=_service())
    valid = OmniDiffusionRequest(
        prompt="ok",
        request_id="ok",
        sampling_params=OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": 1, "profile": "sparse"}]),
    )
    invalid = OmniDiffusionRequest(
        prompt="bad",
        request_id="bad",
        sampling_params=OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": 1, "profile": "missing"}]),
    )

    assert scheduler.add_request(valid) == "ok"
    with pytest.raises(ValueError, match="unknown profile") as excinfo:
        scheduler.add_request(invalid)
    assert "bad" not in scheduler._request_states
    # A client input error must reach HTTP as 400, not as a server error.
    assert client_error_metadata(excinfo.value) == (400, "invalid_attention_schedule")


def _batch_key(params):
    state = SimpleNamespace(sampling_params=params, batch_compatibility_key=None)
    return build_request_batch_sampling_params_key(state)


def _late_schedule_request(request_id, schedule, *, typed=None):
    """Params whose extra_args schedule is written after construction, then cloned like the inline client."""
    params = OmniDiffusionSamplingParams(attention_schedule=typed)
    params.extra_args = {"attention_schedule": schedule, "other": 1}
    return OmniDiffusionRequest(prompt=request_id, request_id=request_id, sampling_params=params.clone())


_DENSE_1_2 = [{"start": 1, "end": 2, "profile": "dense"}]


@pytest.mark.parametrize(
    ("schedule", "typed", "expected"),
    [
        ([], None, ()),
        (_DENSE_1_2, None, (AttentionScheduleRange(start=1, end=2, profile="dense"),)),
        # null means the same as an omitted key, so the typed value stays.
        (None, _DENSE_1_2, (AttentionScheduleRange(start=1, end=2, profile="dense"),)),
    ],
    ids=["disable", "replace", "null-keeps-typed"],
)
def test_scheduler_absorbs_schedule_written_into_extra_args_after_construction(schedule, typed, expected):
    # clone() does not re-run __post_init__, so without admission-time absorption the typed
    # field keeps its old value and the request silently runs a schedule it did not ask for.
    scheduler = RequestScheduler()
    scheduler.od_config = SimpleNamespace(diffusion_attention_schedule=_service())
    request = _late_schedule_request("late", schedule, typed=typed)
    assert "attention_schedule" in request.sampling_params.extra_args

    assert scheduler.add_request(request) == "late"

    admitted = scheduler._request_states["late"].req.sampling_params
    assert admitted.attention_schedule == expected
    assert admitted.extra_args == {"other": 1}
    # The batch key reads the typed field, so the schedule is the only field that separates it from an
    # admitted request that inherits. Both go through admission, which fills the same guidance defaults.
    inheriting = OmniDiffusionRequest(
        prompt="inherit", request_id="inherit", sampling_params=OmniDiffusionSamplingParams(extra_args={"other": 1})
    )
    assert scheduler.add_request(inheriting) == "inherit"
    inherited_key = _batch_key(scheduler._request_states["inherit"].req.sampling_params)
    assert inherited_key.attention_schedule is None
    assert _batch_key(admitted).attention_schedule == expected
    assert dataclasses.replace(_batch_key(admitted), attention_schedule=None) == inherited_key


@pytest.mark.parametrize(
    ("schedule", "typed", "message"),
    [
        ([{"start": 0, "end": 1, "profile": "missing"}], None, "unknown profile"),
        ([{"start": 0, "end": 2, "profile": "sparse"}, {"start": 1, "end": 3, "profile": "dense"}], None, "overlap"),
        ([{"start": True, "end": 2, "profile": "sparse"}], None, "integer"),
        ([{"start": 0, "end": 1, "profile": "dense"}], [{"start": 1, "end": 2, "profile": "dense"}], "conflicting"),
    ],
    ids=["unknown-profile", "overlap", "bool-step", "conflict"],
)
def test_scheduler_rejects_invalid_late_schedule_as_client_error(schedule, typed, message):
    scheduler = RequestScheduler()
    scheduler.od_config = SimpleNamespace(diffusion_attention_schedule=_service())

    with pytest.raises(InvalidAttentionScheduleError, match=message) as excinfo:
        scheduler.add_request(_late_schedule_request("bad", schedule, typed=typed))

    assert "bad" not in scheduler._request_states
    assert not scheduler._waiting
    assert client_error_metadata(excinfo.value) == (400, "invalid_attention_schedule")


def test_different_ranges_are_not_scheduled_together_and_ids_stay_put():
    scheduler = RequestScheduler()
    scheduler.max_num_running_reqs = 2
    scheduler.od_config = SimpleNamespace(diffusion_attention_schedule=_service())
    first = OmniDiffusionRequest(
        prompt="a",
        request_id="req-a",
        sampling_params=OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": 1, "profile": "sparse"}]),
    )
    second = OmniDiffusionRequest(
        prompt="b",
        request_id="req-b",
        sampling_params=OmniDiffusionSamplingParams(attention_schedule=[{"start": 1, "end": 2, "profile": "dense"}]),
    )
    assert scheduler.add_request(first) == "req-a"
    assert scheduler.add_request(second) == "req-b"

    output = scheduler.schedule()
    assert [item.request_id for item in output.scheduled_new_reqs] == ["req-a"]
    assert scheduler._waiting[0] == "req-b"
    assert scheduler._request_states["req-a"].req.request_id == "req-a"
    assert scheduler._request_states["req-b"].req.sampling_params.attention_schedule[0].profile == "dense"


def test_batch_schedule_requires_one_identity():
    first = OmniDiffusionSamplingParams(attention_schedule=[{"start": 0, "end": 1, "profile": "sparse"}])
    second = OmniDiffusionSamplingParams(attention_schedule=[{"start": 1, "end": 2, "profile": "sparse"}])
    states = [SimpleNamespace(sampling=first), SimpleNamespace(sampling=second)]
    with pytest.raises(ValueError, match="batch"):
        resolve_batch_attention_schedule(states, SimpleNamespace(diffusion_attention_schedule=_service()))


# Runner integration: the real runner binds the schedule, a fake pipeline
# publishes denoise progress, and a layer with prepared candidates selects.


@pytest.fixture
def runner_platform(monkeypatch):
    platform = model_runner_module.current_omni_platform
    monkeypatch.setattr(platform, "is_available", lambda: False)
    monkeypatch.setattr(platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(platform, "max_memory_allocated", lambda: 0)


def _runner_config(schedule):
    config = _make_config(schedule=schedule)
    config.parallel_config.use_hsdp = False
    config.cache_backend = None
    config.enable_cache_dit_summary = False
    config.streaming_output = False
    return config


def _make_runner(pipeline, config):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = None
    runner.od_config = config
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = None
    return runner


def _request(request_id, schedule=None, steps=4):
    return OmniDiffusionRequest(
        prompt=f"prompt-{request_id}",
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=steps, attention_schedule=schedule),
    )


def _new_requests(*requests, finished=()):
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(request_id=request.request_id, req=request) for request in requests],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(finished),
        num_running_reqs=len(requests),
        num_waiting_reqs=0,
    )


def _cached_request(request_id, step_id, finished=()):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(request_ids=[request_id]),
        finished_req_ids=set(finished),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _names(layer, selected):
    """Replace each recorded implementation with its profile name, or "base"."""
    names = {id(layer.attention): "base"}
    names.update({id(record.impl): name for name, record in layer._schedule_candidates.items()})
    return [(*entry[:-1], names[id(entry[-1])]) for entry in selected]


class _PublishingRequestPipeline(DenoiseProgressMixin):
    """Request mode: encode, publish every denoise step, then decode."""

    supports_request_batch = True

    def __init__(self, layer, *, fail_step=None, error=None):
        self.layer = layer
        self.fail_step = fail_step
        self.error = error
        self.selected: list[tuple[Any, ...]] = []
        self.bound_schedules: list[Any] = []

    def _select(self, phase):
        self.selected.append((phase, self.layer.effective_attention()[0]))

    def forward(self, batch):
        self.bound_schedules.append(get_forward_context().attention_schedule)
        total = batch.requests[0].sampling_params.num_inference_steps
        self._select("encode")
        for step in range(total):
            self.record_denoise_step(step, total_steps=total)
            if step == self.fail_step:
                raise self.error
            self._select(step)
        self.record_denoise_step(None)
        self._select("decode")
        return [DiffusionOutput(output={"request_id": request.request_id}) for request in batch.requests]


class _SilentRequestPipeline:
    """Request mode without a progress publisher."""

    supports_request_batch = True

    def __init__(self):
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        return [DiffusionOutput(output={"request_id": request.request_id}) for request in batch.requests]


class _StepPipelineBase:
    supports_step_execution = True

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.prepare_calls = 0
        self.denoise_calls = 0

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1
        state.timesteps = [torch.tensor(float(step)) for step in range(self.total_steps)]
        state.latents = torch.tensor([0.0])
        state.prompt_embeds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        return state

    def denoise_step(self, input_batch, *, states=None, **kwargs):
        del states, kwargs
        self.denoise_calls += 1
        return torch.full_like(input_batch.prompt_embeds, fill_value=0.5)

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        return DiffusionOutput(output=torch.tensor([float(state.step_index)]))


class _SilentStepPipeline(_StepPipelineBase):
    """Step mode without a progress publisher."""


class _PublishingStepPipeline(DenoiseProgressMixin, _StepPipelineBase):
    """Step mode: each denoise call publishes its request's step before selecting."""

    def __init__(self, layer, total_steps, *, fail_at=None):
        super().__init__(total_steps)
        self.layer = layer
        self.fail_at = fail_at
        self.selected: list[tuple[Any, ...]] = []

    def _select(self, state, phase):
        self.selected.append((state.request_id, phase, self.layer.effective_attention()[0]))

    def prepare_encode(self, state, **kwargs):
        self._select(state, "encode")
        return super().prepare_encode(state, **kwargs)

    def denoise_step(self, input_batch, *, states=None, **kwargs):
        (state,) = states
        self.record_denoise_step(state.step_index, total_steps=self.total_steps)
        if (state.request_id, state.step_index) == self.fail_at:
            raise RuntimeError("denoise failed")
        self._select(state, state.step_index)
        return super().denoise_step(input_batch, states=states, **kwargs)

    def post_decode(self, state, **kwargs):
        self._select(state, "decode")
        return super().post_decode(state, **kwargs)


def test_request_runner_applies_inherited_default_only_while_denoising(attention_env, runner_platform):
    service = _service()
    config = _runner_config(service)
    layer = attention_env.build(config)
    pipeline = _PublishingRequestPipeline(layer)
    runner = _make_runner(pipeline, config)

    output = DiffusionModelRunner.execute_model(runner, _request("req-a"))

    assert output.error is None
    assert output.output == {"request_id": "req-a"}
    assert pipeline.bound_schedules == [service.default]
    assert _names(layer, pipeline.selected) == [
        ("encode", "base"),
        (0, "sparse"),
        (1, "sparse"),
        (2, "base"),
        (3, "base"),
        ("decode", "base"),
    ]
    assert is_forward_context_available() is False


@pytest.mark.parametrize(
    ("schedule", "bound", "expected"),
    [
        (
            [{"start": 1, "end": 3, "profile": "dense"}],
            (AttentionScheduleRange(start=1, end=3, profile="dense"),),
            ["base", "base", "dense", "dense", "base", "base"],
        ),
        # A disabled request leaves the forward context unbound.
        ([], None, ["base"] * 6),
    ],
    ids=["replace", "disable"],
)
def test_request_runner_override_replaces_or_disables_default(
    attention_env, runner_platform, schedule, bound, expected
):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    pipeline = _PublishingRequestPipeline(layer)
    runner = _make_runner(pipeline, config)

    DiffusionModelRunner.execute_model(runner, _request("req-a", schedule=schedule))

    assert pipeline.bound_schedules == [bound]
    assert [name for _phase, name in _names(layer, pipeline.selected)] == expected


def test_request_runner_batch_keeps_request_order_and_rejects_mixed_schedules(attention_env, runner_platform):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    pipeline = _PublishingRequestPipeline(layer)
    runner = _make_runner(pipeline, config)
    same = [{"start": 0, "end": 1, "profile": "dense"}]

    output = DiffusionModelRunner.execute_model_batch(
        runner, _new_requests(_request("req-a", schedule=same), _request("req-b", schedule=same)), config
    )

    assert [item.request_id for item in output.runner_outputs] == ["req-a", "req-b"]
    assert [item.result.output["request_id"] for item in output.runner_outputs] == ["req-a", "req-b"]
    assert _names(layer, pipeline.selected)[:3] == [("encode", "base"), (0, "dense"), (1, "base")]

    mixed = _new_requests(
        _request("req-c", schedule=same),
        _request("req-d", schedule=[{"start": 1, "end": 2, "profile": "dense"}]),
    )
    with pytest.raises(ValueError, match="must be identical"):
        DiffusionModelRunner.execute_model_batch(runner, mixed, config)
    assert len(pipeline.bound_schedules) == 1


def test_request_runner_rejects_schedule_without_progress_publisher_before_forward(runner_platform):
    pipeline = _SilentRequestPipeline()
    runner = _make_runner(pipeline, _runner_config(_service()))

    with pytest.raises(ValueError, match="publishes denoise progress"):
        DiffusionModelRunner.execute_model(runner, _request("req-a"))
    assert pipeline.forward_calls == 0

    assert DiffusionModelRunner.execute_model(runner, _request("req-b", schedule=[])).error is None
    assert pipeline.forward_calls == 1


def test_request_runner_rejects_schedule_with_cache_backend_before_forward(attention_env, runner_platform):
    # A cache backend reuses or skips transformer evaluations across steps, so the scheduled
    # attention would not be what ran.
    config = _runner_config(_service())
    config.cache_backend = "tea_cache"
    layer = attention_env.build(config)
    pipeline = _PublishingRequestPipeline(layer)
    runner = _make_runner(pipeline, config)

    with pytest.raises(ValueError, match="cache_backend='tea_cache'"):
        DiffusionModelRunner.execute_model(runner, _request("req-a"))
    assert pipeline.bound_schedules == []

    # A request that disables the schedule still runs on the same service.
    assert DiffusionModelRunner.execute_model(runner, _request("req-b", schedule=[])).error is None
    assert pipeline.bound_schedules == [None]


@pytest.mark.parametrize(
    "error",
    [RuntimeError("denoise failed"), DiffusionRequestAbortedError("Request req-a aborted.")],
    ids=["exception", "cancel"],
)
def test_request_runner_failure_or_cancel_does_not_leak_into_next_request(attention_env, runner_platform, error):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    failing = _PublishingRequestPipeline(layer, fail_step=1, error=error)
    runner = _make_runner(failing, config)
    request = _request("req-a", schedule=[{"start": 0, "end": 4, "profile": "dense"}])

    if isinstance(error, DiffusionRequestAbortedError):
        # The runner reports a cancel as an aborted output instead of raising.
        assert DiffusionModelRunner.execute_model(runner, request).aborted is True
    else:
        with pytest.raises(RuntimeError, match="denoise failed"):
            DiffusionModelRunner.execute_model(runner, request)
    assert _names(layer, failing.selected) == [("encode", "base"), (0, "dense")]
    assert is_forward_context_available() is False

    healthy = _PublishingRequestPipeline(layer)
    runner.pipeline = healthy
    DiffusionModelRunner.execute_model(runner, _request("req-b"))
    assert [name for _phase, name in _names(layer, healthy.selected)] == [
        "base",
        "sparse",
        "sparse",
        "base",
        "base",
        "base",
    ]


def test_step_runner_selects_published_steps_and_keeps_encode_decode_on_baseline(attention_env, runner_platform):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    pipeline = _PublishingStepPipeline(layer, total_steps=4)
    runner = _make_runner(pipeline, config)
    request = _request("req-a", schedule=[{"start": 1, "end": 3, "profile": "dense"}])

    output = DiffusionModelRunner.execute_stepwise(runner, _new_requests(request))
    for step_id in range(1, 4):
        assert is_forward_context_available() is False
        output = DiffusionModelRunner.execute_stepwise(runner, _cached_request("req-a", step_id))

    final = output.get_request_output("req-a")
    assert final is not None and final.finished is True
    assert final.result is not None and final.result.error is None
    assert _names(layer, pipeline.selected) == [
        ("req-a", "encode", "base"),
        ("req-a", 0, "base"),
        ("req-a", 1, "dense"),
        ("req-a", 2, "dense"),
        ("req-a", 3, "base"),
        ("req-a", "decode", "base"),
    ]
    assert runner.state_cache == {}
    assert is_forward_context_available() is False


def test_step_runner_interleaved_requests_keep_their_own_schedules(attention_env, runner_platform):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    pipeline = _PublishingStepPipeline(layer, total_steps=3)
    runner = _make_runner(pipeline, config)
    first = _request("req-a", schedule=[{"start": 0, "end": 1, "profile": "dense"}], steps=3)
    second = _request("req-b", schedule=[{"start": 1, "end": None, "profile": "sparse"}], steps=3)

    DiffusionModelRunner.execute_stepwise(runner, _new_requests(first))
    DiffusionModelRunner.execute_stepwise(runner, _new_requests(second))
    for step_id in range(1, 3):
        DiffusionModelRunner.execute_stepwise(runner, _cached_request("req-a", step_id))
        DiffusionModelRunner.execute_stepwise(runner, _cached_request("req-b", step_id))

    names = _names(layer, pipeline.selected)
    assert [entry[1:] for entry in names if entry[0] == "req-a"] == [
        ("encode", "base"),
        (0, "dense"),
        (1, "base"),
        (2, "base"),
        ("decode", "base"),
    ]
    assert [entry[1:] for entry in names if entry[0] == "req-b"] == [
        ("encode", "base"),
        (0, "base"),
        (1, "sparse"),
        (2, "sparse"),
        ("decode", "base"),
    ]
    assert runner.state_cache == {}


def test_step_runner_failure_in_one_request_leaves_the_other_unaffected(attention_env, runner_platform):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    pipeline = _PublishingStepPipeline(layer, total_steps=3, fail_at=("req-a", 1))
    runner = _make_runner(pipeline, config)
    first = _request("req-a", schedule=[{"start": 0, "end": 3, "profile": "dense"}], steps=3)
    second = _request("req-b", schedule=[], steps=3)

    DiffusionModelRunner.execute_stepwise(runner, _new_requests(first))
    DiffusionModelRunner.execute_stepwise(runner, _new_requests(second))
    with pytest.raises(RuntimeError, match="denoise failed"):
        DiffusionModelRunner.execute_stepwise(runner, _cached_request("req-a", 1))
    assert is_forward_context_available() is False
    # The engine reports req-a as finished; the next wave releases its state.
    DiffusionModelRunner.execute_stepwise(runner, _cached_request("req-b", 1, finished=["req-a"]))
    DiffusionModelRunner.execute_stepwise(runner, _cached_request("req-b", 2))

    names = _names(layer, pipeline.selected)
    assert [entry[1:] for entry in names if entry[0] == "req-a"] == [("encode", "base"), (0, "dense")]
    assert [entry[1:] for entry in names if entry[0] == "req-b"] == [
        ("encode", "base"),
        (0, "base"),
        (1, "base"),
        (2, "base"),
        ("decode", "base"),
    ]
    assert runner.state_cache == {}


def test_step_runner_rejects_schedule_without_progress_publisher_before_prepare(runner_platform):
    pipeline = _SilentStepPipeline(total_steps=2)
    runner = _make_runner(pipeline, _runner_config(_service()))

    with pytest.raises(ValueError, match="publishes denoise progress"):
        DiffusionModelRunner.execute_stepwise(runner, _new_requests(_request("req-a", steps=2)))
    assert (pipeline.prepare_calls, pipeline.denoise_calls) == (0, 0)

    # As with any failed wave, the engine reports req-a as finished and the next wave releases it.
    request = _request("req-b", schedule=[], steps=2)
    DiffusionModelRunner.execute_stepwise(runner, _new_requests(request, finished=["req-a"]))
    assert (pipeline.prepare_calls, pipeline.denoise_calls) == (1, 1)
    assert list(runner.state_cache) == ["req-b"]


# Startup: load_model rejects a service default that every inheriting request would fail.


class _StartupPublishingPipeline(DenoiseProgressMixin):
    """A loaded pipeline that publishes denoise progress."""


class _StartupSilentPipeline:
    """A loaded pipeline without a progress publisher."""


class _StartupCacheBackend:
    def enable(self, pipeline):
        del pipeline


class _StartupMemoryProfiler:
    consumed_memory = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


def _load_model(monkeypatch, pipeline, *, schedule, cache_backend=None, model_class_name="WanPipeline"):
    """Run the real load_model with the loader, memory profiler, offload and cache factory replaced."""

    class _Loader:
        def __init__(self, load_config, od_config=None):
            del load_config, od_config

        def load_model(self, **kwargs):
            del kwargs
            return pipeline

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        cache_backend=cache_backend,
        cache_config={},
        model_class_name=model_class_name,
        enforce_eager=True,
        streaming_output=False,
        diffusion_attention_schedule=schedule,
    )
    cache = _StartupCacheBackend() if cache_backend not in (None, "none") else None
    monkeypatch.setattr(model_runner_module, "LoadConfig", lambda: object())
    monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _Loader)
    monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _StartupMemoryProfiler)
    monkeypatch.setattr(model_runner_module, "enable_offload_backend", lambda od_config, pipe, device: (pipe, None))
    monkeypatch.setattr(model_runner_module, "get_cache_backend", lambda name, cache_config: cache)
    DiffusionModelRunner.load_model(runner)
    return runner


def _profiles_only_service():
    return AttentionScheduleConfig(profiles=_service().profiles, default=[])


def test_startup_rejects_default_schedule_on_pipeline_without_progress_publisher(monkeypatch):
    with pytest.raises(ValueError, match="publishes denoise progress"):
        _load_model(monkeypatch, _StartupSilentPipeline(), schedule=_service())


def test_startup_loads_pipeline_without_progress_publisher_when_no_schedule_is_configured(monkeypatch):
    pipeline = _StartupSilentPipeline()

    runner = _load_model(monkeypatch, pipeline, schedule=None)

    assert runner.pipeline is pipeline


def test_startup_rejects_profiles_on_pipeline_without_progress_publisher(monkeypatch):
    # With an empty default no request inherits a schedule, but every request that opts in would be
    # rejected, and under compile the scheduled layers would still call the eager boundary.
    with pytest.raises(ValueError, match="_StartupSilentPipeline never publishes denoise progress"):
        _load_model(monkeypatch, _StartupSilentPipeline(), schedule=_profiles_only_service())


def test_startup_loads_publishing_pipeline_when_default_is_empty(monkeypatch):
    # Requests may still opt in; each one is then checked per request before denoise.
    pipeline = _StartupPublishingPipeline()

    runner = _load_model(monkeypatch, pipeline, schedule=_profiles_only_service())

    assert runner.pipeline is pipeline


def test_startup_rejects_default_schedule_with_cache_backend(monkeypatch):
    with pytest.raises(ValueError, match="cache_backend='tea_cache'"):
        _load_model(monkeypatch, _StartupPublishingPipeline(), schedule=_service(), cache_backend="tea_cache")


def test_startup_rejects_profiles_with_cache_backend_when_default_is_empty(monkeypatch):
    # Every request that opts in would be rejected before denoise with the cache backend active.
    with pytest.raises(ValueError, match="cache_backend='tea_cache'"):
        _load_model(
            monkeypatch, _StartupPublishingPipeline(), schedule=_profiles_only_service(), cache_backend="tea_cache"
        )


def test_startup_loads_default_schedule_when_cache_backend_is_none_string(monkeypatch):
    # od_config stores "none" when no cache backend is configured; the startup check must accept it.
    pipeline = _StartupPublishingPipeline()

    runner = _load_model(monkeypatch, pipeline, schedule=_service(), cache_backend="none")

    assert runner.pipeline is pipeline
    assert runner.cache_backend is None
    assert runner.od_config.cache_backend == "none"


def test_startup_accepts_default_schedule_when_the_model_disables_cache_acceleration(monkeypatch):
    # NextStep11Pipeline is in _NO_CACHE_ACCELERATION, so load_model clears the cache backend
    # before the schedule check runs.
    runner = _load_model(
        monkeypatch,
        _StartupPublishingPipeline(),
        schedule=_service(),
        cache_backend="tea_cache",
        model_class_name="NextStep11Pipeline",
    )

    assert runner.od_config.cache_backend is None
    assert runner.cache_backend is None
