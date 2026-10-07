# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Scheduled attention under a real torch.compile.

A two-block toy transformer is compiled with the production ``regionally_compile`` helper and a
Dynamo backend that keeps every captured graph and counts its executions. Each block runs a linear
projection and ``torch.sin`` before a real ``Attention`` layer and a linear projection and
``torch.cos`` after it, so the two sides of the attention call can be told apart in the graphs.
The baseline and the prepared candidates compute scaled dot-product attention on CPU. The
TRTLLM_ATTN candidates subclass the production implementation, so ``__init__``, the skip config and
``_resolve_skip_factor`` (including the timestep gate) are production code. ``forward`` is replaced
by the SDPA math and passes ``key.shape[1]``, which equals the ``max_kv_len`` of ``forward_cuda`` for
the unpacked inputs used here. ``forward_cuda`` and the kernel are not exercised here.

Each compiled test records, per denoise step, which implementation ran, how often each compiled
graph ran, how many graphs exist and how many times Dynamo's frame converter was called. That count
is Dynamo's ``counters["frames"]["total"]``. It goes up once for each frame that reaches the
converter of ``torch.compile(fullgraph=False)``, including recompiles, failed compiles,
recompile-limit attempts and frames that Dynamo then skips. Frames on Dynamo's skip lists are not
counted, and fullgraph compiles use another converter that does not update it, so fullgraph runs
record no frame count. The count is needed because a frame whose graph is empty, such as a frame
that reaches the eager boundary before any tensor operation, never reaches the backend, so a
recompile of that frame changes no graph count. For each compiled model in the tests that expect a
compiled run, shapes, dtype, layout, grad mode, the compile settings and the candidate set are fixed
across its requests, and every code path outside the eager boundary runs in the first step. The
baseline and the candidates also return tensors with the same strides and of the same kind: outside
inference mode a view whose base has one shape, inside it a tensor that is not a view. So in those
tests a graph added or a frame compiled after the first step comes from forward-context state: the
step, the timestep, the total or the bound schedule. Requests run eight steps unless they set six,
so the published total changes between requests in some tests; unscheduled requests publish no
total unless they ask to.

Most tests compile with ``dynamic=False`` and run with grad mode off. The production-settings test
and the block-count test repeat their runs with ``dynamic=True``, the runner's default, and under
``torch.inference_mode()``, which the runner uses when neither HSDP nor distributed offload is on.
The frame check allows any number of compiles in the first step; the block-count test shows, under
the same four settings, that a second block compiles no frame there.

The tensor that the eager boundary returns is an input of the frame that resumes after the
attention call, and Dynamo guards properties of that input that do not depend on the step, such as
its strides. The copied-output test makes the approximate candidate return another tensor: one
with other strides than the baseline's, one with the baseline's strides that outside inference
mode is a view of a base with another shape, and one with the baseline's strides that is not a
view. It allows compiles in the first step that runs that candidate, and in no later step or
request. Each record lists the strides of the attention results that ran outside a compiled graph,
and the shape and strides of their base for those that are views.

The runs that the checks must reject (forced eager, Dynamo's recompile-limit fallback, a backend
failure with and without ``suppress_errors`` and a fullgraph compile of a scheduled layer) print
records in the same format. Each such record has a ``rejected`` field with the first line of the
check that rejected the run, or a ``raised`` field with the type of the exception the run raised.
Compilation on the target backend and Inductor is not covered here.
"""

import json
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo.utils import counters as dynamo_counters

import vllm_omni.diffusion.attention.layer as layer_mod
from vllm_omni.diffusion.attention.backends.trtllm_attn import TrtllmAttentionBackend, TrtllmAttentionImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.schedule import AttentionScheduleRange
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionScheduleConfig,
    AttentionSpec,
    OmniDiffusionConfig,
    SkipSoftmaxSpec,
)
from vllm_omni.diffusion.forward_context import (
    DenoiseProgressMixin,
    begin_scheduled_denoise,
    bind_attention_schedule,
    set_forward_context,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_BLOCKS = 2
_HEADS = 2
_HEAD_SIZE = 16
_SEQ = 8
_HIDDEN = _HEADS * _HEAD_SIZE


def _timesteps(count):
    """``count`` evenly spaced timesteps on a 1000-step training scale, from 1000 down."""
    return [1000.0 * (count - index) / count for index in range(count)]


# Eight steps, published as 1.0, 0.875, ..., 0.125. Some requests run six steps instead.
_TIMESTEPS = _timesteps(8)
_WEIGHT_SEED = 0
_LATENT_SEED = 1
# Dynamo settings that the expected graphs and fallbacks depend on; each record prints their values.
_DYNAMO_SETTINGS = ("nested_graph_breaks", "fail_on_recompile_limit_hit")

# One entry per attention call that ran outside a compiled graph: (backend, skip threshold, skip factor).
_TRACE: list[tuple[str, Any, Any]] = []
_DENSE = ("SDPA", None, None)
# The "approx" profile used by several tests: threshold 0.5, so factor = 0.5 * _SEQ.
_APPROX = ("TRTLLM_ATTN", 0.5, 4.0)


class _Layout(NamedTuple):
    """The strides of an attention result and, if it is a view, the shape and strides of its base."""

    strides: tuple[int, ...]
    base_shape: tuple[int, ...] | None
    base_strides: tuple[int, ...] | None


# The layouts of the attention results that ran outside a compiled graph, by backend name.
_LAYOUTS: dict[str, set[_Layout]] = {}


def _record_layout(backend, out):
    base = out._base
    layout = _Layout(
        tuple(out.stride()),
        None if base is None else tuple(base.shape),
        None if base is None else tuple(base.stride()),
    )
    _LAYOUTS.setdefault(backend, set()).add(layout)


def _attention_math(query, key, value, scale, gain=None):
    """Scaled dot-product attention on [batch, seq, heads, head_size] tensors, times ``gain`` if given.

    The result is a transpose of the [batch, heads, seq, head_size] attention output, with or without
    ``gain``, so outside inference mode it is a view whose base has that shape. Its strides follow
    the memory layout of the attention output.
    """
    out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale)
    if gain is not None:
        out = out * gain
    return out.transpose(1, 2)


class _DenseImpl:
    """The baseline. It records a call only when it runs outside a traced graph."""

    def __init__(self, *, softmax_scale, **kwargs):
        self.softmax_scale = softmax_scale

    def forward(self, query, key, value, attn_metadata=None):
        out = _attention_math(query, key, value, self.softmax_scale)
        if not torch.compiler.is_compiling():
            _TRACE.append(_DENSE)
            _record_layout("SDPA", out)
        return out


class _DenseBackend:
    supports_paged_kv = False
    supports_piecewise_spans = False

    @staticmethod
    def get_name() -> str:
        return "SDPA"

    @staticmethod
    def get_impl_cls():
        return _DenseImpl

    @staticmethod
    def supports_attention_mask(spec) -> bool:
        return True


class _RecordingTrtllmImpl(TrtllmAttentionImpl):
    """The production TRTLLM_ATTN implementation with ``forward`` replaced by the SDPA math.

    The factor comes from the production ``_resolve_skip_factor``, including the private timestep
    gate. With a factor the output is scaled by ``1 - threshold``, so an approximate step changes the
    trajectory; without one the dense math runs, as the kernel runs dense without a factor.
    """

    # None: the result has the baseline's strides and is the same kind of tensor. The copied-output
    # test sets "other-strides", the same values laid out in memory as [batch, heads, seq,
    # head_size] and returned as a transpose of that contiguous tensor, so it has other strides than
    # the baseline's result and, outside inference mode, is a view whose base has the same shape;
    # "other-base", a [batch * seq, heads, head_size] copy reshaped back, so it keeps the baseline's
    # strides and, outside inference mode, is a view of a three-dimensional base, as the production
    # result is a reshape of the kernel's [tokens, heads, head_size] output; or "same-strides", a
    # copy that keeps the baseline's strides and is not a view.
    output_copy: str | None = None

    def forward(self, query, key, value, attn_metadata=None):
        factor = self._resolve_skip_factor(key.shape[1])
        _TRACE.append(("TRTLLM_ATTN", self.skip.threshold, factor))
        gain = None if factor is None else 1.0 - self.skip.threshold
        out = _attention_math(query, key, value, self.softmax_scale, gain)
        if self.output_copy == "other-strides":
            out = out.transpose(1, 2).contiguous().transpose(1, 2)
        elif self.output_copy == "other-base":
            out = out.reshape(-1, *out.shape[2:]).clone().reshape(out.shape)
        elif self.output_copy == "same-strides":
            # clone keeps the strides of a dense tensor whose elements do not overlap.
            out = out.clone()
        if not torch.compiler.is_compiling():
            _record_layout("TRTLLM_ATTN", out)
        return out


class _RecordingTrtllmBackend(TrtllmAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return _RecordingTrtllmImpl


def _resolve(*, role, head_size, attention_config=None, role_category=None, allow_trtllm_default=True):
    """Stand-in for the platform selector; the platform default is _DenseBackend."""
    spec = None
    if attention_config is not None:
        spec, _source = attention_config.resolve_with_source(role=role, role_category=role_category)
    if spec is None:
        return _DenseBackend, None
    assert spec.backend.upper() == "TRTLLM_ATTN", spec.backend
    return _RecordingTrtllmBackend, spec


def _config(schedule: AttentionScheduleConfig | None = None) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(diffusion_attention_schedule=schedule)


def _trtllm(threshold, disabled_until_timestep=0.0):
    skip = SkipSoftmaxSpec(threshold=threshold, disabled_until_timestep=disabled_until_timestep)
    return AttentionConfig(default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax=skip))


def _ranges(*entries):
    return tuple(AttentionScheduleRange(start=start, end=end, profile=profile) for start, end, profile in entries)


class _Block(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.to_qkv = nn.Linear(_HIDDEN, 3 * _HIDDEN)
        self.to_out = nn.Linear(_HIDDEN, _HIDDEN)
        self.attn = Attention(
            num_heads=_HEADS,
            head_size=_HEAD_SIZE,
            causal=False,
            softmax_scale=_HEAD_SIZE**-0.5,
            prefix=f"blocks.{index}.attn",
        )

    def forward(self, hidden_states):
        batch, seq, _ = hidden_states.shape
        qkv = torch.sin(self.to_qkv(hidden_states)).view(batch, seq, 3, _HEADS, _HEAD_SIZE)
        query, key, value = qkv.unbind(2)
        out = self.attn(query, key, value)
        return hidden_states + torch.cos(self.to_out(out.reshape(batch, seq, _HIDDEN)))


class _Model(nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(self, blocks=_BLOCKS):
        super().__init__()
        self.blocks = nn.ModuleList(_Block(index) for index in range(blocks))

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _call_name(node) -> str:
    return getattr(node.target, "__name__", str(node.target))


class _CountingBackend:
    """A Dynamo backend that keeps the calls in each captured graph and counts the graph's executions."""

    def __init__(self):
        self.graphs: list[set[str]] = []
        self.executions: list[int] = []
        # How the model was compiled with this backend; set by _pipeline.
        self.settings: dict[str, Any] | None = None

    def __call__(self, gm, example_inputs):
        index = len(self.graphs)
        self.graphs.append({_call_name(node) for node in gm.graph.nodes if node.op.startswith("call_")})
        self.executions.append(0)

        def run(*args):
            self.executions[index] += 1
            return gm.forward(*args)

        return run


class _FailingBackend(_CountingBackend):
    """Records the graph it is given, then raises RuntimeError."""

    def __call__(self, gm, example_inputs):
        super().__call__(gm, example_inputs)
        raise RuntimeError("backend refused the graph")


def _frame_compiles() -> int:
    """Calls of Dynamo's fullgraph=False frame converter in this process.

    This includes recompiles, failed compiles and recompile-limit attempts. Fullgraph compiles do not
    update it.
    """
    return dynamo_counters["frames"]["total"]


def _frame_compiles_without_error() -> int:
    """The calls counted by ``_frame_compiles`` that returned without an error, including skipped frames."""
    return dynamo_counters["frames"]["ok"]


@dataclass
class _Step:
    trace: list[tuple[str, Any, Any]]
    executions: list[int]
    graphs: int
    # None for fullgraph runs, whose compiles do not update the frame counter.
    compiles: int | None
    latents: torch.Tensor
    compiles_without_error: int | None = None


class _Run(list[_Step]):
    """The steps of one request, with the conditions the request ran under."""

    conditions: dict[str, Any]


class _ToyPipeline(DenoiseProgressMixin):
    """A denoise loop around the model: it publishes each step, then runs the model once."""

    def __init__(self, model, config, counter, *, fullgraph=False):
        self.model = model
        self.config = config
        self.counter = counter
        self.fullgraph = fullgraph
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        # The steps of the latest request, kept when the request raises.
        self.last_run = _Run()

    def run(
        self,
        schedule,
        *,
        grad_enabled=False,
        inference_mode=False,
        stance="default",
        timesteps=_TIMESTEPS,
        publish_total=False,
    ):
        """One request with the given request schedule; returns what each step ran.

        Requests run with grad mode off. Dynamo guards grad mode on every cache entry, so with
        ``grad_enabled=True`` no entry compiled with grad mode off matches. ``inference_mode=True``
        runs the request under ``torch.inference_mode()``, as the runner does without HSDP or
        distributed offload; the latents are then created inside it, so every step's input is an
        inference tensor. ``stance`` other than ``"default"`` is passed to ``torch.compiler.set_stance``
        for the request. ``timesteps`` sets the number of steps and the timestep of each. A request
        without a bound schedule publishes no total, as ``begin_scheduled_denoise`` gives none;
        ``publish_total=True`` makes it publish its step count.
        """
        assert not (grad_enabled and inference_mode)
        steps = self.last_run = _Run()
        steps.conditions = {
            "grad_enabled": grad_enabled,
            "inference_mode": inference_mode,
            "stance": stance,
            "timesteps": list(timesteps),
            "recompile_limit": torch._dynamo.config.recompile_limit,
            "error_on_recompile": torch._dynamo.config.error_on_recompile,
            "suppress_errors": torch._dynamo.config.suppress_errors,
        }
        with (
            torch.inference_mode() if inference_mode else torch.set_grad_enabled(grad_enabled),
            torch.compiler.set_stance(stance) if stance != "default" else nullcontext(),
            set_forward_context(omni_diffusion_config=self.config),
            bind_attention_schedule(schedule),
        ):
            latents = torch.randn(1, _SEQ, _HIDDEN, generator=torch.Generator().manual_seed(_LATENT_SEED))
            total = begin_scheduled_denoise(len(timesteps))
            if publish_total:
                total = len(timesteps)
            steps.conditions["published_total"] = total
            for step_idx, timestep in enumerate(timesteps):
                self.record_denoise_step(step_idx, timestep, total_steps=total)
                trace_start = len(_TRACE)
                before = list(self.counter.executions)
                compiles_before, without_error_before = _frame_compiles(), _frame_compiles_without_error()
                latents = latents - 0.1 * self.model(latents)
                compiles = None if self.fullgraph else _frame_compiles() - compiles_before
                without_error = None if self.fullgraph else _frame_compiles_without_error() - without_error_before
                executions = [
                    count - (before[index] if index < len(before) else 0)
                    for index, count in enumerate(self.counter.executions)
                ]
                steps.append(
                    _Step(
                        trace=_TRACE[trace_start:],
                        executions=executions,
                        graphs=len(self.counter.graphs),
                        compiles=compiles,
                        latents=latents,
                        compiles_without_error=without_error,
                    )
                )
            self.record_denoise_step(None)
        return steps


class _MessageLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextmanager
def _dynamo_warnings():
    """Collects what Dynamo logs at WARNING or above while the block runs."""
    log = _MessageLog()
    logger = logging.getLogger("torch._dynamo")
    logger.addHandler(log)
    try:
        yield log.messages
    finally:
        logger.removeHandler(log)


@pytest.fixture
def compile_env(monkeypatch):
    monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _DenseImpl))
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", lambda **kwargs: _resolve(**kwargs))
    _TRACE.clear()
    _LAYOUTS.clear()
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    _TRACE.clear()
    _LAYOUTS.clear()


def _pipeline(schedule_config, *, compiled=True, fullgraph=False, backend=None, dynamic=False, blocks=_BLOCKS):
    # Every test reports a pipeline before it builds the next, so each record lists its own layouts.
    _LAYOUTS.clear()
    config = _config(schedule_config)
    with set_current_diffusion_config(config):
        torch.manual_seed(_WEIGHT_SEED)
        model = _Model(blocks)
    counter = _CountingBackend() if backend is None else backend
    if compiled:
        regionally_compile(model, backend=counter, fullgraph=fullgraph, dynamic=dynamic)
        counter.settings = {"helper": "regionally_compile", "dynamic": dynamic, "fullgraph": fullgraph}
    return _ToyPipeline(model, config, counter, fullgraph=fullgraph)


def _require_compiled_steps(counter, steps, blocks=_BLOCKS):
    """Each step ran compiled graphs on both sides of every attention call.

    An eager run, or a frame that Dynamo stopped compiling after too many recompiles, fails here.
    """
    assert counter.graphs, "no compiled graph was captured; the model ran eagerly"
    for index, step in enumerate(steps):
        runs = list(zip(counter.graphs, step.executions))
        before = sum(count for calls, count in runs if "sin" in calls)
        after = sum(count for calls, count in runs if "cos" in calls)
        assert (before, after) == (blocks, blocks), f"step {index}: compiled executions {before=} {after=}"


def _require_no_compile_after_first_step(requests, also_allowed=()):
    """Dynamo's frame converter is called only during the first step of the first request.

    ``requests`` holds the steps of each request in run order. A call in any later step or request
    fails here, including one for a frame whose graph is empty and so never reaches the backend.
    ``also_allowed`` lists (request, step) pairs that may compile as well. Fullgraph runs have no
    frame count and cannot use this check.
    """
    assert requests[0][0].compiles, (
        "the first step compiled no frame, so the frame counter did not change and cannot show a later compile"
    )
    for request, steps in enumerate(requests):
        for index, step in enumerate(steps):
            if (request, index) != (0, 0) and (request, index) not in also_allowed:
                assert step.compiles == 0, (
                    f"request {request} step {index}: {step.compiles} frame compile attempt(s) after the first step"
                )


def _require_attention_outside_graphs(counter, graphs=2):
    # Two graphs: the work before attention (to_qkv, sin, view, unbind) and the work after it
    # (reshape, to_out, cos, add). Both blocks run the same two. A test that allows another version
    # of one of them passes the count it expects.
    assert len(counter.graphs) == graphs, counter.graphs
    for calls in counter.graphs:
        assert "scaled_dot_product_attention" not in calls, calls
        # sin and cos in one graph would mean the attention call between them was traced.
        assert not {"sin", "cos"} <= calls, calls


def _settings_id(dynamic, inference_mode) -> str:
    """The parametrize id of a compile-settings case; record names use it."""
    return ("dynamic" if dynamic else "static") + "-" + ("inference-mode" if inference_mode else "no-grad")


def _first_failure(check, *args) -> str | None:
    """The first line of the AssertionError that ``check`` raises, or None if it passes."""
    try:
        check(*args)
    except AssertionError as error:
        return (str(error).splitlines() or [""])[0]
    return None


def _profiles(config):
    """Backend, threshold and timestep gate of each startup profile, or None without a schedule."""
    schedule = config.diffusion_attention_schedule
    if schedule is None:
        return None
    return {
        name: {
            "backend": profile.default.backend,
            "threshold": profile.default.skip_softmax.threshold,
            "disabled_until_timestep": profile.default.skip_softmax.disabled_until_timestep,
        }
        for name, profile in sorted(schedule.profiles.items())
    }


def _report(name, pipeline, requests, **outcome):
    """Print one JSON record with the conditions, the counts and the profile sequence of each request.

    ``outcome`` adds fields for a run that the checks must reject: ``rejected`` (the first line of
    the check that rejected it) or ``raised`` (the type of the exception it raised).
    """
    counter = pipeline.counter
    record = {
        "test": name,
        "torch": torch.__version__,
        "input": {
            "shape": [1, _SEQ, _HIDDEN],
            "dtype": "float32",
            "device": "cpu",
            "attention_layout": "[batch, seq, heads, head_size]",
            "blocks": len(pipeline.model.blocks),
            "heads": _HEADS,
            # The default; each request's timesteps are in its conditions.
            "timesteps": _TIMESTEPS,
            "num_train_timesteps": pipeline.scheduler.config.num_train_timesteps,
            "weight_seed": _WEIGHT_SEED,
            "latent_seed": _LATENT_SEED,
            "approximate_candidate_output_copy": _RecordingTrtllmImpl.output_copy,
        },
        "profiles": _profiles(pipeline.config),
        "compile": counter.settings,
        "backend": type(counter).__name__,
        "dynamo_config": {setting: getattr(torch._dynamo.config, setting, None) for setting in _DYNAMO_SETTINGS},
        "graphs": len(counter.graphs),
        "graph_calls": [sorted(calls) for calls in counter.graphs],
        "output_layouts": {
            backend: [layout._asdict() for layout in sorted(layouts, key=str)]
            for backend, layouts in sorted(_LAYOUTS.items())
        },
        "requests": [
            {
                "schedule": None if schedule is None else [[r.start, r.end, r.profile] for r in schedule],
                "conditions": getattr(steps, "conditions", None),
                "graphs_after_step": [step.graphs for step in steps],
                "frame_compile_attempts_per_step": [step.compiles for step in steps],
                "frame_compiles_without_error_per_step": [step.compiles_without_error for step in steps],
                "executions_per_step": [step.executions for step in steps],
                "trace_per_step": [step.trace for step in steps],
            }
            for schedule, steps in requests
        ],
        **outcome,
    }
    print("attention-schedule-compile-record " + json.dumps(record))


def _expected(labels, entries, blocks=_BLOCKS):
    """Per-step traces: each letter in ``labels`` is one step, with one entry per block."""
    return [[entries[label]] * blocks for label in labels]


def test_unseen_boundaries_across_requests_add_no_graph(compile_env):
    # After the first step, none of the requests below adds a graph or compiles a frame,
    # although three of them use boundaries the model has not run and one runs six steps instead of
    # eight, so the open range [3, None) ends at another step and the published total changes. The
    # same shapes run in every request; the ranges, the step count and the timesteps change. The two
    # unscheduled requests publish no total.
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}))
    entries = {"D": _DENSE, "A": _APPROX}
    # The number of steps of each request is the length of its labels.
    requests = [
        (_ranges((2, 5, "approx")), "DDAAADDD"),
        (_ranges((0, 1, "approx")), "ADDDDDDD"),
        (_ranges((3, None, "approx")), "DDDAAAAA"),
        (_ranges((1, 2, "approx"), (6, 7, "approx")), "DADDDDAD"),
        (_ranges((3, None, "approx")), "DDDAAA"),
        ((), "DDDDDDDD"),
        (None, "DDDDDDDD"),
    ]

    results: list[tuple[Any, list[_Step]]] = []
    first = pipeline.run(requests[0][0])
    results.append((requests[0][0], first))
    graphs = first[0].graphs
    with torch._dynamo.config.patch(error_on_recompile=True):
        for schedule, labels in requests[1:]:
            results.append((schedule, pipeline.run(schedule, timesteps=_timesteps(len(labels)))))
    _report("unseen-boundaries", pipeline, results)

    for (schedule, labels), (_schedule, steps) in zip(requests, results):
        assert [step.trace for step in steps] == _expected(labels, entries), schedule
        assert [step.graphs for step in steps] == [graphs] * len(labels), schedule
        _require_compiled_steps(pipeline.counter, steps)
    _require_no_compile_after_first_step([steps for _schedule, steps in results])
    _require_attention_outside_graphs(pipeline.counter)


def test_dense_approximate_dense_keeps_the_prefix_and_routes_back(compile_env):
    # The steps before the first switch match an all-dense run bit for bit. The run switches
    # back to dense after the range; the trajectory after the switch is not compared with all-dense.
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}))
    schedule = _ranges((2, 5, "approx"))

    scheduled = pipeline.run(schedule)
    dense = pipeline.run(())
    _report("dense-approximate-dense", pipeline, [(schedule, scheduled), ((), dense)])
    eager = _pipeline(None, compiled=False).run(None)

    assert [step.trace for step in scheduled] == _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX})
    for index in (0, 1):
        assert torch.equal(scheduled[index].latents, dense[index].latents), index
    # The candidate changed the result of the first scheduled step.
    assert not torch.equal(scheduled[2].latents, dense[2].latents)
    # The compiled all-dense run computes the same model as an uncompiled one.
    for compiled_step, eager_step in zip(dense, eager):
        torch.testing.assert_close(compiled_step.latents, eager_step.latents)
    assert [step.graphs for step in scheduled + dense] == [scheduled[0].graphs] * (2 * len(_TIMESTEPS))
    _require_compiled_steps(pipeline.counter, scheduled + dense)
    _require_no_compile_after_first_step([scheduled, dense])
    _require_attention_outside_graphs(pipeline.counter)


def test_same_backend_candidates_run_their_own_parameters(compile_env):
    # Two TRTLLM_ATTN profiles that differ only in the threshold keep separate implementations,
    # and each scheduled step calls the one whose range it is in (factor = threshold * seq).
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"skip_low": _trtllm(0.25), "skip_high": _trtllm(0.75)}))
    schedule = _ranges((1, 3, "skip_low"), (5, 7, "skip_high"))

    steps = pipeline.run(schedule)
    _report("same-backend-parameters", pipeline, [(schedule, steps)])

    for block in pipeline.model.blocks:
        candidates = block.attn._schedule_candidates
        assert candidates["skip_low"].impl is not candidates["skip_high"].impl
    entries = {"D": _DENSE, "L": ("TRTLLM_ATTN", 0.25, 2.0), "H": ("TRTLLM_ATTN", 0.75, 6.0)}
    assert [step.trace for step in steps] == _expected("DLLDDHHD", entries)
    assert [step.graphs for step in steps] == [steps[0].graphs] * len(_TIMESTEPS)
    _require_compiled_steps(pipeline.counter, steps)
    _require_no_compile_after_first_step([steps])
    _require_attention_outside_graphs(pipeline.counter)


@pytest.mark.parametrize(
    ("disabled_until_timestep", "labels"),
    [(0.0, "DDFFFFDD"), (0.5, "DDGGFFDD")],
    ids=["gate-off", "gate-on"],
)
def test_private_timestep_gate_runs_inside_the_scheduled_range(compile_env, disabled_until_timestep, labels):
    # The generic range [2, 6) decides which steps call the candidate in both cases. With the
    # backend's timestep gate on, the candidate itself stays dense while the published timestep is
    # above 0.5 (steps 2 and 3). The timestep changes every step and adds no graph.
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"skip": _trtllm(0.5, disabled_until_timestep)}))
    schedule = _ranges((2, 6, "skip"))

    steps = pipeline.run(schedule)
    _report(f"private-gate-{disabled_until_timestep}", pipeline, [(schedule, steps)])

    entries = {"D": _DENSE, "F": ("TRTLLM_ATTN", 0.5, 4.0), "G": ("TRTLLM_ATTN", 0.5, None)}
    assert [step.trace for step in steps] == _expected(labels, entries)
    assert [step.graphs for step in steps] == [steps[0].graphs] * len(_TIMESTEPS)
    _require_compiled_steps(pipeline.counter, steps)
    _require_no_compile_after_first_step([steps])
    _require_attention_outside_graphs(pipeline.counter)


def test_attention_without_a_schedule_stays_in_the_compiled_graph(compile_env):
    # No startup schedule: the layer adds no boundary, so the attention math is traced into the
    # single block graph and the baseline never runs eagerly.
    pipeline = _pipeline(None)

    steps = pipeline.run(None)
    _report("no-schedule", pipeline, [(None, steps)])

    assert len(pipeline.counter.graphs) == 1, pipeline.counter.graphs
    assert "scaled_dot_product_attention" in pipeline.counter.graphs[0]
    assert [step.trace for step in steps] == [[]] * len(_TIMESTEPS)
    assert [step.graphs for step in steps] == [steps[0].graphs] * len(_TIMESTEPS)
    _require_compiled_steps(pipeline.counter, steps)
    _require_no_compile_after_first_step([steps])


# Requests of the production-settings test: both step counts, an open and a split range, and an
# unscheduled request last. The number of steps of each request is the length of its labels.
_SETTINGS_REQUESTS = [
    (_ranges((2, 5, "approx")), "DDAAADDD"),
    (_ranges((3, None, "approx")), "DDDAAA"),
    (_ranges((0, 1, "approx"), (6, 7, "approx")), "ADDDDDAD"),
    (None, "DDDDDDDD"),
]


@pytest.mark.parametrize("inference_mode", [False, True], ids=["no-grad", "inference-mode"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_production_compile_settings_compile_only_in_the_first_step(compile_env, dynamic, inference_mode):
    # The runner compiles with diffusion_compile_dynamic, True by default, and runs requests under
    # inference_mode when neither HSDP nor distributed offload is on; most other tests use
    # dynamic=False and grad mode off. Under each combination, a model with a startup schedule runs
    # the requests above and a model without one runs 8-step and 6-step requests. Each keeps its graph
    # count from the first step (2 with a schedule, 1 without), and only the first step of its first
    # request compiles frames.
    settings = _settings_id(dynamic, inference_mode)
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), dynamic=dynamic)
    results = [
        (schedule, pipeline.run(schedule, inference_mode=inference_mode, timesteps=_timesteps(len(labels))))
        for schedule, labels in _SETTINGS_REQUESTS
    ]
    _report(f"production-settings-{settings}-scheduled", pipeline, results)

    torch._dynamo.reset()
    unscheduled = _pipeline(None, dynamic=dynamic)
    # An unscheduled layer is traced into the block graph. Without a bound schedule the toy pipeline
    # publishes no total; MiniMax H3 publishes its step count in unscheduled runs too, so the last two
    # requests publish theirs.
    plain = [
        (None, unscheduled.run(None, inference_mode=inference_mode, timesteps=_timesteps(count), publish_total=total))
        for total in (False, True)
        for count in (8, 6)
    ]
    _report(f"production-settings-{settings}-no-schedule", unscheduled, plain)

    entries = {"D": _DENSE, "A": _APPROX}
    for (schedule, labels), (_schedule, steps) in zip(_SETTINGS_REQUESTS, results):
        assert [step.trace for step in steps] == _expected(labels, entries), schedule
        assert [step.graphs for step in steps] == [2] * len(labels), schedule
        _require_compiled_steps(pipeline.counter, steps)
    _require_no_compile_after_first_step([steps for _schedule, steps in results])
    _require_attention_outside_graphs(pipeline.counter)

    assert len(unscheduled.counter.graphs) == 1, unscheduled.counter.graphs
    assert "scaled_dot_product_attention" in unscheduled.counter.graphs[0]
    for _schedule, steps in plain:
        assert [step.trace for step in steps] == [[]] * len(steps)
        assert [step.graphs for step in steps] == [1] * len(steps)
        _require_compiled_steps(unscheduled.counter, steps)
    _require_no_compile_after_first_step([steps for _schedule, steps in plain])


@pytest.mark.parametrize("inference_mode", [False, True], ids=["no-grad", "inference-mode"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("output_copy", ["other-strides", "other-base", "same-strides"])
def test_copied_candidate_output_compiles_only_when_it_first_runs(
    compile_env, monkeypatch, output_copy, dynamic, inference_mode
):
    # A finite candidate set may add a finite number of graph variants. The frame that
    # resumes after the attention call takes the tensor the eager boundary returns as an input, and
    # Dynamo guards properties of that input. Here the approximate candidate returns another tensor
    # than the dense result. With "other-strides" it has other strides but is the same kind of
    # tensor: outside inference mode a view whose base has the dense result's base shape. With
    # "other-base" it keeps the dense strides and, outside inference mode, is a view whose base has
    # another shape and rank. With "same-strides" it is a copy that keeps the dense strides and
    # differs only in not being a view, which the dense result is outside inference mode. So the
    # first step that runs the approximate candidate, step 2 of the first request, may compile
    # frames and add one version of the graph after attention. No other step may: not the later
    # steps, and not the later requests with other ranges, another step count or no schedule. The
    # assertions bound the outcome for every variant, and the last ones check on the returned tensors
    # that each variant differs from the dense result as described; whether a given variant compiled
    # a frame in that step is in the printed record.
    monkeypatch.setattr(_RecordingTrtllmImpl, "output_copy", output_copy)
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), dynamic=dynamic)
    results = [
        (schedule, pipeline.run(schedule, inference_mode=inference_mode, timesteps=_timesteps(len(labels))))
        for schedule, labels in _SETTINGS_REQUESTS
    ]
    _report(f"copied-candidate-output-{output_copy}-{_settings_id(dynamic, inference_mode)}", pipeline, results)

    entries = {"D": _DENSE, "A": _APPROX}
    for (schedule, labels), (_schedule, steps) in zip(_SETTINGS_REQUESTS, results):
        assert [step.trace for step in steps] == _expected(labels, entries), schedule
        _require_compiled_steps(pipeline.counter, steps)
    graphs = [step.graphs for _schedule, steps in results for step in steps]
    assert graphs[:2] == [2, 2]
    assert graphs[2] in (2, 3), graphs
    assert graphs[2:] == [graphs[2]] * len(graphs[2:]), graphs
    _require_no_compile_after_first_step([steps for _schedule, steps in results], also_allowed=[(0, 2)])
    _require_attention_outside_graphs(pipeline.counter, graphs=graphs[2])
    # A third graph is another version of the graph after attention.
    assert sum("sin" in calls for calls in pipeline.counter.graphs) == 1, pipeline.counter.graphs
    # Each backend returned one layout in every call. With "other-strides" the result differs from
    # the dense result in its strides, and outside inference mode in the strides of its base, but not
    # in being a view or in the shape of its base. With "other-base" it differs only outside
    # inference mode, in the shape of its base. With "same-strides" it differs in being no view, and
    # only outside inference mode, where the dense result is a view.
    assert {backend: len(layouts) for backend, layouts in _LAYOUTS.items()} == {"SDPA": 1, "TRTLLM_ATTN": 1}, _LAYOUTS
    (dense,), (copy,) = _LAYOUTS["SDPA"], _LAYOUTS["TRTLLM_ATTN"]
    if output_copy == "other-strides":
        assert copy.strides != dense.strides and copy.base_shape == dense.base_shape, _LAYOUTS
        assert (copy.base_strides == dense.base_strides) == inference_mode, _LAYOUTS
    elif output_copy == "other-base":
        assert copy.strides == dense.strides and (copy.base_shape is None) == inference_mode, _LAYOUTS
        assert (copy.base_shape == dense.base_shape) == inference_mode, _LAYOUTS
        if not inference_mode:
            assert copy.base_shape is not None and dense.base_shape is not None, _LAYOUTS
            assert len(copy.base_shape) == len(dense.base_shape) - 1, _LAYOUTS
    else:
        assert copy.strides == dense.strides and copy.base_shape is None, _LAYOUTS
        assert (dense.base_shape is None) == inference_mode, _LAYOUTS


@pytest.mark.parametrize("inference_mode", [False, True], ids=["no-grad", "inference-mode"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("scheduled", [True, False], ids=["scheduled", "no-schedule"])
def test_second_block_compiles_no_frame_in_the_first_step(compile_env, scheduled, dynamic, inference_mode):
    # The frame check allows any number of compiles in the first step. Both blocks run the same code,
    # so the second block should reuse every cache entry the first block compiled, and a two-block
    # model should compile as many frames in its first step as a one-block model. A guard on a value
    # that differs per layer, read before the eager boundary, would recompile that frame for the
    # second block without adding a graph. The comparison runs under the four compile settings of the
    # production-settings test.
    schedule_config = AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}) if scheduled else None
    schedule = _ranges((2, 5, "approx")) if scheduled else None
    name = f"{'scheduled' if scheduled else 'no-schedule'}-{_settings_id(dynamic, inference_mode)}"
    runs = []
    first_step_compiles = {}
    for blocks in (1, 2):
        torch._dynamo.reset()
        pipeline = _pipeline(schedule_config, dynamic=dynamic, blocks=blocks)
        steps = pipeline.run(schedule, inference_mode=inference_mode)
        _report(f"block-count-{name}-{blocks}", pipeline, [(schedule, steps)])
        runs.append((blocks, pipeline, steps))
        first_step_compiles[blocks] = steps[0].compiles

    for blocks, pipeline, steps in runs:
        _require_compiled_steps(pipeline.counter, steps, blocks=blocks)
        _require_no_compile_after_first_step([steps])
        if scheduled:
            assert [step.trace for step in steps] == _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX}, blocks)
            _require_attention_outside_graphs(pipeline.counter)
        else:
            assert len(pipeline.counter.graphs) == 1, pipeline.counter.graphs
            assert "scaled_dot_product_attention" in pipeline.counter.graphs[0]
            assert [step.trace for step in steps] == [[]] * len(steps)
    one, two = first_step_compiles[1], first_step_compiles[2]
    assert one == two, f"the second block compiled {two - one} frame(s) in the first step ({one} with one block)"


def test_eager_fallback_is_not_counted_as_compiled_execution(compile_env):
    # A run that falls back to eager still selects the right profiles, so the profile sequence
    # alone cannot prove compilation; the execution counts do. The first model never compiles. The
    # second runs eagerly after a compiled request captured its graphs, so the per-step count check
    # has to catch it.
    schedule = _ranges((2, 5, "approx"))
    expected = _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX})

    never_compiled = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}))
    steps = never_compiled.run(schedule, stance="force_eager")
    rejected = _first_failure(_require_compiled_steps, never_compiled.counter, steps)
    _report("force-eager-never-compiled", never_compiled, [(schedule, steps)], rejected=rejected)

    assert [step.trace for step in steps] == expected
    assert rejected == "no compiled graph was captured; the model ran eagerly"

    torch._dynamo.reset()
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}))
    compiled = pipeline.run(schedule)
    steps = pipeline.run(schedule, stance="force_eager")
    rejected = _first_failure(_require_compiled_steps, pipeline.counter, steps)
    _report(
        "force-eager-after-compile",
        pipeline,
        [(schedule, compiled), (schedule, steps)],
        rejected=rejected,
    )

    _require_compiled_steps(pipeline.counter, compiled)
    _require_no_compile_after_first_step([compiled])
    assert [step.trace for step in steps] == expected
    assert rejected == "step 0: compiled executions before=0 after=0"


def test_recompile_limit_fallback_is_not_counted_as_compiled_execution(compile_env):
    # Dynamo's own eager fallback. With fullgraph=False and config.fail_on_recompile_limit_hit=False
    # (the default), a frame that needs a compile beyond config.recompile_limit does not raise:
    # Dynamo logs a warning and runs the frame and the frames it calls without compiling them, and
    # does the same, until a reset, whenever no cache entry of the frame matches. The first request
    # compiles with grad mode off. The second runs with grad mode on, so it fails the global-state
    # guard of every cache entry, under a limit of 1. It still selects the right profiles and adds
    # no graph; the counts reject it.
    assert logging.getLogger("torch._dynamo.convert_frame").isEnabledFor(logging.WARNING), (
        "Dynamo warnings are disabled, for example by TORCH_LOGS"
    )
    schedule = _ranges((2, 5, "approx"))
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}))

    compiled = pipeline.run(schedule)
    with _dynamo_warnings() as warnings, torch._dynamo.config.patch(recompile_limit=1):
        fallback = pipeline.run(schedule, grad_enabled=True)
    steps_rejected = _first_failure(_require_compiled_steps, pipeline.counter, fallback)
    compiles_rejected = _first_failure(_require_no_compile_after_first_step, [compiled, fallback])
    _report(
        "recompile-limit-fallback",
        pipeline,
        [(schedule, compiled), (schedule, fallback)],
        dynamo_warnings=[(message.splitlines() or [""])[0] for message in warnings],
        rejected={"compiled_steps": steps_rejected, "frame_compiles": compiles_rejected},
    )

    _require_compiled_steps(pipeline.counter, compiled)
    _require_no_compile_after_first_step([compiled])
    assert any("hit config.recompile_limit" in message for message in warnings), warnings
    assert [step.trace for step in fallback] == _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX})
    assert [step.graphs for step in fallback] == [2] * len(_TIMESTEPS)
    assert steps_rejected == "step 0: compiled executions before=0 after=0"
    # The attempt that hit the limit is one call of the frame converter, in the first step of the
    # second request, although it compiles nothing.
    assert compiles_rejected is not None and compiles_rejected.startswith("request 1 step 0: "), compiles_rejected


def test_compiled_step_check_names_the_step_with_a_missing_side():
    # A frame can stop compiling on one side of attention only, for example after too many
    # recompiles. The check counts the graphs before and after attention separately and names the
    # first step in which one side did not run compiled.
    counter = _CountingBackend()
    counter.graphs = [{"sin"}, {"cos"}]
    compiled = _Step(trace=[], executions=[_BLOCKS, _BLOCKS], graphs=2, compiles=0, latents=torch.zeros(1))
    one_sided = _Step(trace=[], executions=[_BLOCKS, 0], graphs=2, compiles=0, latents=torch.zeros(1))

    _require_compiled_steps(counter, [compiled])
    with pytest.raises(AssertionError, match=f"step 1: compiled executions before={_BLOCKS} after=0"):
        _require_compiled_steps(counter, [compiled, one_sided])


def test_frame_compile_check_names_the_first_recompile():
    # Only the first step of the first request may compile. The check names the first later step
    # that compiled a frame, and fails when the first step compiled nothing, since a frame count
    # that never moves cannot show a recompile.
    def steps(*compiles):
        return [_Step(trace=[], executions=[], graphs=2, compiles=count, latents=torch.zeros(1)) for count in compiles]

    _require_no_compile_after_first_step([steps(5, 0, 0), steps(0, 0)])
    with pytest.raises(AssertionError, match="request 0 step 2: 1 frame compile"):
        _require_no_compile_after_first_step([steps(5, 0, 1), steps(0, 2)])
    with pytest.raises(AssertionError, match="request 1 step 0: 2 frame compile"):
        _require_no_compile_after_first_step([steps(5, 0, 0), steps(2, 0)])
    with pytest.raises(AssertionError, match="the first step compiled no frame"):
        _require_no_compile_after_first_step([steps(0, 0, 0)])
    # A step listed in also_allowed may compile; every other later step still may not.
    _require_no_compile_after_first_step([steps(5, 0, 1), steps(0, 0)], also_allowed=[(0, 2)])
    with pytest.raises(AssertionError, match="request 1 step 0: 2 frame compile"):
        _require_no_compile_after_first_step([steps(5, 0, 1), steps(2, 0)], also_allowed=[(0, 2)])


def test_fullgraph_compile_cannot_contain_the_schedule_boundary(compile_env):
    # With fullgraph=True (Ming-Image, the DreamZero blocks) an unscheduled layer traces into one
    # graph that both blocks run. A layer built with a startup schedule calls the eager boundary,
    # which a fullgraph compile cannot contain, so the first forward raises. None of the in-tree
    # fullgraph compile sites publishes denoise progress, and startup rejects schedule profiles on
    # such pipelines, so this combination does not reach a request.
    pipeline = _pipeline(None, fullgraph=True)

    steps = pipeline.run(None)
    _report("fullgraph-no-schedule", pipeline, [(None, steps)])

    assert len(pipeline.counter.graphs) == 1, pipeline.counter.graphs
    assert {"sin", "scaled_dot_product_attention", "cos"} <= pipeline.counter.graphs[0]
    assert [step.trace for step in steps] == [[]] * len(_TIMESTEPS)
    _require_compiled_steps(pipeline.counter, steps)
    # Fullgraph compiles do not update the frame counter. The fullgraph block frame always contains
    # tensor operations, so every compile of it reaches the backend, and a constant graph count shows
    # that no step after the first compiled again.
    assert [step.graphs for step in steps] == [1] * len(_TIMESTEPS)

    torch._dynamo.reset()
    scheduled = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), fullgraph=True)
    schedule = _ranges((2, 5, "approx"))
    with pytest.raises(torch._dynamo.exc.Unsupported, match="disable") as raised:
        scheduled.run(schedule)
    _report("fullgraph-scheduled", scheduled, [(schedule, scheduled.last_run)], raised=type(raised.value).__name__)

    assert scheduled.counter.graphs == []
    assert _TRACE == []


def test_compile_failure_is_raised_instead_of_running_eagerly(compile_env):
    # A backend failure reaches the caller: with suppress_errors=False (the default when
    # TORCHDYNAMO_SUPPRESS_ERRORS is unset) the frame does not continue eagerly. The first graph,
    # the projection before attention, fails, so no attention call runs.
    assert not torch._dynamo.config.suppress_errors, "TORCHDYNAMO_SUPPRESS_ERRORS is set"
    failing = _FailingBackend()
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), backend=failing)
    schedule = _ranges((2, 5, "approx"))

    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed, match="backend refused the graph") as raised:
        pipeline.run(schedule)
    _report("compile-failure", pipeline, [(schedule, pipeline.last_run)], raised=type(raised.value).__name__)

    assert len(failing.graphs) == 1, failing.graphs
    assert "sin" in failing.graphs[0] and "cos" not in failing.graphs[0], failing.graphs
    assert _TRACE == []


def test_suppressed_compile_failure_is_not_counted_as_compiled_execution(compile_env):
    # With suppress_errors=True, Dynamo logs the backend failure, runs the block frame without
    # compiling it and skips that frame's code until a reset. The frames it calls are still offered
    # to Dynamo: torch.nn frames are on its skip list, and Attention.forward compiles with an empty
    # graph and calls the eager boundary. Only the graph before attention reaches the backend, and no
    # backend graph runs. The request still selects the right profiles, and the failed graph was
    # captured, so only the execution counts reject the run.
    failing = _FailingBackend()
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), backend=failing)
    schedule = _ranges((2, 5, "approx"))

    with torch._dynamo.config.patch(suppress_errors=True):
        steps = pipeline.run(schedule)
    rejected = _first_failure(_require_compiled_steps, failing, steps)
    _report("suppressed-compile-failure", pipeline, [(schedule, steps)], rejected=rejected)

    # One graph reached the backend: the projection before attention.
    assert len(failing.graphs) == 1, failing.graphs
    assert "sin" in failing.graphs[0] and "cos" not in failing.graphs[0], failing.graphs
    assert [step.trace for step in steps] == _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX})
    assert rejected == "step 0: compiled executions before=0 after=0"
