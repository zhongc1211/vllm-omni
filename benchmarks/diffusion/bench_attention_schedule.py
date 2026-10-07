# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Acceptance harness for step-index attention schedules on diffusion models.

One process starts one service with ``Omni(...)``, installs probes by name,
sends a fixed list of requests and writes one session record. A separate
command reads the session records and decides, item by item, whether the
evidence shows a real switch (dense to quantized or sparse), a bitwise-equal
dense prefix, executed compiled code and the target kernel.

The harness supports one process and one GPU. A config that asks for another
GPU count is a config error (exit code 2). With several stages, or with a
diffusion stage in another process, the probes do not reach the model, and
``run`` stops with exit code 3.

Sessions of one combination (C1 to C6, see ``COMBINATIONS``). Each session
starts one service:
    prechange   a service without a schedule, run on the code from before
                the schedule feature (a checkout on ``PYTHONPATH``);
                requests ``warmup`` and ``plain``
    plain       the same two requests on the current code, again without a
                schedule
    scheduled   a service that declares the profiles of ``schedule_config``;
                requests ``warmup``, ``reference_a`` and ``reference_b``
                (schedule ``[]``: every step dense), ``candidate``
                (``candidate_schedule``) and, when the config sets it,
                ``boundary_shift`` (``boundary_shift_schedule``)

Verdict items of one combination. Each is ``pass``, ``fail``, ``not_proven``
or ``not_measured``, and a ``pass`` proves what is written here:
    capture           the scheduled record is complete: the requests are in
                      the expected order, the two references and the candidate
                      ran without error, each has the expected sequences and
                      steps, and no block, attention or kernel call was counted
                      outside a step
    conditions        ``reference_a`` and ``candidate`` differ only in the
                      schedule: same prompt, sampling parameters, initial
                      state, step count and output shape
    dense_repeatable  the two dense requests give equal digests on every
                      step, equal final states and equal output bytes, so a
                      difference of the candidate comes from its schedule
    profile_switch    on every step the attention layers selected what the
                      schedule states (the profile, or the layer's own
                      implementation), the candidate's digests first differ
                      from the reference at the first profile step, and a
                      call of the configured kind ran on that step
    dense_prefix      before the first profile step the candidate's digests
                      equal those of ``reference_a`` on every step
    kernel_target     the profile steps reached the FlashInfer kernel of
                      ``TRTLLM_ATTN`` with the arguments of the profile's kind
                      (SAGE scale factors for ``quantized``, a positive
                      skip-softmax factor for ``sparse``) on at least one
                      attention call
    compiled_graphs   the blocks of every model were compiled without error and
                      every block call of ``reference_a`` and ``candidate``
                      ran a compiled graph, none ran eagerly
    shifted_boundary  the ``boundary_shift`` request selected what its
                      schedule states, made no backend compilation attempt and
                      ran compiled graphs only: moving the first profile step needs no
                      new compilation
    timing            the seconds per step are usable for a comparison: every
                      sample is present, the batch size is 1, no backend compile
                      attempt occurred in a measured request and the warm-up ran
                      every selection first. The item applies no speed
                      threshold; the report holds the samples and medians
    lpips             an LPIPS score of the candidate output against the
                      ``reference_a`` output was computed over aligned frames
                      and is stored with its metadata. No threshold applies
    video             the side-by-side video and the videos of
                      ``reference_a`` and ``candidate`` are in the evidence
                      directory and match their recorded sha256
    backend_report    the attention probe was installed and every step of
                      ``reference_a`` and ``candidate`` holds the selection,
                      backend, outcome and kernel counters
    no_schedule       the ``plain`` session equals the ``prechange`` session
                      bit for bit (step digests, final state, output), or
                      differs within ``no_schedule_tolerance``, and the
                      ``prechange`` record comes from the source that
                      ``prechange_source`` states

A combination is proven when every item except ``shifted_boundary`` passes.
The first line of the ``verdict`` summary is ``Acceptance: SUCCESS`` when all
six combinations are proven and ``shifted_boundary`` passed on at least one of
them and failed on none; otherwise it is ``Acceptance: NOT PROVEN``. The
second line, ``Shifted boundary: ...``, lists the combinations on which that
item passed and failed.

Compiler records use ``inductor-attempts-v1``: backend calls, successful
returns, tensorification restarts and other errors are separate counters. A
restart is rethrown unchanged for Dynamo to handle. Request-boundary snapshots
are required to prove that measurement and boundary-shift requests made no
backend compile attempt. Older records lack that evidence; missing counters
are never assumed to be zero. Attempts that stop before reaching the backend
are not counted, so warm-up coverage and executed-graph checks remain required.

Requirements for ``run``:
    pip install lpips Pillow numpy

Check the probe targets, the compile probes and the comparison tools without
loading model weights (LPIPS can download its backbone weights once). The
compile self-tests and the LPIPS score run on ``--device`` (default ``cpu``):
    python benchmarks/diffusion/bench_attention_schedule.py preflight \
        --config /evidence/c5/config.json --device cuda

Run the three sessions of one combination (the evidence directory must be
outside the repository):
    PYTHONPATH=/path/to/pre-change/checkout \
    python benchmarks/diffusion/bench_attention_schedule.py run \
        --config /evidence/c5/config.json --session prechange --out /evidence/c5
    python benchmarks/diffusion/bench_attention_schedule.py run \
        --config /evidence/c5/config.json --session plain --out /evidence/c5 \
        --prior /evidence/c5/prechange.json
    python benchmarks/diffusion/bench_attention_schedule.py run \
        --config /evidence/c5/config.json --session scheduled --out /evidence/c5

Decide (standard library only, no GPU):
    python benchmarks/diffusion/bench_attention_schedule.py verdict \
        --out /evidence/verdict.json /evidence/c1 /evidence/c2 /evidence/c3 \
        /evidence/c4 /evidence/c5 /evidence/c6

Exit codes:
    preflight   0 all checks passed, 2 usage or config error, 3 a check failed
    run         0 record written and every request ran, 2 usage or config
                error, or the evidence directory cannot be created, 3 the
                probes did not reach the model, a probe target is missing or
                cannot be imported, the comparison tools do not work, the
                service has several stages, the warm-up did not run what the
                later requests run or the output of a request holds no usable
                frames, 4 the sampling parameters could not be built, a request
                or the service start raised, or the record could not be written
    verdict     0 all six combinations are proven and the shifted-boundary
                check passed on at least one of them and failed on none,
                1 otherwise, 2 usage error

Exit code 0 from ``run`` only says that the record is complete. Only
``verdict`` states a result.

Config file (JSON, one per combination, used unchanged by all three sessions):
    schema                    "attention-schedule-bench/1"
    combination               "C1" .. "C6"
    model_family              "minimax_h3" | "wan2_2"
    execution_mode            "request" | "step"
    approximation             "quantized" | "sparse"
    omni_kwargs               keyword arguments of Omni(...), with "model"
    schedule_config           {"profiles": {name: {...}}, "default": []}
    approx_profiles           {name: {"kind": ..., "backend": ...}}
    candidate_schedule        [{"start": int, "end": int | null, "profile": name}]
    warmup_schedule           same form; on each transformer it must run the
                              profiles and the dense path that the later
                              requests run there
    boundary_shift_schedule   null, or the candidate with another first switch;
                              ``verdict`` exits with 0 only when at least one
                              combination sets it
    expected_total_steps      length of the denoise sequence
    expected_sequences        denoise sequences per request (default 1)
    expected_frames           frame count of the output, or null
    prompt                    prompt object passed to Omni.generate
    sampling_params           fields of OmniDiffusionSamplingParams, with "seed"
                              and "num_inference_steps"; "output_type" must be
                              "np" or absent
    prechange_source          {"git_head": ..., "package_tree_sha256": ...}
    no_schedule_tolerance     null, or {"max_abs_diff_uint8": int}
    lpips                     {"net": "alex" | "vgg" | "squeeze"}
    fps                       frame rate of the written videos
    adapter                   optional {"pipeline_module": ..., "pipeline_class": ...}
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

SCHEMA = "attention-schedule-bench/1"
SESSIONS = ("scheduled", "plain", "prechange")
SESSION_FILES = {name: f"{name}.json" for name in SESSIONS}

# (model_family, execution_mode, approximation) of the six combinations.
COMBINATIONS: dict[str, tuple[str, str, str]] = {
    "C1": ("minimax_h3", "request", "quantized"),
    "C2": ("minimax_h3", "request", "sparse"),
    "C3": ("minimax_h3", "step", "quantized"),
    "C4": ("minimax_h3", "step", "sparse"),
    "C5": ("wan2_2", "request", "quantized"),
    "C6": ("wan2_2", "request", "sparse"),
}
SEQUENCE_SOURCES: dict[tuple[str, str], str] = {
    ("minimax_h3", "request"): "h3_request_loop",
    ("minimax_h3", "step"): "h3_step_scheduler",
    ("wan2_2", "request"): "wan_request_diffuse",
}

H3_PIPELINE_MODULE = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"
H3_PIPELINE_CLASS = "MiniMaxH3Pipeline"
WAN_PIPELINE_MODULE = "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2"
WAN_PIPELINE_CLASS = "Wan22Pipeline"
RUNNER_MODULE = "vllm_omni.diffusion.worker.diffusion_model_runner"
ATTENTION_MODULE = "vllm_omni.diffusion.attention.layer"
FORWARD_CONTEXT_MODULE = "vllm_omni.diffusion.forward_context"
TRTLLM_MODULE = "vllm_omni.diffusion.attention.backends.trtllm_attn"
TRTLLM_KERNEL = "trtllm_ragged_attention_deepseek"
KERNEL_PROBE_BACKEND = "TRTLLM_ATTN"
COMPILE_MECHANISM = "torch._TorchCompileInductorWrapper.__call__"
COMPILE_COUNTER_SCHEMA = "inductor-attempts-v1"
H3_LOOP_PARAMETERS = ("on_step", "step_profiler", "initial_video_rows", "initial_audio_rows", "sigmas_video")

SCHEDULED_REQUESTS = ("warmup", "reference_a", "reference_b", "candidate")
# Reasons _classify_attention gives for a layer call that selected a profile and ran no approximate kernel call.
FALLBACK_REASONS = (
    "sage_not_applied",
    "ignored_layer",
    "private_gate",
    "profile_not_approximate",
    "layer_fallback",
    "unclassified",
)
ENV_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONPATH",
    "TORCH_LOGS",
    "DIFFUSION_ATTENTION_BACKEND",
    "DIFFUSION_ATTENTION_QUANT",
)
ENV_PREFIXES = ("TORCHINDUCTOR_", "TORCHDYNAMO_", "PYTORCH_")
LABEL_BAND = 32

# An item of the verdict is evaluated only when every dependency passed.
ITEM_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "capture": (),
    "conditions": ("capture",),
    "dense_repeatable": ("capture",),
    "profile_switch": ("capture", "conditions"),
    "dense_prefix": ("capture", "conditions", "dense_repeatable"),
    "kernel_target": ("capture",),
    "compiled_graphs": ("capture",),
    "shifted_boundary": ("capture",),
    "timing": ("capture",),
    "lpips": ("conditions",),
    "video": ("conditions",),
    "backend_report": ("capture",),
}
# no_schedule reads the plain and pre-change sessions and has no dependency.
REQUIRED_ITEMS = (
    "capture",
    "conditions",
    "dense_repeatable",
    "profile_switch",
    "timing",
    "lpips",
    "video",
    "backend_report",
    "dense_prefix",
    "no_schedule",
    "kernel_target",
    "compiled_graphs",
)

_CONFIG_REQUIRED = (
    "schema",
    "combination",
    "model_family",
    "execution_mode",
    "approximation",
    "omni_kwargs",
    "schedule_config",
    "approx_profiles",
    "candidate_schedule",
    "warmup_schedule",
    "expected_total_steps",
    "prompt",
    "sampling_params",
    "prechange_source",
    "fps",
)
_CONFIG_OPTIONAL = (
    "boundary_shift_schedule",
    "expected_sequences",
    "expected_frames",
    "no_schedule_tolerance",
    "lpips",
    "adapter",
    "_sha256",
)
_MISSING = object()


class EvidenceError(RuntimeError):
    """A probe target is missing, the probes did not reach the model, or a measurement input is unusable."""


class ConfigError(ValueError):
    """The config file or the command line is invalid."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_profile_name(name: Any) -> bool:
    """Return whether production accepts ``name`` as a profile name."""
    return (
        isinstance(name, str)
        and bool(name)
        and name.isascii()
        and name[0].isalpha()
        and all(char.isalnum() or char in "_-" for char in name)
    )


def _is_skip_factor(factor: Any) -> bool:
    """Return whether a kernel call with this skip factor skips softmax work.

    A threshold of 0.0 gives the factor 0.0, which skips nothing, and the kernel
    probe records a factor it cannot read as NaN. Neither counts as a sparse call.
    """
    return isinstance(factor, float) and math.isfinite(factor) and factor > 0


def expected_profiles(schedule: Sequence[Mapping[str, Any]], total_steps: int) -> list[str | None]:
    """Return the profile each step selects; None marks a dense step.

    This is the harness's own reading of the schedule rule. It does not call
    the production selector, so a production change shows as a mismatch.
    """
    if not _is_int(total_steps) or total_steps < 1:
        raise ConfigError(f"total_steps must be a positive integer, got {total_steps!r}")
    if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
        raise ConfigError("a schedule must be a list of ranges")
    ranges: list[tuple[int, int | None, str]] = []
    previous_end: int | None = 0
    for position, item in enumerate(schedule):
        if not isinstance(item, Mapping) or set(item) != {"start", "end", "profile"}:
            raise ConfigError(f"range {position} must have exactly the keys start, end and profile")
        start, end, profile = item["start"], item["end"], item["profile"]
        if not _is_int(start) or start < 0:
            raise ConfigError(f"range {position}: start must be a non-negative integer, got {start!r}")
        if end is not None and (not _is_int(end) or end <= start):
            raise ConfigError(f"range {position}: end must be null or an integer above start, got {end!r}")
        if not _valid_profile_name(profile):
            raise ConfigError(f"range {position}: profile must match [A-Za-z][A-Za-z0-9_-]*, got {profile!r}")
        if previous_end is None or start < previous_end:
            raise ConfigError(f"range {position}: ranges must be ordered and must not overlap")
        if start >= total_steps or (end is not None and end > total_steps):
            raise ConfigError(f"range {position}: the range lies outside the {total_steps} steps")
        previous_end = end
        ranges.append((start, end, profile))
    selected: list[str | None] = []
    for step in range(total_steps):
        name = None
        for start, end, profile in ranges:
            if start <= step and (end is None or step < end):
                name = profile
                break
        selected.append(name)
    return selected


def first_switch(expected: Sequence[str | None]) -> int | None:
    """Return the first step that selects a profile."""
    for step, name in enumerate(expected):
        if name is not None:
            return step
    return None


def _config_schedule(config: Mapping[str, Any], key: str, total_steps: int, allowed: Sequence[str]) -> list[str | None]:
    try:
        selected = expected_profiles(config[key], total_steps)
    except ConfigError as exc:
        raise ConfigError(f"{key}: {exc}") from exc
    unknown = sorted({name for name in selected if name is not None and name not in allowed})
    if unknown:
        raise ConfigError(f"{key}: profile(s) {', '.join(unknown)} are not declared")
    return selected


def validate_config(config: Mapping[str, Any], *, session: str | None = None) -> dict[str, Any]:
    """Check every config rule, fill the defaults and return a new dict."""
    if not isinstance(config, Mapping):
        raise ConfigError("the config must be a JSON object")
    if session is not None and session not in SESSIONS:
        raise ConfigError(f"session must be one of {', '.join(SESSIONS)}, got {session!r}")
    unknown = sorted(set(config) - set(_CONFIG_REQUIRED) - set(_CONFIG_OPTIONAL))
    if unknown:
        raise ConfigError(f"unknown config key(s): {', '.join(unknown)}")
    missing = [key for key in _CONFIG_REQUIRED if key not in config]
    if missing:
        raise ConfigError(f"missing config key(s): {', '.join(missing)}")
    try:
        out: dict[str, Any] = json.loads(json.dumps(dict(config)))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"the config is not JSON data: {exc}") from exc

    if out["schema"] != SCHEMA:
        raise ConfigError(f"schema must be {SCHEMA!r}, got {out['schema']!r}")
    combination = out["combination"]
    if not isinstance(combination, str) or combination not in COMBINATIONS:
        raise ConfigError(f"combination must be one of {', '.join(COMBINATIONS)}, got {combination!r}")
    for key, expected in zip(("model_family", "execution_mode", "approximation"), COMBINATIONS[combination]):
        if out[key] != expected:
            raise ConfigError(f"{key} must be {expected!r} for combination {combination}, got {out[key]!r}")

    omni_kwargs = out["omni_kwargs"]
    if not isinstance(omni_kwargs, dict) or not isinstance(omni_kwargs.get("model"), str) or not omni_kwargs["model"]:
        raise ConfigError("omni_kwargs must be an object with a non-empty string 'model'")
    if omni_kwargs.get("enforce_eager"):
        raise ConfigError("omni_kwargs.enforce_eager must not be true: an eager run is not compile evidence")
    if "diffusion_attention_schedule" in omni_kwargs:
        raise ConfigError("omni_kwargs.diffusion_attention_schedule is not allowed: use schedule_config")
    if omni_kwargs.get("num_gpus", 1) != 1:
        raise ConfigError("omni_kwargs.num_gpus must be 1: the probes reach the model in one process only")
    if omni_kwargs.get("distributed_executor_backend", "uni") != "uni":
        raise ConfigError("omni_kwargs.distributed_executor_backend must be 'uni'")
    if omni_kwargs.get("diffusion_compile_granularity", "regional") != "regional":
        raise ConfigError("omni_kwargs.diffusion_compile_granularity must be 'regional'")
    step_execution = omni_kwargs.get("step_execution", False)
    if not isinstance(step_execution, bool) or step_execution != (out["execution_mode"] == "step"):
        raise ConfigError(f"omni_kwargs.step_execution must be {out['execution_mode'] == 'step'} for {combination}")

    schedule_config = out["schedule_config"]
    if not isinstance(schedule_config, dict) or set(schedule_config) - {"profiles", "default"}:
        raise ConfigError("schedule_config must be an object with the keys profiles and default")
    profiles = schedule_config.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ConfigError("schedule_config.profiles must be a non-empty object")
    for name, spec in profiles.items():
        if not _valid_profile_name(name):
            raise ConfigError(f"schedule_config.profiles: the name {name!r} must match [A-Za-z][A-Za-z0-9_-]*")
        if not isinstance(spec, dict):
            raise ConfigError(f"schedule_config.profiles[{name!r}] must be an object")
    if schedule_config.setdefault("default", []) != []:
        raise ConfigError("schedule_config.default must be []: every request states its own schedule")

    approx_profiles = out["approx_profiles"]
    if not isinstance(approx_profiles, dict) or not approx_profiles:
        raise ConfigError("approx_profiles must be a non-empty object")
    for name, entry in approx_profiles.items():
        if name not in profiles:
            raise ConfigError(f"approx_profiles[{name!r}] is not a key of schedule_config.profiles")
        if not isinstance(entry, dict) or set(entry) != {"kind", "backend"}:
            raise ConfigError(f"approx_profiles[{name!r}] must have exactly the keys kind and backend")
        if entry["kind"] not in ("quantized", "sparse"):
            raise ConfigError(f"approx_profiles[{name!r}].kind must be 'quantized' or 'sparse'")
        if not isinstance(entry["backend"], str) or not entry["backend"]:
            raise ConfigError(f"approx_profiles[{name!r}].backend must be a non-empty string")

    total_steps = out["expected_total_steps"]
    if not _is_int(total_steps) or total_steps < 2:
        raise ConfigError("expected_total_steps must be an integer of at least 2")

    candidate = _config_schedule(out, "candidate_schedule", total_steps, list(approx_profiles))
    switch = first_switch(candidate)
    if switch is None or switch < 1:
        raise ConfigError("candidate_schedule: the first switch must be at step 1 or later, after a dense step")
    if approx_profiles[candidate[switch]]["kind"] != out["approximation"]:
        raise ConfigError(f"candidate_schedule: the profile at the first switch must be {out['approximation']}")

    warmup = _config_schedule(out, "warmup_schedule", total_steps, list(profiles))
    unused = sorted(set(profiles) - {name for name in warmup if name is not None})
    if unused:
        raise ConfigError(f"warmup_schedule: profile(s) {', '.join(unused)} run on no step")
    if None not in warmup:
        raise ConfigError("warmup_schedule: at least one step must stay dense")

    if out.setdefault("boundary_shift_schedule", None) is not None:
        shifted = _config_schedule(out, "boundary_shift_schedule", total_steps, list(approx_profiles))
        shifted_switch = first_switch(shifted)
        if shifted_switch is None or shifted_switch < 1 or shifted_switch == switch:
            raise ConfigError("boundary_shift_schedule: the first switch must be at step 1 or later and differ")
        if {name for name in shifted if name is not None} != {name for name in candidate if name is not None}:
            raise ConfigError("boundary_shift_schedule: the profiles must be those of candidate_schedule")

    sequences = out.setdefault("expected_sequences", 1)
    if not _is_int(sequences) or sequences < 1:
        raise ConfigError("expected_sequences must be a positive integer")
    frames = out.setdefault("expected_frames", None)
    if frames is not None and (not _is_int(frames) or frames < 1):
        raise ConfigError("expected_frames must be null or a positive integer")

    if not isinstance(out["prompt"], dict):
        raise ConfigError("prompt must be an object")
    sampling = out["sampling_params"]
    if not isinstance(sampling, dict):
        raise ConfigError("sampling_params must be an object")
    for key in ("seed", "num_inference_steps"):
        if not _is_int(sampling.get(key)):
            raise ConfigError(f"sampling_params.{key} must be an integer")
    for key in ("attention_schedule", "generator"):
        if key in sampling:
            raise ConfigError(f"sampling_params.{key} is not allowed: the harness sets the schedule and the seed")
    extra_args = sampling.get("extra_args")
    if isinstance(extra_args, dict) and "attention_schedule" in extra_args:
        raise ConfigError("sampling_params.extra_args.attention_schedule is not allowed")
    if sampling.get("output_type") not in (None, "np"):
        raise ConfigError("sampling_params.output_type must be 'np' or absent: the harness reads numpy frames")
    if sampling.get("emit_request_lifecycle"):
        raise ConfigError("sampling_params.emit_request_lifecycle must not be true: the harness reads one output")
    if isinstance(extra_args, dict) and extra_args.get("preencode_mp4"):
        raise ConfigError("sampling_params.extra_args.preencode_mp4 must not be true: the harness reads frames")

    prechange = out["prechange_source"]
    if not isinstance(prechange, dict) or set(prechange) - {"git_head", "package_tree_sha256"}:
        raise ConfigError("prechange_source must be an object with the keys git_head and package_tree_sha256")
    for key in ("git_head", "package_tree_sha256"):
        value = prechange.setdefault(key, None)
        if value is not None and (not isinstance(value, str) or not value):
            raise ConfigError(f"prechange_source.{key} must be null or a non-empty string")
    if prechange["git_head"] is None and prechange["package_tree_sha256"] is None:
        raise ConfigError("prechange_source must state git_head or package_tree_sha256 before the run")

    tolerance = out.setdefault("no_schedule_tolerance", None)
    if tolerance is not None:
        if not isinstance(tolerance, dict) or set(tolerance) != {"max_abs_diff_uint8"}:
            raise ConfigError("no_schedule_tolerance must be null or an object with max_abs_diff_uint8")
        if not _is_int(tolerance["max_abs_diff_uint8"]) or tolerance["max_abs_diff_uint8"] < 0:
            raise ConfigError("no_schedule_tolerance.max_abs_diff_uint8 must be a non-negative integer")

    lpips_config = out.setdefault("lpips", {"net": "alex"})
    if not isinstance(lpips_config, dict) or set(lpips_config) - {"net"}:
        raise ConfigError("lpips must be an object with the key net")
    if lpips_config.setdefault("net", "alex") not in ("alex", "vgg", "squeeze"):
        raise ConfigError("lpips.net must be 'alex', 'vgg' or 'squeeze'")
    if not _is_number(out["fps"]) or out["fps"] <= 0:
        raise ConfigError("fps must be a positive number")

    adapter = out.setdefault("adapter", None)
    if adapter is not None:
        if not isinstance(adapter, dict) or set(adapter) - {"pipeline_module", "pipeline_class"}:
            raise ConfigError("adapter must be an object with the keys pipeline_module and pipeline_class")
        for key, value in adapter.items():
            if not isinstance(value, str) or not value:
                raise ConfigError(f"adapter.{key} must be a non-empty string")
    if "_sha256" in out and not isinstance(out["_sha256"], str):
        raise ConfigError("_sha256 must be a string")
    return out


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a config file; '_sha256' holds the sha256 of the file bytes."""
    try:
        raw = Path(path).read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read config {os.fspath(path)}: {exc}") from exc
    if isinstance(data, dict):
        data.pop("_sha256", None)
    config = validate_config(data)
    config["_sha256"] = hashlib.sha256(raw).hexdigest()
    return config


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pointer(document: Any, pointer: str) -> Any:
    """Look up an RFC 6901 JSON pointer; raise KeyError(pointer) when a part is missing."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise KeyError(pointer)
    current = document
    for raw in pointer[1:].split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(pointer)
            current = current[part]
        elif isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                raise KeyError(pointer)
            current = current[int(part)]
        else:
            raise KeyError(pointer)
    return current


def tensor_digest(tensor: Any) -> dict[str, Any]:
    """Return sha256, shape, dtype and the NaN and Inf counts of a tensor."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise EvidenceError(f"cannot digest a {type(tensor).__name__}: a tensor is required")
    data = tensor.detach()
    nan = inf = 0
    if data.is_floating_point() or data.is_complex():
        nan = int(torch.isnan(data).sum().item())
        inf = int(torch.isinf(data).sum().item())
    data = data.to("cpu").contiguous()
    shape = [int(size) for size in data.shape]
    digest = hashlib.sha256(f"{data.dtype}|{shape}|".encode())
    if data.numel() > 0:
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"sha256": digest.hexdigest(), "shape": shape, "dtype": str(data.dtype), "nan": nan, "inf": inf}


def _digest_map(tensors: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: tensor_digest(value) for name, value in tensors.items()}


def _json_safe(value: Any) -> Any:
    """Return a copy that json.dump accepts: non-finite floats become null, unknown objects their repr."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _write_json(path: Path, data: Any) -> None:
    """Write through a temporary file so a reader never sees a partial record."""
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repository_root() -> Path:
    """Return the repository that contains this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def _inside_repository(path: Path) -> bool:
    root = _repository_root()
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def _owner_path(owner: Any) -> str:
    if inspect.ismodule(owner):
        return owner.__name__
    module = getattr(owner, "__module__", None)
    qualname = getattr(owner, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(owner)


def require_attr(owner: Any, name: str, *, where: str) -> Any:
    """Return getattr(owner, name); raise EvidenceError when the attribute is missing."""
    value = getattr(owner, name, _MISSING)
    if value is _MISSING:
        raise EvidenceError(f"{where}: {_owner_path(owner)} has no attribute {name}")
    return value


class PatchSet:
    """Attribute replacements and hook handles that restore() undoes."""

    def __init__(self) -> None:
        self.installed: list[dict[str, str]] = []
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._handles: list[Any] = []

    def add(self, owner: Any, name: str, replacement: Any, *, where: str, kind: str) -> Any:
        """Replace owner.name and return the original attribute."""
        original = require_attr(owner, name, where=where)
        own = vars(owner)
        had_own = name in own
        saved = own[name] if had_own else None
        setattr(owner, name, replacement)
        self._patches.append((owner, name, had_own, saved))
        self.installed.append({"target": f"{_owner_path(owner)}:{name}", "kind": kind})
        return original

    def add_handle(self, handle: Any) -> None:
        self._handles.append(handle)

    def restore(self) -> None:
        """Put back own attributes by value and remove attributes that were inherited."""
        for handle in reversed(self._handles):
            with contextlib.suppress(Exception):
                handle.remove()
        self._handles.clear()
        for owner, name, had_own, saved in reversed(self._patches):
            if had_own:
                setattr(owner, name, saved)
            else:
                with contextlib.suppress(AttributeError):
                    delattr(owner, name)
        self._patches.clear()

    def __enter__(self) -> PatchSet:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.restore()


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


def _new_counts() -> dict[str, int]:
    return {"block_calls": 0, "eager_block_calls": 0, "graph_execs": 0, "attention_layer_calls": 0, "kernel_calls": 0}


def _new_attention() -> dict[str, Any]:
    return {
        "layer_calls": 0,
        "selection": {},
        "backends": {},
        "outcome": {
            "baseline": 0,
            "unscheduled": 0,
            "approx_quantized": 0,
            "approx_sparse": 0,
            "approx_calls": 0,
            "fallback": dict.fromkeys(FALLBACK_REASONS, 0),
        },
        "kernel": {
            "calls": 0,
            "sage_calls": 0,
            "skip_calls": 0,
            "dense_calls": 0,
            "orphan_calls": 0,
            "skip_factor_min": None,
            "skip_factor_max": None,
        },
        "step_mismatch_calls": 0,
        "probe_errors": 0,
    }


def _classify_attention(info: Mapping[str, Any], kernel_calls: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the outcome counters one attention layer call adds to."""
    selection = info.get("selection")
    if selection == "<error>":
        # The selection is unknown, so the call counts only as a probe error.
        return []
    if selection == "<unscheduled>":
        return ["unscheduled"]
    if selection == "<baseline>":
        return ["baseline"]
    sage = any(call["sage"] for call in kernel_calls)
    skip = any(_is_skip_factor(call["skip_factor"]) for call in kernel_calls)
    if sage or skip:
        return [name for name, hit in (("approx_quantized", sage), ("approx_sparse", skip)) if hit] + ["approx_calls"]
    if not kernel_calls:
        return ["fallback.layer_fallback"]
    quant, skip_enabled, skip_configured = info.get("quant"), info.get("skip_enabled"), info.get("skip_configured")
    if quant is True:
        # Quantization is enabled on the implementation and kernel calls were seen, but none carried
        # SAGE scale factors. The current TRTLLM backend passes the scale factors on every kernel call
        # when quantization is enabled, so it is not expected to produce this state.
        return ["fallback.sage_not_applied"]
    if skip_configured is True and skip_enabled is False:
        return ["fallback.ignored_layer"]
    if skip_enabled is True:
        return ["fallback.private_gate"]
    if quant is False and skip_enabled is False and skip_configured is False:
        # A TRTLLM profile that configures only quantization is also counted here on a causal layer:
        # the backend disables quantization on causal layers when it builds the implementation.
        return ["fallback.profile_not_approximate"]
    return ["fallback.unclassified"]


class Recorder:
    """Process-wide capture store.

    The model runs on another thread of the harness process, so every method
    takes one lock and nothing is thread-local. Events outside a request go to
    ``outside_requests``; events inside a request but outside a step go to the
    request's ``unattributed`` counts.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter, sync: Callable[[], None] | None = None) -> None:
        self._lock = threading.RLock()
        self.clock = clock
        self._sync = sync
        self._attention_enabled = False
        self._request: dict[str, Any] | None = None
        self._sequence: dict[str, Any] | None = None
        self._step: dict[str, Any] | None = None
        self._outside_sequence = False
        self._outside_steps = 0
        self._setup_calls: list[dict[str, Any]] = []
        self._graphs_compiled = 0
        self._compile_attempts = 0
        self._compile_restarts = 0
        self._compile_errors = 0
        self._graph_execs = 0
        self._block_stack: list[tuple[int, int]] = []
        self._attention_stack: list[dict[str, Any]] = []
        self._outside = {"sequences": 0, "steps": 0, **_new_counts()}

    def sync(self) -> None:
        """Wait for queued device work, so the clock reads after the work is done."""
        if self._sync is not None:
            self._sync()

    def enable_attention(self) -> None:
        with self._lock:
            self._attention_enabled = True

    # -- requests, sequences and steps --

    def request_open(self) -> bool:
        with self._lock:
            return self._request is not None

    def begin_request(self, name: str) -> None:
        with self._lock:
            if self._request is not None:
                raise EvidenceError(f"request {self._request['name']} is still open")
            self._outside_sequence = False
            self._request = {
                "name": name,
                "sequences": [],
                "unattributed": _new_counts(),
                "graphs_compiled_before": self._graphs_compiled,
                "compile_before": self._compile_counts(),
            }

    def end_request(self) -> dict[str, Any]:
        with self._lock:
            request = self._request
            if request is None:
                raise EvidenceError("no request is open")
            self._close_step()
            if self._sequence is not None:
                self._sequence["incomplete"] = True
                if self._sequence["total_steps"] is None:
                    self._sequence["total_steps"] = len(self._sequence["steps"])
                self._sequence = None
            self._request = None
            return {
                "sequences": request["sequences"],
                "unattributed": request["unattributed"],
                "graphs_compiled_before": request["graphs_compiled_before"],
                "graphs_compiled_after": self._graphs_compiled,
                "compile_before": request["compile_before"],
                "compile_after": self._compile_counts(),
            }

    def begin_sequence(self, source: str, total_steps: int | None, initial: Mapping[str, Any]) -> None:
        with self._lock:
            if self._request is None:
                self._outside["sequences"] += 1
                self._outside_sequence = True
                self._outside_steps = 0
                return
            self._close_step()
            if self._sequence is not None:
                self._sequence["incomplete"] = True
                if self._sequence["total_steps"] is None:
                    self._sequence["total_steps"] = len(self._sequence["steps"])
            self._sequence = {
                "source": source,
                "total_steps": total_steps,
                "incomplete": False,
                "initial": dict(initial),
                "final": None,
                "steps": [],
            }
            self._request["sequences"].append(self._sequence)

    def end_sequence(self, final: Mapping[str, Any] | None) -> None:
        with self._lock:
            if self._request is None or self._sequence is None:
                self._outside_sequence = False
                return
            self._close_step()
            if self._sequence["total_steps"] is None:
                self._sequence["total_steps"] = len(self._sequence["steps"])
            self._sequence["final"] = dict(final) if final is not None else None
            self._sequence = None

    def sequence_open(self) -> bool:
        with self._lock:
            return self._sequence is not None or (self._request is None and self._outside_sequence)

    def steps_in_sequence(self) -> int:
        """Return how many steps the open sequence has, counting an open step."""
        with self._lock:
            if self._sequence is None:
                return self._outside_steps
            return len(self._sequence["steps"]) + (1 if self._step is not None else 0)

    def step_open(self) -> bool:
        with self._lock:
            return self._step is not None

    def begin_step(self, step: int, *, batch_size: int = 1, transformer: str | None = None) -> None:
        with self._lock:
            if self._request is None:
                if self._outside_sequence:
                    self._outside["steps"] += 1
                    self._outside_steps += 1
                return
            if self._sequence is None:
                return
            self._close_step()
            self.sync()
            self._step = {
                "step": int(step),
                "batch_size": int(batch_size),
                "transformer": transformer,
                "seconds": None,
                "digest_seconds": 0.0,
                "digests": {},
                "compile": {
                    "block_calls": 0,
                    "eager_block_calls": 0,
                    "graph_execs": 0,
                    "graphs_compiled": 0,
                    "by_model": {},
                },
                "attention": _new_attention() if self._attention_enabled else None,
                "_start": self.clock(),
            }

    def add_digests(self, digests: Mapping[str, Any], seconds: float) -> None:
        with self._lock:
            if self._step is None:
                return
            self._step["digests"].update(digests)
            self._step["digest_seconds"] += float(seconds)

    def end_step(self) -> None:
        with self._lock:
            self._close_step()

    def _close_step(self) -> None:
        step = self._step
        if step is None:
            return
        self.sync()
        step["seconds"] = self.clock() - step.pop("_start") - step["digest_seconds"]
        self._step = None
        if self._sequence is not None:
            self._sequence["steps"].append(step)

    # -- compile and block events --

    def note_setup(
        self,
        model_class: str,
        compiled_blocks: int,
        block_class_names: Sequence[str],
        kwargs: Mapping[str, Any],
        error: str | None = None,
    ) -> int:
        with self._lock:
            index = len(self._setup_calls)
            self._setup_calls.append(
                {
                    "index": index,
                    "model_class": model_class,
                    "compiled_blocks": int(compiled_blocks),
                    "block_class_names": list(block_class_names),
                    "kwargs": dict(kwargs),
                    "error": error,
                }
            )
            return index

    def _compile_counts(self) -> dict[str, int]:
        """Snapshot counters while the caller holds the recorder lock."""
        return {
            "compile_attempts": self._compile_attempts,
            "graphs_compiled": self._graphs_compiled,
            "compile_restarts": self._compile_restarts,
            "compile_errors": self._compile_errors,
        }

    def note_compile_attempt(self) -> None:
        with self._lock:
            self._compile_attempts += 1

    def note_compile_restart(self) -> None:
        with self._lock:
            self._compile_restarts += 1

    def note_graph_compiled(self) -> int:
        with self._lock:
            index = self._graphs_compiled
            self._graphs_compiled += 1
            if self._step is not None:
                self._step["compile"]["graphs_compiled"] += 1
            return index

    def note_compile_error(self) -> None:
        with self._lock:
            self._compile_errors += 1

    def _counts(self) -> dict[str, Any]:
        """Return the counters that events add to right now."""
        if self._step is not None:
            return self._step["compile"]
        if self._request is not None:
            return self._request["unattributed"]
        return self._outside

    def note_graph_executed(self, index: int) -> None:
        del index
        with self._lock:
            self._graph_execs += 1
            self._counts()["graph_execs"] += 1

    def block_enter(self, model_index: int) -> None:
        with self._lock:
            self._block_stack.append((model_index, self._graph_execs))

    def block_exit(self, model_index: int) -> None:
        with self._lock:
            executed = 0
            while self._block_stack:
                entered_index, execs_at_enter = self._block_stack.pop()
                if entered_index == model_index:
                    executed = self._graph_execs - execs_at_enter
                    break
            eager = executed == 0
            counts = self._counts()
            counts["block_calls"] += 1
            counts["eager_block_calls"] += int(eager)
            if self._step is not None:
                by_model = counts["by_model"].setdefault(str(model_index), {"block_calls": 0, "eager_block_calls": 0})
                by_model["block_calls"] += 1
                by_model["eager_block_calls"] += int(eager)

    # -- attention and kernel events --

    def attention_enter(self, info: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            call = {"info": dict(info), "kernel": []}
            self._attention_stack.append(call)
            return call

    def attention_exit(self, call: dict[str, Any]) -> None:
        with self._lock:
            # Remove this call by identity: two open calls can hold equal data.
            for position in range(len(self._attention_stack) - 1, -1, -1):
                if self._attention_stack[position] is call:
                    del self._attention_stack[position]
                    break
            attention = self._step["attention"] if self._step is not None else None
            if attention is None:
                if self._step is None:
                    counts = self._counts()
                    counts["attention_layer_calls"] += 1
                    counts["kernel_calls"] += len(call["kernel"])
                return
            info, kernel_calls = call["info"], call["kernel"]
            attention["layer_calls"] += 1
            selection = str(info.get("selection"))
            attention["selection"][selection] = attention["selection"].get(selection, 0) + 1
            backend = info.get("backend")
            if backend is not None:
                attention["backends"][str(backend)] = attention["backends"].get(str(backend), 0) + 1
            for counter in _classify_attention(info, kernel_calls):
                if counter.startswith("fallback."):
                    attention["outcome"]["fallback"][counter.partition(".")[2]] += 1
                else:
                    attention["outcome"][counter] += 1
            for kernel_call in kernel_calls:
                self._count_kernel_call(attention["kernel"], kernel_call)
            ctx_step = info.get("ctx_step")
            if ctx_step is not None and ctx_step != cast(dict[str, Any], self._step)["step"]:
                attention["step_mismatch_calls"] += 1
            if selection == "<error>":
                attention["probe_errors"] += 1

    @staticmethod
    def _count_kernel_call(kernel: dict[str, Any], call: Mapping[str, Any]) -> None:
        kernel["calls"] += 1
        factor = call["skip_factor"]
        skip = _is_skip_factor(factor)
        if call["sage"]:
            kernel["sage_calls"] += 1
        if skip:
            kernel["skip_calls"] += 1
            low, high = kernel["skip_factor_min"], kernel["skip_factor_max"]
            kernel["skip_factor_min"] = factor if low is None else min(low, factor)
            kernel["skip_factor_max"] = factor if high is None else max(high, factor)
        if not call["sage"] and not skip:
            kernel["dense_calls"] += 1

    def note_kernel_call(self, *, skip_factor: float | None, sage: bool) -> None:
        with self._lock:
            call = {"skip_factor": skip_factor, "sage": bool(sage)}
            if self._attention_stack:
                self._attention_stack[-1]["kernel"].append(call)
                return
            attention = self._step["attention"] if self._step is not None else None
            if attention is not None:
                attention["kernel"]["orphan_calls"] += 1
                self._count_kernel_call(attention["kernel"], call)
            elif self._step is None:
                self._counts()["kernel_calls"] += 1

    # -- summaries --

    def probes_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "setup_calls": copy.deepcopy(self._setup_calls),
                **self._compile_counts(),
                "outside_requests": dict(self._outside),
            }

    def totals(self) -> dict[str, int]:
        with self._lock:
            return {
                "setup_calls": len(self._setup_calls),
                **self._compile_counts(),
                "graph_execs": self._graph_execs,
            }


def _record_digests(recorder: Recorder, tensors: Mapping[str, Any]) -> None:
    """Add digests to the open step and report their cost, so step time excludes it."""
    if not recorder.step_open():
        return
    recorder.sync()
    started = recorder.clock()
    digests = _digest_map(tensors)
    recorder.add_digests(digests, recorder.clock() - started)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def install_compile_probe(patches: PatchSet, recorder: Recorder) -> None:
    """Count the graphs Inductor compiles and how often each compiled graph runs.

    The production ``torch.compile`` call is unchanged. Dynamo calls the patched
    ``__call__`` for each backend attempt and receives a closure around each
    successful callable. Tensorification restarts are counted separately, not
    suppressed: Dynamo must still receive the same exception and handle it.
    """
    import torch

    wrapper_cls = require_attr(torch, "_TorchCompileInductorWrapper", where="compile probe")
    original: Any = None

    def counting_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorder.note_compile_attempt()
        try:
            compiled = original(self, *args, **kwargs)
        except Exception as exc:
            # Do not import Dynamo or change its settings from this observer. If
            # this version has no recognized type, retain the error classification.
            restart_type = getattr(sys.modules.get("torch._dynamo.exc"), "TensorifyScalarRestartAnalysis", None)
            if isinstance(restart_type, type) and issubclass(restart_type, Exception) and isinstance(exc, restart_type):
                recorder.note_compile_restart()
            else:
                recorder.note_compile_error()
            raise
        index = recorder.note_graph_compiled()

        def run(*run_args: Any, **run_kwargs: Any) -> Any:
            recorder.note_graph_executed(index)
            return compiled(*run_args, **run_kwargs)

        # Dynamo reads attributes of the callable it receives. Its own markers and
        # __wrapped__ are left out, so it does not treat ``run`` as ``compiled``.
        for name, value in getattr(compiled, "__dict__", {}).items():
            if name.startswith("_torchdynamo") or name == "__wrapped__":
                continue
            with contextlib.suppress(Exception):
                setattr(run, name, value)
        return run

    original = patches.add(wrapper_cls, "__call__", counting_call, where="compile probe", kind="compile")


def _forward_slots(module: Any) -> tuple[Any, Any]:
    own = vars(module)
    return own.get("forward"), own.get("_omni_original_forward")


def install_block_probe(patches: PatchSet, recorder: Recorder) -> None:
    """Observe the runner's regionally_compile call and hook the blocks it compiled.

    The probe does not change the compile call. A block call during which no
    compiled graph ran counts as eager.
    """
    import torch

    runner = importlib.import_module(RUNNER_MODULE)
    original: Any = None

    def observing_regionally_compile(model: Any, *args: Any, **kwargs: Any) -> Any:
        model_class = type(model).__name__
        described = {name: repr(value) for name, value in kwargs.items()}
        before = [(module, _forward_slots(module)) for module in model.modules()]
        try:
            result = original(model, *args, **kwargs)
        except Exception as exc:
            recorder.note_setup(model_class, 0, [], described, error=f"{type(exc).__name__}: {exc}")
            raise
        blocks = []
        for module, (forward, original_forward) in before:
            now_forward, now_original_forward = _forward_slots(module)
            if now_forward is not forward or now_original_forward is not original_forward:
                blocks.append(module)
        class_names = sorted({type(block).__name__ for block in blocks})
        index = recorder.note_setup(model_class, len(blocks), class_names, described)

        def on_enter(_module: Any, _args: Any) -> None:
            if not torch.compiler.is_compiling():
                recorder.block_enter(index)

        def on_exit(_module: Any, _args: Any, _output: Any) -> None:
            if not torch.compiler.is_compiling():
                recorder.block_exit(index)

        for block in blocks:
            patches.add_handle(block.register_forward_pre_hook(on_enter))
            patches.add_handle(block.register_forward_hook(on_exit, always_call=True))
        return result

    original = patches.add(
        runner, "regionally_compile", observing_regionally_compile, where="block probe", kind="block"
    )


def _attention_info(layer: Any, is_available: Callable[[], bool], get_context: Callable[[], Any]) -> dict[str, Any]:
    """Describe the implementation one attention layer call will use.

    ``selection`` is ``<unscheduled>`` for a layer without a schedule,
    ``<baseline>`` when the layer's own implementation runs, the profile names
    that share the selected implementation joined with ``|``, ``<unknown>`` when
    no prepared candidate matches the selected implementation, and ``<error>``
    when the layer's selection raised.

    ``effective_attention`` returns the layer's own implementation or the
    implementation of a prepared candidate, so ``<unknown>`` does not occur with
    the current layer code. The value is kept for a later change of that method.
    """
    info: dict[str, Any] = {
        "selection": "<unscheduled>",
        "backend": None,
        "quant": None,
        "skip_enabled": None,
        "skip_configured": None,
        "ctx_step": None,
    }
    with contextlib.suppress(Exception):
        if is_available():
            info["ctx_step"] = getattr(get_context(), "denoise_step_idx", None)
    impl = getattr(layer, "attention", None)
    backend_cls = getattr(layer, "attn_backend", None)
    if getattr(layer, "_schedule_configured", False):
        try:
            impl, backend_cls, _spec = layer.effective_attention()
        except Exception:
            info["selection"] = "<error>"
            return info
        if impl is getattr(layer, "attention", None):
            info["selection"] = "<baseline>"
        else:
            candidates = getattr(layer, "_schedule_candidates", None) or {}
            names = sorted(name for name, record in candidates.items() if getattr(record, "impl", None) is impl)
            info["selection"] = "|".join(names) if names else "<unknown>"
    with contextlib.suppress(Exception):
        if backend_cls is not None:
            info["backend"] = backend_cls.get_name()
    quant = getattr(getattr(impl, "quant", None), "enabled", None)
    skip = getattr(impl, "skip", None)
    skip_enabled, skip_configured = getattr(skip, "enabled", None), getattr(skip, "configured", None)
    info["quant"] = None if quant is None else bool(quant)
    info["skip_enabled"] = None if skip_enabled is None else bool(skip_enabled)
    info["skip_configured"] = None if skip_configured is None else bool(skip_configured)
    return info


def install_attention_probe(patches: PatchSet, recorder: Recorder) -> None:
    """Record, per attention layer call, which implementation the layer selected."""
    import torch

    layer_module = importlib.import_module(ATTENTION_MODULE)
    context_module = importlib.import_module(FORWARD_CONTEXT_MODULE)
    attention_cls = require_attr(layer_module, "Attention", where="attention probe")
    require_attr(attention_cls, "effective_attention", where="attention probe")
    is_available = require_attr(context_module, "is_forward_context_available", where="attention probe")
    get_context = require_attr(context_module, "get_forward_context", where="attention probe")
    original: Any = None

    def probed_run_local_attention(self: Any, *args: Any, **kwargs: Any) -> Any:
        if torch.compiler.is_compiling():
            return original(self, *args, **kwargs)
        call = recorder.attention_enter(_attention_info(self, is_available, get_context))
        try:
            return original(self, *args, **kwargs)
        finally:
            recorder.attention_exit(call)

    original = patches.add(
        attention_cls, "_run_local_attention", probed_run_local_attention, where="attention probe", kind="attention"
    )
    recorder.enable_attention()


def install_trtllm_kernel_probe(patches: PatchSet, recorder: Recorder) -> dict[str, Any]:
    """Record the approximation arguments each TRTLLM kernel call receives."""
    import torch

    module = importlib.import_module(TRTLLM_MODULE)
    original = require_attr(module, TRTLLM_KERNEL, where="kernel probe")

    @functools.wraps(original)
    def probed_kernel(*args: Any, **kwargs: Any) -> Any:
        if torch.compiler.is_compiling():
            return original(*args, **kwargs)
        factor = kwargs.get("skip_softmax_threshold_scale_factor")
        if factor is not None:
            try:
                factor = float(factor)
            except (TypeError, ValueError):
                factor = float("nan")
        recorder.note_kernel_call(skip_factor=factor, sage=kwargs.get("sage_attn_sfs") is not None)
        return original(*args, **kwargs)

    patches.add(module, TRTLLM_KERNEL, probed_kernel, where="kernel probe", kind="kernel")
    return {
        "target": f"{module.__name__}:{TRTLLM_KERNEL}",
        "module": getattr(original, "__module__", None),
        "qualname": getattr(original, "__qualname__", None),
    }


def install_h3_request_adapter(patches: PatchSet, recorder: Recorder, *, module_path: str = H3_PIPELINE_MODULE) -> None:
    """Capture each step of the MiniMax-H3 request-mode loop through its own callbacks."""
    module = importlib.import_module(module_path)
    name = "minimax_h3_denoise_loop"
    original = require_attr(module, name, where="h3 request adapter")
    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"h3 request adapter: cannot read the signature of {module_path}.{name}") from exc
    missing = [parameter for parameter in H3_LOOP_PARAMETERS if parameter not in parameters]
    if missing:
        raise EvidenceError(f"h3 request adapter: {module_path}.{name} lacks parameter(s) {', '.join(missing)}")

    def capturing_loop(*args: Any, **kwargs: Any) -> Any:
        sigmas = kwargs.get("sigmas_video")
        recording = recorder.request_open()
        initial: dict[str, Any] = {}
        if recording:
            initial_rows = {"video": kwargs.get("initial_video_rows"), "audio": kwargs.get("initial_audio_rows")}
            initial = _digest_map(initial_rows)
        recorder.begin_sequence("h3_request_loop", len(sigmas) - 1 if sigmas is not None else None, initial)
        caller_on_step = kwargs.get("on_step")
        caller_profiler = kwargs.get("step_profiler")

        def on_step(step: int, video_rows: Any, audio_rows: Any) -> None:
            if caller_on_step is not None:
                caller_on_step(step, video_rows, audio_rows)
            _record_digests(recorder, {"video": video_rows, "audio": audio_rows})

        @contextlib.contextmanager
        def step_profiler(step: int) -> Any:
            inner = caller_profiler(step) if caller_profiler is not None else contextlib.nullcontext()
            recorder.begin_step(int(step))
            try:
                with inner:
                    yield
            finally:
                recorder.end_step()

        kwargs["on_step"] = on_step
        kwargs["step_profiler"] = step_profiler
        result = original(*args, **kwargs)
        final = None
        if recording and isinstance(result, tuple) and len(result) == 2:
            final = _digest_map({"video": result[0], "audio": result[1]})
        recorder.end_sequence(final)
        return result

    patches.add(module, name, capturing_loop, where="h3 request adapter", kind="step")


def install_h3_step_adapter(
    patches: PatchSet,
    recorder: Recorder,
    *,
    module_path: str = H3_PIPELINE_MODULE,
    class_name: str = H3_PIPELINE_CLASS,
) -> None:
    """Capture each step of MiniMax-H3 step mode at denoise_step and step_scheduler."""
    module = importlib.import_module(module_path)
    pipeline_cls = require_attr(module, class_name, where="h3 step adapter")
    audio_key = require_attr(module, "_STEP_AUDIO_ROWS", where="h3 step adapter")
    originals: dict[str, Any] = {}

    def state_tensors(state: Any) -> dict[str, Any]:
        return {"video": state.latents, "audio": state.extra[audio_key]}

    def capturing_denoise_step(self: Any, input_batch: Any, *args: Any, **kwargs: Any) -> Any:
        states = kwargs.get("states")
        batch_states = list(states if states is not None else input_batch.states)
        if batch_states:
            state = batch_states[0]
            step = int(state.step_index)
            if not recorder.sequence_open() or step == 0:
                initial = _digest_map(state_tensors(state)) if recorder.request_open() else {}
                recorder.begin_sequence("h3_step_scheduler", None, initial)
            recorder.begin_step(step, batch_size=len(batch_states))
        return originals["denoise_step"](self, input_batch, *args, **kwargs)

    def capturing_step_scheduler(self: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
        result = originals["step_scheduler"](self, state, *args, **kwargs)
        _record_digests(recorder, state_tensors(state))
        recorder.end_step()
        return result

    def capturing_post_decode(self: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
        recorder.end_sequence(None)
        return originals["post_decode"](self, state, *args, **kwargs)

    replacements: dict[str, Callable[..., Any]] = {
        "denoise_step": capturing_denoise_step,
        "step_scheduler": capturing_step_scheduler,
        "post_decode": capturing_post_decode,
    }
    for name in replacements:
        require_attr(pipeline_cls, name, where="h3 step adapter")
    for name, replacement in replacements.items():
        originals[name] = patches.add(pipeline_cls, name, replacement, where="h3 step adapter", kind="step")


def _first_tensor(value: Any) -> Any:
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    return value


def install_wan_request_adapter(
    patches: PatchSet,
    recorder: Recorder,
    *,
    module_path: str = WAN_PIPELINE_MODULE,
    class_name: str = WAN_PIPELINE_CLASS,
) -> None:
    """Capture each Wan2.2 step at noise prediction: latent, conditioning, timestep and output.

    The capture is at predict_noise_maybe_with_cfg because the DMD branch of
    diffuse skips the scheduler step. A step ends when the next one begins.
    Conditioning and timestep tensors are hashed when supplied, on both CFG
    branches. The original tensors and arguments are passed through unchanged.
    """
    import torch

    module = importlib.import_module(module_path)
    pipeline_cls = require_attr(module, class_name, where="wan request adapter")
    originals: dict[str, Any] = {
        "diffuse": require_attr(pipeline_cls, "diffuse", where="wan request adapter"),
        "predict": require_attr(pipeline_cls, "predict_noise_maybe_with_cfg", where="wan request adapter"),
    }
    try:
        diffuse_signature = inspect.signature(originals["diffuse"])
        predict_signature = inspect.signature(originals["predict"])
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"wan request adapter: cannot read a signature on {module_path}.{class_name}") from exc
    required_parameters = ((diffuse_signature, ("latents", "timesteps")), (predict_signature, ("positive_kwargs",)))
    for signature, required in required_parameters:
        missing = [parameter for parameter in required if parameter not in signature.parameters]
        if missing:
            raise EvidenceError(f"wan request adapter: {module_path}.{class_name} lacks parameter(s) {missing}")

    def capturing_diffuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        arguments = diffuse_signature.bind(self, *args, **kwargs).arguments
        recording = recorder.request_open()
        initial = _digest_map({"latents": arguments["latents"]}) if recording else {}
        recorder.begin_sequence("wan_request_diffuse", len(arguments["timesteps"]), initial)
        result = originals["diffuse"](self, *args, **kwargs)
        final = None
        if recording and isinstance(result, torch.Tensor):
            final = _digest_map({"latents": result})
        recorder.end_sequence(final)
        return result

    def capturing_predict(self: Any, *args: Any, **kwargs: Any) -> Any:
        if torch.compiler.is_compiling() or not recorder.sequence_open():
            return originals["predict"](self, *args, **kwargs)
        arguments = predict_signature.bind(self, *args, **kwargs).arguments
        positive = arguments.get("positive_kwargs") or {}
        negative = arguments.get("negative_kwargs") or {}
        model = positive.get("current_model")
        label = None
        if model is not None:
            if model is getattr(self, "transformer", None):
                label = "transformer"
            elif model is getattr(self, "transformer_2", None):
                label = "transformer_2"
            else:
                label = f"<other:{type(model).__name__}>"
        recorder.begin_step(recorder.steps_in_sequence(), transformer=label)
        inputs = {"latent_in": positive.get("hidden_states")}
        for branch, supplied in (("positive", positive), ("negative", negative)):
            for key in ("timestep", "encoder_hidden_states"):
                value = supplied.get(key)
                if value is not None:
                    if key == "timestep" and isinstance(value, torch.Tensor):
                        # A singleton expand has stride 0 even when contiguous;
                        # a dtype view needs canonical strides for byte hashing.
                        value = value.detach().clone(memory_format=torch.contiguous_format)
                    inputs[f"{branch}_{key}"] = value
        if negative.get("hidden_states") is not None:
            inputs["negative_latent_in"] = negative["hidden_states"]
        _record_digests(recorder, inputs)
        result = originals["predict"](self, *args, **kwargs)
        _record_digests(recorder, {"noise_pred": _first_tensor(result)})
        return result

    patches.add(pipeline_cls, "diffuse", capturing_diffuse, where="wan request adapter", kind="step")
    patches.add(
        pipeline_cls, "predict_noise_maybe_with_cfg", capturing_predict, where="wan request adapter", kind="step"
    )


def install_probes(
    config: Mapping[str, Any], session: str, recorder: Recorder
) -> tuple[PatchSet, dict[str, Any] | None]:
    """Install the probes of one session; restore everything when a target is missing."""
    patches = PatchSet()
    kernel_info = None
    adapter = config.get("adapter") or {}
    module_override = {"module_path": adapter["pipeline_module"]} if "pipeline_module" in adapter else {}
    class_override = {"class_name": adapter["pipeline_class"]} if "pipeline_class" in adapter else {}
    try:
        install_compile_probe(patches, recorder)
        install_block_probe(patches, recorder)
        if config["model_family"] == "minimax_h3":
            install_h3_request_adapter(patches, recorder, **module_override)
            install_h3_step_adapter(patches, recorder, **module_override, **class_override)
        else:
            install_wan_request_adapter(patches, recorder, **module_override, **class_override)
        if session == "scheduled":
            install_attention_probe(patches, recorder)
            if any(entry["backend"] == KERNEL_PROBE_BACKEND for entry in config["approx_profiles"].values()):
                kernel_info = install_trtllm_kernel_probe(patches, recorder)
    except ImportError as exc:
        patches.restore()
        raise EvidenceError(f"a probe target cannot be imported: {type(exc).__name__}: {exc}") from exc
    except BaseException:
        patches.restore()
        raise
    return patches, kernel_info


# ---------------------------------------------------------------------------
# Frames, LPIPS and video
# ---------------------------------------------------------------------------


def extract_frames(outputs: Any) -> Any:
    """Return the frames [F, H, W, C] of the value Omni.generate returns."""
    import numpy as np
    import torch

    first = outputs[0] if isinstance(outputs, (list, tuple)) and outputs else outputs
    error = getattr(first, "error", None)
    if error:
        raise EvidenceError(f"the request returned an error: {error}")
    images = getattr(first, "images", None)
    if not images:
        raise EvidenceError("could not extract frames: the first output has no images")
    frames = images[0]
    # Audio and video models can return a dict or a (video, audio) tuple.
    if isinstance(frames, dict):
        frames = frames.get("video") if frames.get("video") is not None else frames.get("frames")
    elif isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]
    if frames is None:
        raise EvidenceError("could not extract frames: the output holds no video")
    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            video = video[0]
        if video.dim() == 4:
            channels_first = video.shape[0] in (3, 4)
            channels_last = video.shape[-1] in (1, 3, 4)
            if channels_first and channels_last:
                raise EvidenceError(
                    f"could not extract frames: the layout of a tensor of shape {list(video.shape)} is ambiguous"
                )
            if channels_first:
                video = video.permute(1, 2, 3, 0)
        if video.is_floating_point():
            # [-1, 1] to [0, 1]. Values outside the range and NaN are kept for frames_to_uint8 to count.
            array = (video.float() * 0.5 + 0.5).numpy()
        else:
            array = video.numpy()
    else:
        array = np.asarray(frames)
        if array.ndim == 5:
            array = array[0]
    if array.ndim != 4:
        raise EvidenceError(f"could not extract frames: got an array of shape {list(array.shape)}")
    return array


def frames_to_uint8(frames: Any) -> tuple[Any, dict[str, Any]]:
    """Return uint8 frames [F, H, W, 3] and a description of the input."""
    import numpy as np

    array = np.asarray(frames)
    if array.ndim != 4:
        raise EvidenceError(f"frames must have 4 dimensions, got shape {list(array.shape)}")
    source_dtype = str(array.dtype)
    shape = [int(size) for size in array.shape]
    raw_digest = hashlib.sha256(f"{source_dtype}|{shape}|".encode())
    raw_digest.update(np.ascontiguousarray(array).tobytes())
    nan_count = inf_count = 0
    if array.dtype == np.uint8:
        out = array
    elif np.issubdtype(array.dtype, np.floating):
        values = array.astype(np.float32)
        nan_count = int(np.isnan(values).sum())
        inf_count = int(np.isinf(values).sum())
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        raise EvidenceError(f"frames must be uint8 or floating point, got {source_dtype}")
    if out.shape[-1] == 4:
        out = out[..., :3]
    elif out.shape[-1] == 1:
        out = np.repeat(out, 3, axis=-1)
    if out.shape[-1] != 3:
        raise EvidenceError(f"frames must have 1, 3 or 4 channels in the last axis, got shape {shape}")
    info = {
        "raw": {
            "sha256": raw_digest.hexdigest(),
            "shape": shape,
            "dtype": source_dtype,
            "nan": nan_count,
            "inf": inf_count,
        },
        "shape": [int(size) for size in out.shape],
        "dtype": "uint8",
        "source_dtype": source_dtype,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    return np.ascontiguousarray(out), info


def check_frame_pair(
    reference_info: Mapping[str, Any], candidate_info: Mapping[str, Any], expected_frames: int | None
) -> list[str]:
    """Return why two outputs cannot be compared; an empty list means they can."""
    reference_shape = list(reference_info.get("shape") or [])
    candidate_shape = list(candidate_info.get("shape") or [])
    reference_frames = reference_shape[0] if reference_shape else None
    candidate_frames = candidate_shape[0] if candidate_shape else None
    problems = []
    if reference_frames is None or reference_frames != candidate_frames:
        problems.append("missing_frames")
    elif expected_frames is not None and reference_frames != expected_frames:
        problems.append("missing_frames")
    if reference_shape[1:] != candidate_shape[1:]:
        problems.append("shape_mismatch")
    if reference_info.get("nan_count", 0) or candidate_info.get("nan_count", 0):
        problems.append("nan")
    if reference_info.get("inf_count", 0) or candidate_info.get("inf_count", 0):
        problems.append("inf")
    return problems


def lpips_metadata(net: str) -> dict[str, Any]:
    """Return the installed lpips version and the sha256 of its linear-layer weights."""
    out: dict[str, Any] = {"package_version": None, "weights_sha256": None}
    try:
        import lpips
    except Exception:
        return out
    with contextlib.suppress(Exception):
        out["package_version"] = importlib.metadata.version("lpips")
    weights = Path(lpips.__file__).resolve().parent / "weights" / "v0.1" / f"{net}.pth"
    if weights.is_file():
        out["weights_sha256"] = sha256_file(weights)
    return out


def _scalar(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def lpips_video(
    reference: Any,
    candidate: Any,
    *,
    net: str = "alex",
    loss_fn: Callable[..., Any] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Return the mean LPIPS distance of two videos; frame i is compared with frame i, without resizing.

    Without ``loss_fn`` the function builds the network and scores on
    ``device``. Without ``device`` it scores on the GPU when one is available;
    if that fails, for example because the service holds the GPU memory, it
    scores on the CPU. The result names the device that gave the scores.
    With ``loss_fn`` the inputs stay on the CPU and ``device`` is not used.
    """
    import numpy as np
    import torch

    reference, candidate = np.asarray(reference), np.asarray(candidate)
    aligned = (
        reference.dtype == np.uint8
        and candidate.dtype == np.uint8
        and reference.ndim == 4
        and reference.shape == candidate.shape
        and reference.shape[0] > 0
    )
    if not aligned:
        raise EvidenceError("lpips inputs are not aligned")

    def to_input(frame: Any, device: str) -> Any:
        return (torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0).to(device)

    def score(network: Callable[..., Any], device: str) -> list[float]:
        values = []
        with torch.no_grad():
            for index in range(reference.shape[0]):
                values.append(_scalar(network(to_input(reference[index], device), to_input(candidate[index], device))))
        return values

    used = "cpu"
    if loss_fn is not None:
        per_frame = score(loss_fn, used)
    else:
        try:
            import lpips

            network = lpips.LPIPS(net=net).eval()
        except Exception as exc:
            raise EvidenceError(f"lpips_unavailable: {type(exc).__name__}: {exc}") from exc
        if device is not None:
            order: tuple[str, ...] = (device,)
        else:
            order = ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",)
        failure: Exception | None = None
        per_frame = []
        for used in order:
            try:
                per_frame = score(network.to(used), used)
            except Exception as exc:
                failure = exc
                continue
            failure = None
            break
        if failure is not None:
            raise EvidenceError(f"lpips_failed: {type(failure).__name__}: {failure}") from failure
    return {
        "value": sum(per_frame) / len(per_frame),
        "per_frame": per_frame,
        "frames": len(per_frame),
        "net": net,
        "device": used,
    }


def side_by_side(reference: Any, candidate: Any, labels: Sequence[str]) -> Any:
    """Return frames with the reference on the left, the candidate on the right and a label band on top."""
    import numpy as np
    from PIL import Image, ImageDraw

    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 4 or reference.dtype != np.uint8:
        raise EvidenceError("side-by-side inputs must be uint8 frames of the same shape")
    if candidate.dtype != np.uint8 or reference.shape[-1] != 3 or len(labels) != 2:
        raise EvidenceError("side-by-side inputs must be uint8 RGB frames with two labels")
    frames, height, width, _ = reference.shape
    header = Image.new("RGB", (2 * width, LABEL_BAND), (0, 0, 0))
    draw = ImageDraw.Draw(header)
    draw.text((4, 8), str(labels[0]), fill=(255, 255, 255))
    draw.text((width + 4, 8), str(labels[1]), fill=(255, 255, 255))
    out = np.empty((frames, height + LABEL_BAND, 2 * width, 3), dtype=np.uint8)
    out[:, :LABEL_BAND] = np.asarray(header, dtype=np.uint8)
    out[:, LABEL_BAND:, :width] = reference
    out[:, LABEL_BAND:, width:] = candidate
    return out


def write_video(frames: Any, path: str | os.PathLike[str], fps: float) -> None:
    from diffusers.utils import export_to_video

    export_to_video(list(frames.astype("float32") / 255.0), os.fspath(path), fps=fps)


# ---------------------------------------------------------------------------
# Source and environment
# ---------------------------------------------------------------------------


def package_tree_sha256(directory: str | os.PathLike[str]) -> str:
    """Return one sha256 over the path and content of every *.py file under a package directory."""
    root = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()):
        digest.update(f"{path.relative_to(root).as_posix()}\0{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _git(repository: Path, *args: str) -> bytes | None:
    try:
        completed = subprocess.run(["git", "-C", str(repository), *args], capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def collect_source() -> dict[str, Any]:
    """Identify the harness file and the imported vllm_omni package."""
    out: dict[str, Any] = {
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "package_file": None,
        "package_tree_sha256": None,
        "git_head": None,
        "git_diff_sha256": None,
        "git_status_clean": None,
    }
    try:
        import vllm_omni
    except Exception:
        return out
    package_file = Path(vllm_omni.__file__).resolve()
    out["package_file"] = str(package_file)
    out["package_tree_sha256"] = package_tree_sha256(package_file.parent)
    repository = package_file.parent.parent
    head = _git(repository, "rev-parse", "HEAD")
    diff = _git(repository, "diff", "HEAD")
    status = _git(repository, "status", "--porcelain")
    if head is not None:
        out["git_head"] = head.decode("utf-8", "replace").strip()
    if diff is not None:
        out["git_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    if status is not None:
        out["git_status_clean"] = not status.strip()
    return out


def _package_version(*names: str) -> str | None:
    for name in names:
        with contextlib.suppress(Exception):
            return importlib.metadata.version(name)
    return None


def collect_environment() -> dict[str, Any]:
    """Describe the software and the GPU. Environment variables pass a whitelist, so no token is stored."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in ENV_NAMES or any(name.startswith(prefix) for prefix in ENV_PREFIXES)
    }
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
        "cudnn": None,
        "vllm": _package_version("vllm"),
        "flashinfer": _package_version("flashinfer-python", "flashinfer"),
        "lpips": _package_version("lpips"),
        "gpu": [],
        "determinism": None,
        "env": environment,
    }
    try:
        import torch
    except Exception:
        return out
    out["torch"] = str(torch.__version__)
    out["cuda"] = torch.version.cuda
    with contextlib.suppress(Exception):
        out["cudnn"] = torch.backends.cudnn.version()
    with contextlib.suppress(Exception):
        out["determinism"] = {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        }
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            for index in range(torch.accelerator.device_count()):
                out["gpu"].append(
                    {
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
                    }
                )
    return out


def _device_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.accelerator.synchronize()


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _compile_self_test(device: str) -> str:
    """Compile a small function with the default backend and count its three calls."""
    import torch

    recorder = Recorder()
    with PatchSet() as patches:
        install_compile_probe(patches, recorder)

        def toy(value: Any) -> Any:
            scaled = torch.sin(value) * 2.0
            return scaled + 1.0

        compiled = torch.compile(toy)
        sample = torch.ones(4, device=device)
        recorder.begin_request("preflight")
        for _ in range(3):
            compiled(sample)
        captured = recorder.end_request()
    graphs = recorder.totals()["graphs_compiled"]
    executions = captured["unattributed"]["graph_execs"]
    if graphs < 1 or executions != 3:
        raise EvidenceError(f"compile probe saw {graphs} graph(s) and {executions} execution(s), expected >=1 and 3")
    return f"{graphs} graph(s), {executions} executions"


def _block_self_test(device: str) -> str:
    """Run a toy model through the real regionally_compile under the block probe."""
    import torch
    from torch import nn

    class _ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, value: Any) -> Any:
            return torch.relu(self.linear(value))

    class _ToyModel(nn.Module):
        _repeated_blocks = ["_ToyBlock"]

        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_ToyBlock(), _ToyBlock()])

        def forward(self, value: Any) -> Any:
            for block in self.blocks:
                value = block(value)
            return value

    recorder = Recorder()
    calls = 3
    with PatchSet() as patches:
        install_compile_probe(patches, recorder)
        install_block_probe(patches, recorder)
        runner = importlib.import_module(RUNNER_MODULE)
        model = runner.regionally_compile(_ToyModel().to(device), dynamic=False)
        sample = torch.ones(2, 4, device=device)
        recorder.begin_request("preflight")
        recorder.begin_sequence("preflight", calls, {})
        with torch.no_grad():
            for step in range(calls):
                recorder.begin_step(step)
                model(sample)
                recorder.end_step()
        recorder.end_sequence(None)
        captured = recorder.end_request()
    setup_calls = recorder.probes_snapshot()["setup_calls"]
    steps = captured["sequences"][0]["steps"]
    block_calls = sum(step["compile"]["block_calls"] for step in steps)
    eager = sum(step["compile"]["eager_block_calls"] for step in steps)
    compiled_blocks = setup_calls[0]["compiled_blocks"] if setup_calls else 0
    if compiled_blocks < 1 or block_calls != calls * 2 or eager != 0:
        raise EvidenceError(
            f"block probe saw {compiled_blocks} compiled block(s), {block_calls} block call(s) and {eager} eager "
            f"call(s), expected 2, {calls * 2} and 0"
        )
    return f"{compiled_blocks} compiled blocks, {block_calls} block calls, 0 eager"


def _comparison_self_test(net: str, device: str | None = None) -> str:
    """Run LPIPS, the side-by-side layout and the video writer on two small frames.

    The scheduled session compares its outputs after the last request. This
    check shows a missing package or a video writer without an encoder before
    the service starts. With ``device`` the LPIPS score runs on that device
    only; without it, on the device ``lpips_video`` chooses.
    """
    import numpy as np

    frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    try:
        metadata = lpips_metadata(net)
        missing = [key for key in ("package_version", "weights_sha256") if not metadata.get(key)]
        if missing:
            raise EvidenceError(f"lpips metadata is missing: {', '.join(missing)}")
        score = lpips_video(frames, frames, net=net, device=device)
        combined = side_by_side(frames, frames, ["reference", "candidate"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "self-test.mp4"
            write_video(combined, target, 8)
            if not target.is_file() or target.stat().st_size == 0:
                raise EvidenceError("the video writer wrote no file")
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError(f"the comparison tools failed: {type(exc).__name__}: {exc}") from exc
    return f"lpips {metadata['package_version']} ({net}) on {score.get('device')}; side-by-side video written"


def preflight(config: Mapping[str, Any], *, session: str = "scheduled", device: str = "cpu") -> dict[str, Any]:
    """Check the probe targets, the compile probes and, for the scheduled session, the comparison tools.

    Loads no model weights. LPIPS can download its backbone weights once.
    The two compile self-tests and the LPIPS score run on ``device``.
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, action: Callable[[], Any]) -> bool:
        try:
            detail = action()
        except Exception as exc:
            checks.append({"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
            return False
        checks.append({"name": name, "ok": True, "detail": "" if detail is None else str(detail)})
        return True

    validated: dict[str, Any] = {}

    def check_config() -> str:
        validated.update(validate_config(config, session=session))
        return f"combination {validated['combination']}, session {session}"

    def check_targets() -> str:
        patches, kernel_info = install_probes(validated, session, Recorder())
        installed = [entry["target"] for entry in patches.installed]
        patches.restore()
        return f"{len(installed)} target(s): {', '.join(installed)}; kernel: {kernel_info}"

    if check("config", check_config):
        check("probe_targets", check_targets)
        if session == "scheduled":
            check("comparison_self_test", lambda: _comparison_self_test(validated["lpips"]["net"], device))
    check("compile_self_test", lambda: _compile_self_test(device))
    check("block_self_test", lambda: _block_self_test(device))
    return {"ok": all(entry["ok"] for entry in checks), "checks": checks, "source": collect_source()}


# ---------------------------------------------------------------------------
# Run one session
# ---------------------------------------------------------------------------


def _create_omni(omni_kwargs: Mapping[str, Any]) -> Any:
    from vllm_omni.entrypoints.omni import Omni

    return Omni(**omni_kwargs)


def _build_sampling_params(sampling_params: Mapping[str, Any], schedule: list | None) -> Any:
    """Build the sampling params; the schedule field is sent only when schedule is not None."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    kwargs = copy.deepcopy(dict(sampling_params))
    if schedule is not None:
        kwargs["attention_schedule"] = copy.deepcopy(schedule)
    return OmniDiffusionSamplingParams(**kwargs)


def read_effective_config(omni: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Read back the diffusion config the service runs with. Never raises."""
    try:
        clients = getattr(getattr(omni, "engine", None), "stage_clients", None) or []
        for client in clients:
            if getattr(client, "stage_type", None) != "diffusion":
                continue
            od_config = getattr(client, "od_config", None)
            if od_config is None:
                return None, "the diffusion stage client has no od_config"
            schedule = getattr(od_config, "diffusion_attention_schedule", None)
            profiles = getattr(schedule, "profiles", None)
            eager = getattr(od_config, "enforce_eager", None)
            return {
                "client_class": type(client).__name__,
                "enforce_eager": None if eager is None else bool(eager),
                "step_execution": getattr(od_config, "step_execution", None),
                "distributed_executor_backend": getattr(od_config, "distributed_executor_backend", None),
                "num_gpus": getattr(od_config, "num_gpus", None),
                "max_num_seqs": getattr(od_config, "max_num_seqs", None),
                "diffusion_compile_granularity": getattr(od_config, "diffusion_compile_granularity", None),
                "diffusion_compile_dynamic": getattr(od_config, "diffusion_compile_dynamic", None),
                "cache_backend": getattr(od_config, "cache_backend", None),
                "schedule_profiles": sorted(profiles) if isinstance(profiles, Mapping) else None,
                "parallel_config": repr(getattr(od_config, "parallel_config", None)),
            }, None
        return None, "no diffusion stage client was found on omni.engine.stage_clients"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _session_requests(config: Mapping[str, Any], session: str) -> list[tuple[str, list | None]]:
    """Return (name, schedule) of the requests of one session; None means the field is not sent."""
    if session != "scheduled":
        return [("warmup", None), ("plain", None)]
    requests: list[tuple[str, list | None]] = [
        ("warmup", config["warmup_schedule"]),
        ("reference_a", []),
        ("reference_b", []),
        ("candidate", config["candidate_schedule"]),
    ]
    if config["boundary_shift_schedule"] is not None:
        requests.append(("boundary_shift", config["boundary_shift_schedule"]))
    return requests


def _request_sums(captured: Mapping[str, Any]) -> dict[str, int]:
    sums = {"steps": 0, "block_calls": 0, "attention_layer_calls": 0, "kernel_calls": 0}
    for sequence in captured["sequences"]:
        for step in sequence["steps"]:
            sums["steps"] += 1
            sums["block_calls"] += step["compile"]["block_calls"]
            if step["attention"] is not None:
                sums["attention_layer_calls"] += step["attention"]["layer_calls"]
                sums["kernel_calls"] += step["attention"]["kernel"]["calls"]
    return sums


def _selection_pairs(sequences: Any) -> set[tuple[str, str]]:
    """Return (transformer, selection name) for every attention selection the steps recorded.

    A selection name is a profile name or one of the markers the attention
    probe writes, such as ``<baseline>`` for a dense call of a scheduled layer.
    """
    pairs: set[tuple[str, str]] = set()
    for sequence in sequences or []:
        for step in sequence.get("steps") or []:
            selection = (step.get("attention") or {}).get("selection") or {}
            for key in selection:
                pairs.update((str(step.get("transformer")), name) for name in str(key).split("|"))
    return pairs


def _warmup_coverage_gaps(
    warmup: Mapping[str, Any], schedules: Sequence[Sequence[Mapping[str, Any]]], total_steps: int
) -> list[str]:
    """Return "transformer:selection" for each selection a later request runs first.

    Each transformer is compiled on its own. The code after an attention call
    can compile again the first time a profile, or the dense path, runs on a
    transformer. The warm-up request must run every selection that a later
    request runs, on the transformer that runs it there.
    """
    covered = _selection_pairs(warmup["sequences"])
    transformers: dict[int, set[str]] = {}
    for sequence in warmup["sequences"]:
        for step in sequence["steps"]:
            transformers.setdefault(step["step"], set()).add(str(step["transformer"]))
    needed: set[tuple[str, str]] = set()
    for schedule in schedules:
        for index, name in enumerate(expected_profiles(schedule, total_steps)):
            needed.update((label, name or "<baseline>") for label in transformers.get(index, ()))
    return sorted(f"{label}:{name}" for label, name in needed - covered)


def _check_reach_after_warmup(
    entry: Mapping[str, Any],
    session: str,
    kernel_probe: bool,
    schedules: Sequence[Sequence[Mapping[str, Any]]],
    total_steps: int,
) -> None:
    """Stop the session when the warm-up request shows that the later requests cannot be evidence.

    ``schedules`` are the schedules of the requests that follow the warm-up.
    """
    sums = _request_sums(entry)
    problems = []
    if sums["steps"] < 1:
        problems.append("no denoise step was captured")
    if entry["graphs_compiled_after"] < 1:
        problems.append("no graph was compiled")
    if sums["block_calls"] < 1:
        problems.append("no call of a compiled block was captured")
    if session == "scheduled" and sums["attention_layer_calls"] < 1:
        problems.append("no attention layer call was captured")
    if session == "scheduled" and kernel_probe and sums["kernel_calls"] < 1:
        problems.append("no kernel call was captured")
    if problems:
        raise EvidenceError("the probes did not reach the model during the warm-up request: " + "; ".join(problems))
    if session == "scheduled":
        gaps = _warmup_coverage_gaps(entry, schedules, total_steps)
        if gaps:
            raise EvidenceError(
                "the warm-up request did not run every selection of the later requests on the transformer "
                f"that runs it there: {', '.join(gaps)}; change warmup_schedule"
            )


def _store_output(out_path: Path, session: str, name: str, array: Any, info: Mapping[str, Any], fps: float) -> dict:
    import numpy as np

    output = dict(info)
    frames_file = f"{session}-{name}.frames.npy"
    np.save(out_path / frames_file, array)
    output["frames_file"] = frames_file
    output["frames_sha256"] = sha256_file(out_path / frames_file)
    video_file = f"{session}-{name}.mp4"
    try:
        write_video(array, out_path / video_file, fps)
        output.update(video_file=video_file, video_sha256=sha256_file(out_path / video_file), video_error=None)
    except Exception as exc:
        output.update(video_file=None, video_sha256=None, video_error=f"{type(exc).__name__}: {exc}")
    return output


def _compare_candidate(
    out_path: Path,
    config: Mapping[str, Any],
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    frames: Mapping[str, Any],
) -> dict[str, Any]:
    net = config["lpips"]["net"]
    lpips_block: dict[str, Any] = {
        "value": None,
        "error": None,
        "per_frame": [],
        "frames": 0,
        "net": net,
        "package_version": None,
        "weights_sha256": None,
        "frame_alignment": "index",
        "resize": "none",
        "input_scaling": "uint8/127.5-1",
        "aggregation": "mean_over_frames",
    }
    comparison: dict[str, Any] = {"lpips": lpips_block, "side_by_side": None, "side_by_side_error": None}
    problems = check_frame_pair(reference, candidate, config["expected_frames"])
    if problems:
        # No score is computed for outputs that cannot be compared frame by frame.
        lpips_block["error"] = problems[0]
        return comparison
    try:
        lpips_block.update(lpips_metadata(net))
        lpips_block.update(lpips_video(frames["reference_a"], frames["candidate"], net=net))
    except Exception as exc:
        # The requests have run. A failed score is recorded, and the session record is still written.
        failed = str(exc) if isinstance(exc, EvidenceError) else f"lpips_failed: {type(exc).__name__}: {exc}"
        lpips_block["error"] = failed
    schedule_text = json.dumps(config["candidate_schedule"], sort_keys=True)
    labels = ["all-dense reference", f"{config['combination']} {schedule_text}"]
    try:
        combined = side_by_side(frames["reference_a"], frames["candidate"], labels)
        file_name = "scheduled-side_by_side.mp4"
        write_video(combined, out_path / file_name, config["fps"])
        comparison["side_by_side"] = {
            "file": file_name,
            "sha256": sha256_file(out_path / file_name),
            "labels": labels,
            "fps": float(config["fps"]),
            "frames": int(combined.shape[0]),
        }
    except Exception as exc:
        comparison["side_by_side_error"] = f"{type(exc).__name__}: {exc}"
    return comparison


def _find_request(record: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for request in record.get("requests") or []:
        if isinstance(request, Mapping) and request.get("name") == name:
            return request
    return None


def _compare_to_prior(prior: str | os.PathLike[str], array: Any) -> dict[str, Any]:
    """Compare the plain output with the plain output of an earlier session record, frame by frame."""
    block: dict[str, Any] = {
        "prior_file": os.fspath(prior),
        "prior_sha256": None,
        "prior_session_id": None,
        "prior_frames_sha256": None,
        "error": None,
        "bit_identical": False,
        "max_abs_diff_uint8": None,
        "mean_abs_diff_uint8": None,
        "differing_frames": None,
    }
    try:
        import numpy as np

        prior_path = Path(prior)
        raw = prior_path.read_bytes()
        block["prior_sha256"] = hashlib.sha256(raw).hexdigest()
        prior_record = json.loads(raw.decode("utf-8"))
        # The verdict compares these two values with the pre-change record it is given.
        block["prior_session_id"] = prior_record.get("session_id")
        request = _find_request(prior_record, "plain")
        output = request.get("output") if request is not None else None
        if not output:
            raise EvidenceError("the prior record has no plain output")
        block["prior_frames_sha256"] = output["frames_sha256"]
        frames_path = prior_path.parent / output["frames_file"]
        if sha256_file(frames_path) != output["frames_sha256"]:
            raise EvidenceError("the prior frames file does not match its recorded sha256")
        prior_frames = np.load(frames_path)
        if prior_frames.shape != array.shape:
            block["error"] = "shape_mismatch"
            return block
        maximum, total, differing = 0, 0.0, 0
        for index in range(array.shape[0]):
            difference = np.abs(array[index].astype(np.int16) - prior_frames[index].astype(np.int16))
            frame_max = int(difference.max()) if difference.size else 0
            maximum = max(maximum, frame_max)
            total += float(difference.sum())
            differing += int(frame_max > 0)
        block["bit_identical"] = maximum == 0
        block["max_abs_diff_uint8"] = maximum
        block["mean_abs_diff_uint8"] = total / array.size if array.size else 0.0
        block["differing_frames"] = differing
    except Exception as exc:
        block["error"] = f"{type(exc).__name__}: {exc}"
    return block


def _new_record(config: dict[str, Any], session: str, argv: Sequence[str] | None) -> dict[str, Any]:
    stored_config = {key: value for key, value in config.items() if key != "_sha256"}
    config_sha256 = config.get("_sha256")
    if config_sha256 is None:
        config_sha256 = hashlib.sha256(json.dumps(stored_config, sort_keys=True).encode()).hexdigest()
    return {
        "schema": SCHEMA,
        "kind": "session",
        "session": session,
        "session_id": uuid.uuid4().hex,
        "status": "aborted",
        "abort": None,
        "exit_code": None,
        "argv": list(argv) if argv is not None else list(sys.argv),
        "started_utc": _utc_now(),
        "finished_utc": None,
        "config": stored_config,
        "config_sha256": config_sha256,
        "source": None,
        "environment": None,
        "startup": {
            "omni_kwargs": None,
            "cold_start_seconds": None,
            "effective": None,
            "effective_missing_reason": None,
        },
        "probes": {
            "installed": [],
            "compile": {
                "mechanism": COMPILE_MECHANISM,
                "counter_schema": COMPILE_COUNTER_SCHEMA,
                "setup_calls": [],
                "compile_attempts": 0,
                "graphs_compiled": 0,
                "compile_restarts": 0,
                "compile_errors": 0,
            },
            "kernel": None,
            "outside_requests": None,
        },
        "requests": [],
        "comparison": None,
        "comparison_to_prior": None,
    }


def run_session(
    config: Mapping[str, Any],
    *,
    session: str,
    out_dir: str | os.PathLike[str],
    prior: str | os.PathLike[str] | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    """Start one service, send the requests of one session and write the session record.

    Returns the exit code. When the record was written, prints one JSON line
    with the record path and the exit code. Prints no verdict.
    """
    try:
        validated = validate_config(config, session=session)
        if prior is not None and session != "plain":
            raise ConfigError("--prior is accepted only with --session plain")
        if prior is not None and not Path(prior).is_file():
            raise ConfigError(f"--prior {os.fspath(prior)} is not a file")
        out_path = Path(out_dir).resolve()
        record_path = out_path / SESSION_FILES[session]
        if _inside_repository(out_path):
            raise ConfigError(f"the evidence directory {out_path} is inside the repository")
        if record_path.exists():
            raise ConfigError(f"{record_path} exists: records of earlier runs are kept, use another directory")
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"the evidence directory {out_path} cannot be created: {exc}") from exc
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    record = _new_record(validated, session, argv)
    recorder = Recorder(sync=_device_sync)
    patches: PatchSet | None = None
    omni: Any = None
    stage = "install"
    exit_code = 4
    try:
        record["source"] = collect_source()
        record["environment"] = collect_environment()
        if session == "scheduled":
            # Before the probes are installed, so the check adds nothing to the compile counters.
            stage = "comparison_self_test"
            _comparison_self_test(validated["lpips"]["net"])
            stage = "install"
        patches, kernel_info = install_probes(validated, session, recorder)
        record["probes"]["installed"] = list(patches.installed)
        record["probes"]["kernel"] = kernel_info

        # Built before the service starts: a field the service does not accept shows here.
        stage = "sampling_params"
        requests = _session_requests(validated, session)
        params_by_name = {
            name: _build_sampling_params(validated["sampling_params"], schedule) for name, schedule in requests
        }
        later_schedules = [schedule for name, schedule in requests if name != "warmup" and schedule is not None]

        stage = "startup"
        omni_kwargs = copy.deepcopy(validated["omni_kwargs"])
        omni_kwargs["enforce_eager"] = False
        omni_kwargs["distributed_executor_backend"] = "uni"
        if session == "scheduled":
            omni_kwargs["diffusion_attention_schedule"] = copy.deepcopy(validated["schedule_config"])
        record["startup"]["omni_kwargs"] = copy.deepcopy(omni_kwargs)
        started = time.perf_counter()
        omni = _create_omni(omni_kwargs)
        record["startup"]["cold_start_seconds"] = time.perf_counter() - started
        effective, reason = read_effective_config(omni)
        record["startup"]["effective"] = effective
        record["startup"]["effective_missing_reason"] = reason

        stage = "reach_after_startup"
        stages = getattr(omni, "num_stages", 1)
        if stages != 1:
            raise EvidenceError(f"the service has {stages} stages: the harness supports one stage")
        setup_calls = recorder.probes_snapshot()["setup_calls"]
        if not setup_calls:
            raise EvidenceError(
                "regionally_compile was not called in this process: enforce_eager is set, the platform lacks "
                "Inductor support, or the diffusion stage runs in another process"
            )
        for call in setup_calls:
            if call["error"] is not None or call["compiled_blocks"] < 1:
                reason = call["error"] or "no block was compiled"
                raise EvidenceError(f"regionally_compile did not compile {call['model_class']}: {reason}")

        frames: dict[str, Any] = {}
        entries: dict[str, dict[str, Any]] = {}
        for name, schedule in requests:
            stage = "request"
            entry: dict[str, Any] = {
                "name": name,
                "attention_schedule": copy.deepcopy(schedule),
                "prompt": copy.deepcopy(validated["prompt"]),
                "sampling_params": copy.deepcopy(validated["sampling_params"]),
                "error": None,
                "wall_seconds": None,
                "graphs_compiled_before": None,
                "graphs_compiled_after": None,
                "compile_before": None,
                "compile_after": None,
                "sequences": [],
                "unattributed": None,
                "output": None,
            }
            record["requests"].append(entry)
            entries[name] = entry
            params = params_by_name[name]
            failure: Exception | None = None
            outputs: Any = None
            recorder.begin_request(name)
            started = time.perf_counter()
            try:
                outputs = omni.generate(copy.deepcopy(validated["prompt"]), params)
            except Exception as exc:
                failure = exc
            finally:
                entry["wall_seconds"] = time.perf_counter() - started
                try:
                    entry.update(recorder.end_request())
                except Exception as exc:
                    # The error of the request itself, if there is one, stays the reported failure.
                    entry["capture_error"] = f"{type(exc).__name__}: {exc}"
                    failure = failure or exc
            if failure is not None:
                entry["error"] = {"error_type": type(failure).__name__, "message": str(failure)}
                raise failure
            if name == "warmup":
                stage = "reach_after_warmup"
                _check_reach_after_warmup(
                    entry, session, kernel_info is not None, later_schedules, validated["expected_total_steps"]
                )
                continue
            try:
                array, info = frames_to_uint8(extract_frames(outputs))
            except EvidenceError as exc:
                entry["error"] = {"error_type": type(exc).__name__, "message": str(exc)}
                raise
            entry["output"] = _store_output(out_path, session, name, array, info, validated["fps"])
            if name in ("reference_a", "candidate", "plain"):
                frames[name] = array
            del outputs, array

        stage = "comparison"
        if session == "scheduled":
            record["comparison"] = _compare_candidate(
                out_path, validated, entries["reference_a"]["output"], entries["candidate"]["output"], frames
            )
        if session == "plain" and prior is not None:
            record["comparison_to_prior"] = _compare_to_prior(prior, frames["plain"])
        record["status"] = "completed"
        exit_code = 0
    except EvidenceError as exc:
        record["abort"] = {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)}
        exit_code = 3
    except Exception as exc:
        record["abort"] = {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)}
        exit_code = 4
    finally:
        if omni is not None:
            with contextlib.suppress(Exception):
                omni.close()
        if patches is not None:
            patches.restore()
        snapshot = recorder.probes_snapshot()
        record["probes"]["compile"].update(
            setup_calls=snapshot["setup_calls"],
            graphs_compiled=snapshot["graphs_compiled"],
            compile_attempts=snapshot["compile_attempts"],
            compile_restarts=snapshot["compile_restarts"],
            compile_errors=snapshot["compile_errors"],
        )
        record["probes"]["outside_requests"] = snapshot["outside_requests"]
        record["exit_code"] = exit_code
        record["finished_utc"] = _utc_now()
        try:
            _write_json(record_path, record)
        except OSError as exc:
            # Without its record a session shows nothing, whatever its requests did.
            print(f"error: the session record {record_path} was not written: {exc}", file=sys.stderr)
            exit_code = exit_code or 4
        else:
            print(json.dumps({"session_file": str(record_path), "exit_code": exit_code}))
    return exit_code


# ---------------------------------------------------------------------------
# Verdict rules
# ---------------------------------------------------------------------------


class _MissingPointerError(Exception):
    def __init__(self, reference: str) -> None:
        super().__init__(reference)
        self.reference = reference


class _Reader:
    """Reads one session record by JSON pointer and lists the pointers a rule relied on."""

    def __init__(self, document: Any, file_name: str) -> None:
        self.document = document
        self.file_name = file_name
        self.evidence: list[str] = []

    def get(self, pointer: str, *, note: bool = True) -> Any:
        reference = f"{self.file_name}#{pointer}"
        try:
            value = resolve_pointer(self.document, pointer)
        except KeyError:
            raise _MissingPointerError(reference) from None
        if note and reference not in self.evidence:
            self.evidence.append(reference)
        return value

    def request_index(self, name: str) -> int:
        requests = self.get("/requests", note=False)
        matches = [
            index
            for index, request in enumerate(requests if isinstance(requests, list) else [])
            if isinstance(request, Mapping) and request.get("name") == name
        ]
        if len(matches) != 1:
            raise _MissingPointerError(f"{self.file_name}#/requests[name={name}]")
        return matches[0]


def _item(
    status: str, reasons: Sequence[str] = (), evidence: Sequence[str] = (), values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {"status": status, "reasons": list(reasons), "evidence": list(evidence), "values": dict(values or {})}


def _decide(
    fail: Sequence[str], unproven: Sequence[str], evidence: Sequence[str], values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """A failed condition outranks an unproven one; no reason at all is a pass."""
    if fail:
        return _item("fail", fail, evidence, values)
    if unproven:
        return _item("not_proven", unproven, evidence, values)
    return _item("pass", (), evidence, values)


def _usable_digests(digests: Any) -> bool:
    return isinstance(digests, Mapping) and bool(digests)


class _Scheduled:
    """Values the rules derive from the config stored in a scheduled session record."""

    def __init__(self, reader: _Reader, *, note: bool = False) -> None:
        self.reader = reader
        self.total_steps = reader.get("/config/expected_total_steps", note=note)
        self.sequences = reader.get("/config/expected_sequences", note=note)
        self.approximation = reader.get("/config/approximation", note=note)
        self.candidate_schedule = reader.get("/config/candidate_schedule", note=note)
        self.approx_profiles = reader.get("/config/approx_profiles", note=note)
        self.expected = expected_profiles(self.candidate_schedule, self.total_steps)
        self.switch = first_switch(self.expected)
        self.indices = {name: reader.request_index(name) for name in SCHEDULED_REQUESTS}

    def steps_pointer(self, request: str, sequence: int) -> str:
        return f"/requests/{self.indices[request]}/sequences/{sequence}/steps"

    def step_pointer(self, request: str, sequence: int, step: int) -> str:
        return f"{self.steps_pointer(request, sequence)}/{step}"


def _sequence_problems(reader: _Reader, index: int, source: str, total_steps: int, sequences: int) -> list[str]:
    """Return what is wrong with the captured sequences of one request."""
    problems = []
    captured = reader.get(f"/requests/{index}/sequences")
    if not isinstance(captured, list) or len(captured) != sequences:
        return ["sequence_count"]
    for sequence in captured:
        if sequence.get("source") != source:
            problems.append("sequence_source")
        if sequence.get("incomplete") is not False:
            problems.append("sequence_incomplete")
        if sequence.get("total_steps") != total_steps:
            problems.append("total_steps")
        steps = sequence.get("steps")
        if not isinstance(steps, list) or [step.get("step") for step in steps] != list(range(total_steps)):
            problems.append("step_indices")
    return problems


def _rule_capture(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when the scheduled record is complete: expected requests, sequences and steps, no call outside a step."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    reasons = []
    if reader.get("/schema") != SCHEMA:
        reasons.append("schema")
    if reader.get("/session") != "scheduled":
        reasons.append("session")
    if reader.get("/status") != "completed":
        reasons.append("status")
    row = tuple(reader.get(f"/config/{key}") for key in ("model_family", "execution_mode", "approximation"))
    if COMBINATIONS.get(reader.get("/config/combination")) != row:
        reasons.append("combination_row")
    context = _Scheduled(reader, note=True)
    names = [request.get("name") for request in reader.get("/requests", note=False)]
    if names[: len(SCHEDULED_REQUESTS)] != list(SCHEDULED_REQUESTS) or names[4:] not in ([], ["boundary_shift"]):
        reasons.append("request_order")
    source = SEQUENCE_SOURCES.get(row[:2])
    for name in ("reference_a", "reference_b", "candidate"):
        index = context.indices[name]
        if reader.get(f"/requests/{index}/error") is not None:
            reasons.append(f"request_error:{name}")
        problems = _sequence_problems(reader, index, str(source), context.total_steps, context.sequences)
        reasons.extend(f"{problem}:{name}" for problem in problems)
        unattributed = reader.get(f"/requests/{index}/unattributed")
        for key in ("block_calls", "attention_layer_calls", "kernel_calls"):
            if unattributed.get(key) != 0:
                reasons.append(f"unattributed_{key}:{name}")
    return _decide((), reasons, reader.evidence)


def _rule_conditions(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when reference_a and candidate differ in the schedule only: same prompt, parameters, start state, shape."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    reference, candidate = context.indices["reference_a"], context.indices["candidate"]
    fail, unproven = [], []
    session_id = reader.get("/session_id")
    if not isinstance(session_id, str) or not session_id:
        unproven.append("session_id")
    if reader.get(f"/requests/{reference}/attention_schedule") != []:
        fail.append("reference_schedule_not_empty")
    if reader.get(f"/requests/{candidate}/attention_schedule") != reader.get("/config/candidate_schedule"):
        fail.append("candidate_schedule_differs")
    if reader.get(f"/requests/{reference}/prompt") != reader.get(f"/requests/{candidate}/prompt"):
        fail.append("prompt_differs")
    if reader.get(f"/requests/{reference}/sampling_params") != reader.get(f"/requests/{candidate}/sampling_params"):
        fail.append("sampling_params_differ")
    for sequence in range(context.sequences):
        reference_base = f"/requests/{reference}/sequences/{sequence}"
        candidate_base = f"/requests/{candidate}/sequences/{sequence}"
        initial_reference = reader.get(f"{reference_base}/initial")
        initial_candidate = reader.get(f"{candidate_base}/initial")
        if not _usable_digests(initial_reference) or not _usable_digests(initial_candidate):
            unproven.append("initial_state_not_captured")
        elif initial_reference != initial_candidate:
            fail.append("initial_state_differs")
        # The capture rule requires the expected total on both requests, so this is false
        # when evaluate_combination applies this rule.
        if reader.get(f"{reference_base}/total_steps") != reader.get(f"{candidate_base}/total_steps"):
            fail.append("total_steps_differ")
    reference_shape = reader.get(f"/requests/{reference}/output/shape")
    if not reference_shape or reference_shape != reader.get(f"/requests/{candidate}/output/shape"):
        fail.append("output_shape_differs")
    return _decide(sorted(set(fail)), sorted(set(unproven)), reader.evidence)


def _first_difference(
    reader: _Reader, context: _Scheduled, left: str, right: str, sequence: int, stop: int
) -> tuple[int | None, bool]:
    """Return the first step below stop whose digests differ, and whether every digest was usable.

    A step without usable digests on either side is not compared: it shows
    neither a difference nor equality.
    """
    usable = True
    for step in range(stop):
        left_digests = reader.get(context.step_pointer(left, sequence, step) + "/digests")
        right_digests = reader.get(context.step_pointer(right, sequence, step) + "/digests")
        if not _usable_digests(left_digests) or not _usable_digests(right_digests):
            usable = False
            continue
        if left_digests != right_digests:
            return step, usable
    return None, usable


def _rule_dense_repeatable(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when reference_a and reference_b, both dense, give equal step digests, final states and output bytes."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    first, second = context.indices["reference_a"], context.indices["reference_b"]
    fail, unproven, values = [], [], {}  # type: list[str], list[str], dict[str, int]
    if reader.get(f"/requests/{second}/attention_schedule") != []:
        fail.append("reference_schedule_not_empty")
    for sequence in range(context.sequences):
        differing, usable = _first_difference(
            reader, context, "reference_a", "reference_b", sequence, context.total_steps
        )
        if not usable:
            unproven.append("digests_not_captured")
        if differing is not None:
            fail.append("reference_not_repeatable")
            values.setdefault("first_differing_step", differing)
        final_first = reader.get(f"/requests/{first}/sequences/{sequence}/final")
        final_second = reader.get(f"/requests/{second}/sequences/{sequence}/final")
        if final_first != final_second:
            fail.append("reference_not_repeatable")
    raw_first = reader.get(f"/requests/{first}/output/raw")
    raw_second = reader.get(f"/requests/{second}/output/raw")
    if not raw_first.get("sha256") or raw_first.get("sha256") != raw_second.get("sha256"):
        fail.append("reference_not_repeatable")
    return _decide(sorted(set(fail)), sorted(set(unproven)), reader.evidence, values)


def _selection_problem(attention: Any, expected: str | None) -> str | None:
    """Return why one step's recorded selection does not match the expected profile."""
    if not isinstance(attention, Mapping):
        return "attention_not_recorded"
    if attention.get("layer_calls", 0) < 1:
        return "no_layer_calls"
    if attention.get("probe_errors", 0) != 0:
        return "probe_errors"
    if attention.get("step_mismatch_calls", 0) != 0:
        return "step_mismatch"
    keys = set(attention.get("selection") or {}) - {"<unscheduled>"}
    if expected is None:
        if keys != {"<baseline>"}:
            return "expected_baseline"
        kernel = attention.get("kernel") or {}
        if kernel.get("sage_calls", 0) or kernel.get("skip_calls", 0):
            # The layers ran their own implementation, and that implementation is approximate.
            return "approximate_kernel_on_dense_step"
        return None
    if not keys or keys & {"<baseline>", "<error>", "<unknown>"}:
        return "expected_profile"
    if any(expected not in key.split("|") for key in keys):
        return "other_profile"
    return None


def _selection_mismatches(
    reader: _Reader, context: _Scheduled, request: str, expected: Sequence[str | None]
) -> list[dict[str, Any]]:
    mismatches = []
    for sequence in range(context.sequences):
        for step in range(context.total_steps):
            attention = reader.get(context.step_pointer(request, sequence, step) + "/attention")
            problem = _selection_problem(attention, expected[step])
            if problem is not None:
                mismatches.append({"sequence": sequence, "step": step, "problem": problem})
    return mismatches


def _rule_profile_switch(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when every step selects as scheduled and the first profile step runs the target kind and differs first."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    reader.get("/config/candidate_schedule")
    switch = context.switch
    if switch is None or switch < 1:
        return _item("fail", ["selection_mismatch"], reader.evidence, {"first_switch": switch})
    mismatches = _selection_mismatches(reader, context, "candidate", context.expected)
    mismatches += _selection_mismatches(reader, context, "reference_a", [None] * context.total_steps)
    if mismatches:
        return _item("fail", ["selection_mismatch"], reader.evidence, {"steps": mismatches})
    fail, unproven, values = [], [], {"first_switch": switch}  # type: list[str], list[str], dict[str, int | None]
    for sequence in range(context.sequences):
        differing, usable = _first_difference(reader, context, "candidate", "reference_a", sequence, switch + 1)
        values.setdefault("first_differing_step", differing)
        if not usable:
            unproven.append("digests_not_captured")
        if differing is not None and differing < switch:
            fail.append("diverged_before_switch")
        elif differing is None:
            unproven.append("no_divergence_at_switch")
        outcome = reader.get(context.step_pointer("candidate", sequence, switch) + "/attention/outcome")
        if outcome.get(f"approx_{context.approximation}", 0) < 1:
            unproven.append("target_kind_not_observed_at_switch")
    return _decide(sorted(set(fail)), sorted(set(unproven)), reader.evidence, values)


def _rule_dense_prefix(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when the candidate's digests equal those of reference_a on every step before the first profile step."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    switch = context.switch
    if switch is None or switch < 1:
        # No dense step runs before the first profile step, so there is no prefix to compare.
        reader.get("/config/candidate_schedule")
        return _item("not_proven", ["no_dense_prefix"], reader.evidence, {"first_switch": switch})
    # Wan records the latent that enters a step, so the input of the switch step is part of the prefix.
    has_step_input = reader.get("/config/model_family", note=False) == "wan2_2"
    fail, unproven, values = [], [], {"first_switch": switch}
    for sequence in range(context.sequences):
        differing, usable = _first_difference(reader, context, "candidate", "reference_a", sequence, switch)
        if not usable:
            unproven.append("digests_not_captured")
        if differing is not None:
            fail.append("prefix_differs")
            values.setdefault("first_differing_step", differing)
            continue
        if has_step_input:
            latent_candidate = reader.get(context.step_pointer("candidate", sequence, switch) + "/digests/latent_in")
            latent_reference = reader.get(context.step_pointer("reference_a", sequence, switch) + "/digests/latent_in")
            if latent_candidate != latent_reference:
                fail.append("prefix_differs")
                values.setdefault("first_differing_step", switch)
    return _decide(sorted(set(fail)), sorted(set(unproven)), reader.evidence, values)


def _plain_capture_ok(reader: _Reader) -> tuple[bool, int | None]:
    """Check a plain or pre-change record the way the capture rule checks a scheduled one."""
    if reader.get("/status") != "completed":
        return False, None
    index = reader.request_index("plain")
    if reader.get(f"/requests/{index}/error") is not None:
        return False, index
    row = (reader.get("/config/model_family"), reader.get("/config/execution_mode"))
    problems = _sequence_problems(
        reader,
        index,
        str(SEQUENCE_SOURCES.get(row)),
        reader.get("/config/expected_total_steps"),
        reader.get("/config/expected_sequences"),
    )
    return not problems, index


def _rule_no_schedule(
    scheduled: Mapping[str, Any] | None, plain: Mapping[str, Any] | None, prechange: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Pass when plain equals the confirmed pre-change session bit for bit, or differs within the stated tolerance."""
    if plain is None or prechange is None:
        return _item("not_measured", ["plain_or_prechange_session_absent"])
    plain_reader = _Reader(plain, SESSION_FILES["plain"])
    prior_reader = _Reader(prechange, SESSION_FILES["prechange"])
    scheduled_reader = _Reader(scheduled, SESSION_FILES["scheduled"]) if scheduled is not None else None

    def evidence() -> list[str]:
        extra = scheduled_reader.evidence if scheduled_reader is not None else []
        return plain_reader.evidence + prior_reader.evidence + extra

    plain_ok, plain_index = _plain_capture_ok(plain_reader)
    prior_ok, prior_index = _plain_capture_ok(prior_reader)
    if not plain_ok or not prior_ok:
        return _item("not_proven", ["capture"], evidence())

    reasons = []
    plain_config_sha = plain_reader.get("/config_sha256")
    if plain_config_sha != prior_reader.get("/config_sha256"):
        reasons.append("config_differs")
    plain_tree = plain_reader.get("/source/package_tree_sha256")
    prior_tree = prior_reader.get("/source/package_tree_sha256")
    declared = plain_reader.get("/config/prechange_source")
    confirmed = bool(declared.get("package_tree_sha256")) and prior_tree == declared.get("package_tree_sha256")
    if not confirmed and declared.get("git_head"):
        head_matches = prior_reader.get("/source/git_head") == declared.get("git_head")
        confirmed = head_matches and prior_reader.get("/source/git_status_clean") is True
    if not confirmed or not plain_tree or plain_tree == prior_tree:
        reasons.append("prechange_identity_unconfirmed")
    if scheduled_reader is not None:
        if scheduled_reader.get("/config_sha256") != plain_config_sha:
            reasons.append("config_differs")
        if scheduled_reader.get("/source/package_tree_sha256") != plain_tree:
            reasons.append("prechange_identity_unconfirmed")
    if reasons:
        return _item("not_proven", sorted(set(reasons)), evidence())

    identical, usable = True, True
    for sequence in range(plain_reader.get("/config/expected_sequences")):
        for step in range(plain_reader.get("/config/expected_total_steps")):
            suffix = f"/sequences/{sequence}/steps/{step}/digests"
            plain_digests = plain_reader.get(f"/requests/{plain_index}{suffix}")
            prior_digests = prior_reader.get(f"/requests/{prior_index}{suffix}")
            usable = usable and _usable_digests(plain_digests) and _usable_digests(prior_digests)
            identical = identical and plain_digests == prior_digests
        plain_final = plain_reader.get(f"/requests/{plain_index}/sequences/{sequence}/final")
        identical = identical and plain_final == prior_reader.get(f"/requests/{prior_index}/sequences/{sequence}/final")
    plain_raw = plain_reader.get(f"/requests/{plain_index}/output/raw")
    prior_raw = prior_reader.get(f"/requests/{prior_index}/output/raw")
    identical = identical and bool(plain_raw.get("sha256")) and plain_raw.get("sha256") == prior_raw.get("sha256")
    if not usable:
        return _item("not_proven", ["digests_not_captured"], evidence())
    if identical:
        return _item("pass", (), evidence(), {"bit_identical": True})

    tolerance = plain_reader.get("/config/no_schedule_tolerance", note=False)
    comparison = plain_reader.get("/comparison_to_prior")
    measured = comparison if isinstance(comparison, Mapping) else {}
    values = {
        "bit_identical": False,
        "max_abs_diff_uint8": measured.get("max_abs_diff_uint8"),
        "mean_abs_diff_uint8": measured.get("mean_abs_diff_uint8"),
        "differing_frames": measured.get("differing_frames"),
        "tolerance": tolerance,
    }
    if tolerance is None:
        return _item("fail", ["output_changed"], evidence(), values)
    maximum = measured.get("max_abs_diff_uint8")
    if measured.get("error") is not None or not _is_int(maximum) or maximum > tolerance["max_abs_diff_uint8"]:
        return _item("fail", ["output_changed"], evidence(), values)
    # The difference was measured against the file given to --prior. It counts only if that
    # file is the pre-change record given to this verdict.
    prior_frames = prior_reader.get(f"/requests/{prior_index}/output/frames_sha256")
    same_session = measured.get("prior_session_id") == prior_reader.get("/session_id")
    if not same_session or not prior_frames or measured.get("prior_frames_sha256") != prior_frames:
        return _item("not_proven", ["prior_record_differs"], evidence(), values)
    return _item("pass", ["within_stated_tolerance"], evidence(), values)


def _fallback_report(reader: _Reader, context: _Scheduled, *, note: bool) -> dict[str, Any]:
    """Sum the approximate calls and the fallbacks of the candidate over the steps that select a profile."""
    totals = dict.fromkeys(FALLBACK_REASONS, 0)
    approximate = {"approx_quantized": 0, "approx_sparse": 0, "approx_calls": 0}
    target, all_fallback = 0, []
    for sequence in range(context.sequences):
        for step, profile in enumerate(context.expected):
            if profile is None:
                continue
            pointer = context.step_pointer("candidate", sequence, step) + "/attention/outcome"
            outcome = reader.get(pointer, note=note)
            for name in approximate:
                approximate[name] += outcome.get(name, 0)
            for name in totals:
                totals[name] += (outcome.get("fallback") or {}).get(name, 0)
            kind = context.approx_profiles[profile]["kind"]
            hits = outcome.get(f"approx_{kind}", 0)
            target += hits
            if hits == 0:
                all_fallback.append(step)
    return {
        "target_calls": target,
        "approximate": approximate,
        "fallback_totals": totals,
        "steps_all_fallback": sorted(set(all_fallback)),
    }


def _rule_kernel_target(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when the probed kernel is FlashInfer's and some layer call on a profile step is of the profile's kind."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    selected = {name for name in context.expected if name is not None}
    if any(context.approx_profiles[name]["backend"] != KERNEL_PROBE_BACKEND for name in selected):
        return _item("not_proven", ["no_probe_for_backend"], reader.evidence)
    kernel = reader.get("/probes/kernel")
    if not isinstance(kernel, Mapping) or not str(kernel.get("module") or "").startswith("flashinfer"):
        return _item("not_proven", ["kernel_not_real"], reader.evidence)
    probe_errors = 0
    for sequence in range(context.sequences):
        for step in range(context.total_steps):
            pointer = context.step_pointer("candidate", sequence, step) + "/attention/probe_errors"
            probe_errors += reader.get(pointer)
    if probe_errors:
        return _item("not_proven", ["probe_errors"], reader.evidence, {"probe_errors": probe_errors})
    values = _fallback_report(reader, context, note=True)
    if values["target_calls"] == 0:
        return _item("not_proven", ["all_fallback"], reader.evidence, values)
    return _item("pass", (), reader.evidence, values)


def _valid_compile_counts(counts: Any) -> bool:
    """Require completed, non-negative integer attempt accounting; bool is not a count."""
    fields = ("compile_attempts", "graphs_compiled", "compile_restarts", "compile_errors")
    if not isinstance(counts, Mapping) or any(type(counts.get(key)) is not int or counts[key] < 0 for key in fields):
        return False
    return counts["compile_attempts"] == sum(counts[key] for key in fields[1:])


def _request_compile_activity(reader: _Reader, index: int) -> tuple[str | None, bool]:
    """Check complete request-boundary counters, including attempts that returned no graph."""
    if reader.get("/probes/compile/counter_schema") != COMPILE_COUNTER_SCHEMA:
        return "compile_counter_schema", False
    before = reader.get(f"/requests/{index}/compile_before")
    after = reader.get(f"/requests/{index}/compile_after")
    if not _valid_compile_counts(before) or not _valid_compile_counts(after):
        # An in-flight attempt at either boundary is not a steady measured phase.
        return "compile_counters_invalid", False
    fields = ("compile_attempts", "graphs_compiled", "compile_restarts", "compile_errors")
    if any(after[key] < before[key] for key in fields):
        return "compile_counters_invalid", False
    if before["graphs_compiled"] != reader.get(f"/requests/{index}/graphs_compiled_before") or after[
        "graphs_compiled"
    ] != reader.get(f"/requests/{index}/graphs_compiled_after"):
        return "compile_counters_invalid", False
    return None, any(after[key] != before[key] for key in fields)


def _rule_compiled_graphs(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when the blocks were compiled without error and no block call of reference_a or candidate ran eagerly."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    fail, unproven, values = [], [], {}
    # The setting the service read back is optional: the counts below are the proof. Only a
    # read-back that says eager is used, and only then is it listed as evidence.
    effective = reader.get("/startup/effective", note=False)
    if isinstance(effective, Mapping) and effective.get("enforce_eager") is True:
        reader.get("/startup/effective/enforce_eager")
        fail.append("enforce_eager")
    setup_calls = reader.get("/probes/compile/setup_calls")
    compiled_models = {str(call.get("index")) for call in setup_calls if call.get("compiled_blocks", 0) >= 1}
    if not setup_calls or len(compiled_models) != len(setup_calls) or reader.get("/probes/compile/graphs_compiled") < 1:
        fail.append("not_compiled")
    if reader.get("/probes/compile/compile_errors") != 0:
        fail.append("compile_errors")
    if any(call.get("error") is not None for call in setup_calls):
        fail.append("compile_setup_errors")
    if reader.get("/probes/compile/counter_schema") != COMPILE_COUNTER_SCHEMA:
        unproven.append("compile_counter_schema")
    if not _valid_compile_counts(reader.get("/probes/compile")):
        unproven.append("compile_counters_invalid")
    eager_steps = []
    for request in ("reference_a", "candidate"):
        for sequence in range(context.sequences):
            for step in range(context.total_steps):
                compile_counts = reader.get(context.step_pointer(request, sequence, step) + "/compile")
                block_calls = compile_counts.get("block_calls", 0)
                if block_calls < 1:
                    unproven.append("no_block_calls_captured")
                eager_calls = compile_counts.get("eager_block_calls", 0)
                if eager_calls != 0 or compile_counts.get("graph_execs", 0) < block_calls:
                    fail.append("eager_block_calls")
                    eager_steps.append({"request": request, "sequence": sequence, "step": step})
                for index, counts in (compile_counts.get("by_model") or {}).items():
                    # The capture store cannot write this state. The check rejects a record
                    # that is not consistent.
                    if counts.get("block_calls", 0) > 0 and index not in compiled_models:
                        fail.append("uncompiled_model_used")
    if eager_steps:
        values["steps"] = eager_steps
    return _decide(sorted(set(fail)), sorted(set(unproven)), reader.evidence, values)


def _rule_shifted_boundary(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when boundary_shift selects as configured, makes no compile attempt and runs compiled graphs only."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    shifted_schedule = reader.get("/config/boundary_shift_schedule", note=False)
    names = [request.get("name") for request in reader.get("/requests", note=False)]
    if shifted_schedule is None or "boundary_shift" not in names:
        return _item("not_measured", ["boundary_shift_not_run"])
    reader.get("/config/boundary_shift_schedule")
    index = reader.request_index("boundary_shift")
    candidate = context.indices["candidate"]
    fail, unproven = [], []
    row = (reader.get("/config/model_family", note=False), reader.get("/config/execution_mode", note=False))
    # The capture rule requires boundary_shift after candidate, so ``index < candidate`` is
    # false when evaluate_combination applies this rule.
    if reader.get(f"/requests/{index}/error") is not None or index < candidate:
        unproven.append("capture")
    elif _sequence_problems(reader, index, str(SEQUENCE_SOURCES.get(row)), context.total_steps, context.sequences):
        unproven.append("capture")
    if unproven:
        return _item("not_proven", unproven, reader.evidence)
    if reader.get(f"/requests/{index}/sampling_params") != reader.get(f"/requests/{candidate}/sampling_params"):
        fail.append("sampling_params_differ")
    shifted = expected_profiles(shifted_schedule, context.total_steps)
    if first_switch(shifted) == context.switch:
        fail.append("boundary_not_shifted")
    shifted_context = copy.copy(context)
    shifted_context.indices = {**context.indices, "boundary_shift": index}
    mismatches = _selection_mismatches(reader, shifted_context, "boundary_shift", shifted)
    values: dict[str, Any] = {"first_switch": first_switch(shifted), "candidate_first_switch": context.switch}
    if mismatches:
        fail.append("selection_mismatch")
        values["steps"] = mismatches
    compiled_before = reader.get(f"/requests/{index}/graphs_compiled_before")
    delta = reader.get(f"/requests/{index}/graphs_compiled_after") - compiled_before
    values["delta"] = delta
    if delta != 0:
        fail.append("graphs_grew")
    problem, activity = _request_compile_activity(reader, index)
    if problem is not None:
        unproven.append(problem)
    elif activity and delta == 0:
        # A restart or failed attempt still invalidates the no-recompile claim.
        fail.append("compile_attempts_during_boundary_shift")
    # No new graph is evidence only if the shifted request ran compiled code.
    for sequence in range(context.sequences):
        for step in range(context.total_steps):
            compile_counts = reader.get(shifted_context.step_pointer("boundary_shift", sequence, step) + "/compile")
            block_calls = compile_counts.get("block_calls", 0)
            if block_calls < 1:
                unproven.append("no_block_calls_captured")
            if compile_counts.get("eager_block_calls", 0) != 0 or compile_counts.get("graph_execs", 0) < block_calls:
                fail.append("eager_block_calls")
    return _decide(sorted(set(fail)), sorted(set(unproven)), reader.evidence, values)


def _median(samples: Sequence[float]) -> float | None:
    return statistics.median(samples) if samples else None


def _step_seconds(reader: _Reader, context: _Scheduled, request: str, *, note: bool) -> list[float]:
    samples = []
    for sequence in range(context.sequences):
        for step in range(context.total_steps):
            samples.append(reader.get(context.step_pointer(request, sequence, step) + "/seconds", note=note))
    return samples


def _rule_timing(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when the timing samples are complete and comparable; it reports them and applies no speed threshold."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    unproven = []
    cold_start = reader.get("/startup/cold_start_seconds")
    if not _is_number(cold_start):
        unproven.append("cold_start_missing")
    warmup = context.indices["warmup"]
    warmup_sequences = reader.get(f"/requests/{warmup}/sequences")
    warmup_samples = [step.get("seconds") for sequence in warmup_sequences for step in sequence.get("steps", [])]
    if reader.get(f"/requests/{warmup}/error") is not None or not warmup_samples:
        unproven.append("warmup_samples_missing")
    elif not all(_is_number(sample) for sample in warmup_samples):
        unproven.append("warmup_samples_missing")
    # Without warm-up samples the reason above already says what is missing.
    warmed_up = _selection_pairs(warmup_sequences) if warmup_samples else None
    samples: dict[str, list[float]] = {}
    for request in ("reference_a", "reference_b", "candidate"):
        index = context.indices[request]
        samples[request] = _step_seconds(reader, context, request, note=True)
        if not all(_is_number(sample) and sample > 0 for sample in samples[request]):
            unproven.append(f"step_samples_missing:{request}")
        for sequence in range(context.sequences):
            for step in range(context.total_steps):
                if reader.get(context.step_pointer(request, sequence, step) + "/batch_size") != 1:
                    unproven.append("batch_size")
        before = reader.get(f"/requests/{index}/graphs_compiled_before")
        problem, activity = _request_compile_activity(reader, index)
        if problem is not None:
            unproven.append(f"{problem}:{request}")
        if activity or reader.get(f"/requests/{index}/graphs_compiled_after") != before:
            unproven.append(f"compiled_during_measured_request:{request}")
        # A selection that a transformer runs for the first time in a measured request can be
        # traced again there, and the graph count does not show a trace that compiles no graph.
        measured = _selection_pairs(reader.get(f"/requests/{index}/sequences", note=False))
        if warmed_up is not None and measured - warmed_up:
            unproven.append("warmup_coverage")
    if unproven:
        return _item("not_proven", sorted(set(unproven)), reader.evidence)
    names = context.expected * context.sequences
    dense = [sample for sample, name in zip(samples["candidate"], names) if name is None]
    approximate = [sample for sample, name in zip(samples["candidate"], names) if name]
    values = {
        "cold_start_seconds": cold_start,
        "compile_warmup": warmup_samples,
        "reference": {
            "samples": [samples["reference_a"], samples["reference_b"]],
            "median": _median(samples["reference_a"] + samples["reference_b"]),
        },
        "candidate": {
            "samples": samples["candidate"],
            "median": _median(samples["candidate"]),
            "dense_median": _median(dense),
            "approx_median": _median(approximate),
        },
        "batch_size": 1,
        "probes_active": True,
    }
    return _item("pass", (), reader.evidence, values)


def _rule_lpips(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when an LPIPS score of candidate against reference_a is stored with its metadata; no threshold applies."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    if reader.get("/comparison", note=False) is None:
        return _item("not_measured", ["comparison_absent"])
    lpips_block = reader.get("/comparison/lpips")
    error = lpips_block.get("error")
    if isinstance(error, str) and error.startswith(("lpips_unavailable", "lpips_failed")):
        # The score could not be computed. That says nothing about the outputs.
        return _item("not_measured", [error.partition(":")[0]], reader.evidence)
    reference = reader.get(f"/requests/{context.indices['reference_a']}/output")
    candidate = reader.get(f"/requests/{context.indices['candidate']}/output")
    # The conditions rule requires two outputs of the same shape, so the next two checks are
    # false when evaluate_combination applies this rule.
    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        return _item("fail", ["no_output"], reader.evidence)
    fail = check_frame_pair(reference, candidate, reader.get("/config/expected_frames", note=False))
    if reference.get("shape") != candidate.get("shape") and "shape_mismatch" not in fail:
        fail.append("shape_mismatch")
    if error is not None:
        fail.append(str(error))
    frames = (reference.get("shape") or [None])[0]
    if lpips_block.get("frames") != frames:
        fail.append("misaligned")
    if fail:
        return _item("fail", list(dict.fromkeys(fail)), reader.evidence)
    described = ("net", "package_version", "weights_sha256", "frame_alignment", "input_scaling", "aggregation")
    if any(lpips_block.get(key) is None for key in described):
        return _item("not_proven", ["lpips_metadata_incomplete"], reader.evidence)
    value, per_frame = lpips_block.get("value"), lpips_block.get("per_frame")
    if not _is_number(value) or not isinstance(per_frame, list) or len(per_frame) != frames:
        return _item("fail", ["lpips_value_missing"], reader.evidence)
    values = {key: lpips_block.get(key) for key in ("value", "frames", *described)}
    return _item("pass", (), reader.evidence, values)


def _artifact_problem(directory: Path | None, file_name: Any, sha256: Any) -> str | None:
    if directory is None or not isinstance(file_name, str) or not file_name or not isinstance(sha256, str):
        return "artifact_missing"
    path = directory / file_name
    try:
        if not path.is_file():
            return "artifact_missing"
        return None if sha256_file(path) == sha256 else "artifact_changed"
    except OSError:
        return "artifact_missing"


def _rule_video(scheduled: Mapping[str, Any], directory: Path | None) -> dict[str, Any]:
    """Pass when the side-by-side video and both request videos exist and match their recorded sha256."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    if reader.get("/comparison", note=False) is None:
        return _item("not_proven", ["side_by_side_missing"])
    if reader.get("/comparison/side_by_side", note=False) is None:
        reader.get("/comparison/side_by_side")
        return _item("not_proven", ["side_by_side_missing"], reader.evidence)
    fail = []
    labels = reader.get("/comparison/side_by_side/labels")
    labelled = isinstance(labels, list) and all(isinstance(label, str) and label for label in labels)
    if not labelled or len(labels) != 2:
        fail.append("labels")
    reference, candidate = context.indices["reference_a"], context.indices["candidate"]
    if reader.get("/comparison/side_by_side/frames") != reader.get(f"/requests/{reference}/output/shape")[0]:
        fail.append("frame_count")
    problems = [
        _artifact_problem(
            directory, reader.get("/comparison/side_by_side/file"), reader.get("/comparison/side_by_side/sha256")
        )
    ]
    for index in (reference, candidate):
        video_file = reader.get(f"/requests/{index}/output/video_file")
        problems.append(_artifact_problem(directory, video_file, reader.get(f"/requests/{index}/output/video_sha256")))
    if "artifact_changed" in problems:
        fail.append("artifact_changed")
    unproven = ["artifact_missing"] if "artifact_missing" in problems else []
    return _decide(fail, unproven, reader.evidence)


def _rule_backend_report(scheduled: Mapping[str, Any]) -> dict[str, Any]:
    """Pass when the attention probe was installed and reference_a and candidate hold an attention record per step."""
    reader = _Reader(scheduled, SESSION_FILES["scheduled"])
    context = _Scheduled(reader)
    installed = reader.get("/probes/installed")
    complete = any(isinstance(entry, Mapping) and entry.get("kind") == "attention" for entry in installed)
    per_step = []
    for request in ("reference_a", "candidate"):
        for sequence in range(context.sequences):
            for step in range(context.total_steps):
                attention = reader.get(context.step_pointer(request, sequence, step) + "/attention")
                keys = ("selection", "backends", "outcome", "kernel")
                if not isinstance(attention, Mapping) or any(key not in attention for key in keys):
                    complete = False
                    continue
                if request == "candidate":
                    step_record = reader.get(context.step_pointer(request, sequence, step), note=False)
                    per_step.append(
                        {
                            "sequence": sequence,
                            "step": step,
                            "expected": context.expected[step],
                            "transformer": step_record.get("transformer"),
                            "selection": attention["selection"],
                            "backends": attention["backends"],
                            "outcome": attention["outcome"],
                            "kernel": attention["kernel"],
                        }
                    )
    if not complete:
        return _item("not_proven", ["backend_report_incomplete"], reader.evidence)
    fallbacks = _fallback_report(reader, context, note=False)
    values = {
        "per_step": per_step,
        "fallback_totals": fallbacks["fallback_totals"],
        "steps_all_fallback": fallbacks["steps_all_fallback"],
    }
    return _item("pass", (), reader.evidence, values)


def _guard(rule: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
    """Apply one rule; a record that lacks data or is malformed gives not_proven, never an exception."""
    try:
        return rule(*args)
    except _MissingPointerError as exc:
        return _item("not_proven", [f"missing:{exc.reference}"])
    except (AttributeError, ConfigError, IndexError, KeyError, TypeError, ValueError) as exc:
        return _item("not_proven", [f"malformed:{type(exc).__name__}"])


def _step_digests(request: Mapping[str, Any]) -> list[list[Any]]:
    return [[step["digests"] for step in sequence["steps"]] for sequence in request["sequences"]]


def _reference_vs_plain(scheduled: Mapping[str, Any] | None, plain: Mapping[str, Any] | None) -> dict[str, Any]:
    """Report whether the [] reference of the scheduled service equals the service without a schedule."""
    try:
        reference = _find_request(scheduled or {}, "reference_a")
        request = _find_request(plain or {}, "plain")
        if reference is None or request is None:
            return {"bit_identical": None}
        same_steps = _step_digests(reference) == _step_digests(request)
        same_output = reference["output"]["raw"]["sha256"] == request["output"]["raw"]["sha256"]
        return {"bit_identical": bool(same_steps and same_output)}
    except (AttributeError, KeyError, TypeError):
        return {"bit_identical": None}


def _plain_seconds(plain: Mapping[str, Any] | None) -> dict[str, Any] | None:
    try:
        request = _find_request(plain or {}, "plain")
        if request is None:
            return None
        samples = [step["seconds"] for sequence in request["sequences"] for step in sequence["steps"]]
        return {"samples": samples, "median": _median(samples)}
    except (AttributeError, KeyError, TypeError):
        return None


def evaluate_combination(
    sessions: Mapping[str, Mapping[str, Any] | None], *, directory: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Apply the verdict rules to the session records of one combination.

    The rules are recomputed from the records in dependency order; a stored
    verdict is never read. Missing data gives not_proven.
    """
    scheduled, plain, prechange = (sessions.get(name) for name in SESSIONS)
    folder = Path(directory) if directory is not None else None
    combination = None
    for record in (scheduled, plain, prechange):
        if isinstance(record, Mapping) and isinstance(record.get("config"), Mapping):
            combination = record["config"].get("combination")
            break
    rules: dict[str, Callable[[], dict[str, Any]]] = {
        "capture": lambda: _guard(_rule_capture, scheduled),
        "conditions": lambda: _guard(_rule_conditions, scheduled),
        "dense_repeatable": lambda: _guard(_rule_dense_repeatable, scheduled),
        "profile_switch": lambda: _guard(_rule_profile_switch, scheduled),
        "dense_prefix": lambda: _guard(_rule_dense_prefix, scheduled),
        "kernel_target": lambda: _guard(_rule_kernel_target, scheduled),
        "compiled_graphs": lambda: _guard(_rule_compiled_graphs, scheduled),
        "shifted_boundary": lambda: _guard(_rule_shifted_boundary, scheduled),
        "timing": lambda: _guard(_rule_timing, scheduled),
        "lpips": lambda: _guard(_rule_lpips, scheduled),
        "video": lambda: _guard(_rule_video, scheduled, folder),
        "backend_report": lambda: _guard(_rule_backend_report, scheduled),
    }
    items: dict[str, dict[str, Any]] = {}
    for name, dependencies in ITEM_DEPENDENCIES.items():
        if scheduled is None:
            items[name] = _item("not_measured", ["scheduled_session_absent"])
            continue
        blocked = [dependency for dependency in dependencies if items[dependency]["status"] != "pass"]
        if blocked:
            items[name] = _item("not_proven", [f"depends:{dependency}" for dependency in blocked])
        else:
            items[name] = rules[name]()
    items["no_schedule"] = _guard(_rule_no_schedule, scheduled, plain, prechange)

    if scheduled is None:
        status = "not_measured"
    elif all(items[name]["status"] == "pass" for name in REQUIRED_ITEMS):
        status = "success"
    else:
        status = "not_success"
    timing = dict(items["timing"]["values"])
    if timing:
        timing["plain"] = _plain_seconds(plain)
    comparison = scheduled.get("comparison") if isinstance(scheduled, Mapping) else None
    report = {
        "seconds_per_step": timing,
        "lpips": dict(items["lpips"]["values"]),
        "side_by_side": (comparison or {}).get("side_by_side") if isinstance(comparison, Mapping) else None,
        "backend": dict(items["backend_report"]["values"]),
        "target": dict(items["kernel_target"]["values"]),
        "no_schedule": dict(items["no_schedule"]["values"]),
        "reference_vs_plain": _reference_vs_plain(scheduled, plain),
    }
    return {"combination": combination, "status": status, "items": items, "report": report}


def evaluate_matrix(combinations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """All six combinations must be proven; the shifted-boundary check must pass on one and fail on none."""
    statuses = {name: (combinations.get(name) or {}).get("status", "not_measured") for name in COMBINATIONS}
    shifted = {
        name: ((combinations.get(name) or {}).get("items") or {})
        .get("shifted_boundary", {})
        .get("status", "not_measured")
        for name in COMBINATIONS
    }
    passed = [name for name, status in shifted.items() if status == "pass"]
    failed = [name for name, status in shifted.items() if status == "fail"]
    if failed:
        shifted_status = "fail"
    elif passed:
        shifted_status = "pass"
    elif "not_proven" in shifted.values():
        shifted_status = "not_proven"
    else:
        shifted_status = "not_measured"
    proven = all(status == "success" for status in statuses.values()) and shifted_status == "pass"
    return {
        "status": "success" if proven else "not_success",
        "missing": [name for name, status in statuses.items() if status == "not_measured"],
        "shifted_boundary": {"status": shifted_status, "combinations": passed, "failed": failed},
    }


def render_summary(combinations: Mapping[str, Mapping[str, Any]], matrix: Mapping[str, Any]) -> str:
    """Return the text summary. Its first line is the only place that states the overall result."""
    lines = ["Acceptance: SUCCESS" if matrix.get("status") == "success" else "Acceptance: NOT PROVEN"]
    shifted = matrix.get("shifted_boundary") or {}
    lines.append(
        f"Shifted boundary: {shifted.get('status')}; "
        f"passed on {', '.join(shifted.get('combinations') or []) or 'none'}; "
        f"failed on {', '.join(shifted.get('failed') or []) or 'none'}"
    )
    for name in COMBINATIONS:
        result = combinations.get(name)
        if result is None:
            lines.append(f"{name}: not_measured (no session records)")
            continue
        lines.append(f"{name}: {result.get('status')}")
        items = result.get("items") or {}
        for item_name, item in items.items():
            if item.get("status") == "pass":
                continue
            # The first evidence pointer is not always the one that decided, so none is printed.
            lines.append(f"  {item_name}: {item.get('status')} [{', '.join(item.get('reasons') or [])}]")
        if result.get("status") != "success":
            continue
        report = result.get("report") or {}
        timing = report.get("seconds_per_step") or {}
        reference, candidate = timing.get("reference") or {}, timing.get("candidate") or {}
        lines.append(
            f"  cold start {timing.get('cold_start_seconds')} s; s/step median: reference {reference.get('median')}, "
            f"candidate {candidate.get('median')} (dense steps {candidate.get('dense_median')}, "
            f"approximate steps {candidate.get('approx_median')}); probes active in both runs"
        )
        lpips_report = report.get("lpips") or {}
        lines.append(
            f"  LPIPS {lpips_report.get('value')} ({lpips_report.get('net')}, {lpips_report.get('frames')} frames)"
        )
        target = report.get("target") or {}
        lines.append(
            f"  approximate calls {target.get('approximate')}; fallbacks {target.get('fallback_totals')}; "
            f"steps with no approximate call {target.get('steps_all_fallback')}"
        )
        no_schedule = report.get("no_schedule") or {}
        lines.append(
            f"  no-schedule output bit-identical to pre-change: {no_schedule.get('bit_identical')}; "
            f"[] reference bit-identical to the no-schedule service: "
            f"{(report.get('reference_vs_plain') or {}).get('bit_identical')}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_sessions(directory: Path) -> tuple[dict[str, dict[str, Any] | None], list[dict[str, str]]]:
    sessions: dict[str, dict[str, Any] | None] = {}
    inputs = []
    for name, file_name in SESSION_FILES.items():
        path = directory / file_name
        if not path.is_file():
            sessions[name] = None
            continue
        try:
            sessions[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read {path}: {exc}") from exc
        inputs.append({"file": str(path), "sha256": sha256_file(path)})
    return sessions, inputs


def _run_verdict(out_file: str, directories: Sequence[str]) -> int:
    out_path = Path(out_file).resolve()
    if _inside_repository(out_path):
        raise ConfigError(f"the verdict file {out_path} is inside the repository")
    combinations: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, str]] = []
    for directory in directories:
        folder = Path(directory)
        if not folder.is_dir():
            raise ConfigError(f"{folder} is not a directory")
        sessions, session_inputs = _load_sessions(folder)
        if not session_inputs:
            raise ConfigError(f"{folder} holds no session record")
        result = evaluate_combination(sessions, directory=folder)
        name = result["combination"]
        if not isinstance(name, str) or name not in COMBINATIONS:
            raise ConfigError(f"{folder}: the records name no known combination")
        if name in combinations:
            raise ConfigError(f"combination {name} was given twice")
        combinations[name] = result
        inputs.extend(session_inputs)
    matrix = evaluate_matrix(combinations)
    summary = render_summary(combinations, matrix)
    verdict = {
        "schema": SCHEMA,
        "kind": "verdict",
        "inputs": inputs,
        "combinations": combinations,
        "matrix": matrix,
        "summary_text": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, verdict)
    print(summary, end="")
    return 0 if matrix["status"] == "success" else 1


def _run_preflight(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    out_path = Path(args.out).resolve() if args.out else None
    if out_path is not None and _inside_repository(out_path):
        raise ConfigError(f"the preflight file {out_path} is inside the repository")
    result = preflight(config, session=args.session, device=args.device)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_path, result)
    print(json.dumps(_json_safe(result), indent=1, sort_keys=True))
    return 0 if result["ok"] else 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acceptance harness for step-index attention schedules. Verdict items: see the module docstring."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight", help="check probe targets and compile probes without weights")
    preflight_parser.add_argument("--config", required=True)
    preflight_parser.add_argument("--session", choices=SESSIONS, default="scheduled")
    preflight_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    preflight_parser.add_argument("--out")
    run_parser = commands.add_parser("run", help="run one session and write its record")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--session", choices=SESSIONS, required=True)
    run_parser.add_argument("--out", required=True, help="evidence directory outside the repository")
    run_parser.add_argument("--prior", help="pre-change session record; only with --session plain")
    verdict_parser = commands.add_parser(
        "verdict", help="evaluate the verdict items from session records and print the summary; standard library only"
    )
    verdict_parser.add_argument("--out", required=True, help="verdict file outside the repository")
    verdict_parser.add_argument("directories", nargs="+", help="one evidence directory per combination")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _build_parser().parse_args(arguments)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2
    try:
        if args.command == "preflight":
            return _run_preflight(args)
        if args.command == "run":
            if args.prior is not None and args.session != "plain":
                raise ConfigError("--prior is accepted only with --session plain")
            config = load_config(args.config)
            return run_session(config, session=args.session, out_dir=args.out, prior=args.prior, argv=arguments)
        return _run_verdict(args.out, args.directories)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EvidenceError as exc:
        # run_session and the preflight checks catch their own evidence errors. This covers
        # one raised outside them.
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
