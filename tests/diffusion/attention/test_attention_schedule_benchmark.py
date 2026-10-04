# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for benchmarks/diffusion/bench_attention_schedule.py, the acceptance harness for attention schedules.

The harness decides from session records whether a run proves a step-index attention schedule. Most
tests here build session records by hand and check the decision of each rule: complete evidence gives
success, and each defect (a shifted boundary, an all-dense candidate, an all-fallback candidate, an
eager run, a broken output, changed reference conditions) blocks it. Other tests cover the capture
store, the probes on small fake modules, and ``run_session`` against a fake service that sends the
capture store the events the probes would send.

No model weights are loaded and no GPU is used. These tests use production code: the schedule
selector, the names the probes patch, and the default ``torch.compile`` backend on CPU. The tests that
compile are skipped, with the reason, when that backend cannot run on the host. The attention probe and
the kernel probe are not run against a real attention layer here.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import re
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.schedule import parse_attention_schedule, select_attention_profile

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_MODULE_PATH = _REPO_ROOT / "benchmarks" / "diffusion" / "bench_attention_schedule.py"
_BENCH_MODULE_NAME = "benchmarks.diffusion.bench_attention_schedule"

if _BENCH_MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_BENCH_MODULE_NAME, _BENCH_MODULE_PATH)
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_BENCH_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)

bench = sys.modules[_BENCH_MODULE_NAME]

_PROFILE = "approx"
_LAYER_CALLS = 6
_FRAMES = 2
_TREE_AFTER = "tree-after"
_TREE_BEFORE = "tree-before"
# Positions of the requests in a scheduled session record.
_WU, _RA, _RB, _CA, _BS = range(5)
_DROP = object()


# ---------------------------------------------------------------------------
# Session records built by hand
# ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _digest(text: str) -> dict[str, Any]:
    """Return a digest object as the capture store writes it, derived from a short string."""
    return {"sha256": _sha(text), "shape": [1, 4], "dtype": "torch.float32", "nan": 0, "inf": 0}


def _from(start: int, profile: str = _PROFILE) -> list[dict[str, Any]]:
    """Return a schedule that selects one profile from a step to the end."""
    return [{"start": start, "end": None, "profile": profile}]


def make_config(combination: str = "C1", *, steps: int = 8, switch: int = 3, **overrides: Any) -> dict[str, Any]:
    """Return the body of a config file for one combination. Overrides replace top-level keys."""
    family, mode, kind = bench.COMBINATIONS[combination]
    config: dict[str, Any] = {
        "schema": bench.SCHEMA,
        "combination": combination,
        "model_family": family,
        "execution_mode": mode,
        "approximation": kind,
        "omni_kwargs": {"model": "org/model", "step_execution": mode == "step"},
        "schedule_config": {"profiles": {_PROFILE: {"backend": "TRTLLM_ATTN"}}, "default": []},
        "approx_profiles": {_PROFILE: {"kind": kind, "backend": "TRTLLM_ATTN"}},
        "candidate_schedule": _from(switch),
        # Dense on the first and the last step, so each Wan transformer runs the profile and a dense step.
        "warmup_schedule": [{"start": 1, "end": steps - 1, "profile": _PROFILE}],
        "boundary_shift_schedule": _from(switch + 1),
        "expected_total_steps": steps,
        "expected_frames": _FRAMES,
        "prompt": {"prompt": "a cat walks through a garden"},
        "sampling_params": {"seed": 7, "num_inference_steps": steps},
        "prechange_source": {"package_tree_sha256": _TREE_BEFORE},
        "fps": 8,
    }
    config.update(overrides)
    return config


def _attention(kind: str | None) -> dict[str, Any]:
    """Return the attention block of one step: dense when kind is None, else every layer call approximate."""
    outcome: dict[str, Any] = {
        "baseline": 0,
        "unscheduled": 0,
        "approx_quantized": 0,
        "approx_sparse": 0,
        "approx_calls": 0,
        "fallback": dict.fromkeys(bench.FALLBACK_REASONS, 0),
    }
    kernel: dict[str, Any] = {
        "calls": 0,
        "sage_calls": 0,
        "skip_calls": 0,
        "dense_calls": 0,
        "orphan_calls": 0,
        "skip_factor_min": None,
        "skip_factor_max": None,
    }
    if kind is None:
        outcome["baseline"] = _LAYER_CALLS
        selection, backend = "<baseline>", "SDPA"
    else:
        outcome[f"approx_{kind}"] = outcome["approx_calls"] = _LAYER_CALLS
        kernel["calls"] = _LAYER_CALLS
        if kind == "quantized":
            kernel["sage_calls"] = _LAYER_CALLS
        else:
            kernel.update(skip_calls=_LAYER_CALLS, skip_factor_min=0.5, skip_factor_max=0.5)
        selection, backend = _PROFILE, "TRTLLM_ATTN"
    return {
        "layer_calls": _LAYER_CALLS,
        "selection": {selection: _LAYER_CALLS},
        "backends": {backend: _LAYER_CALLS},
        "outcome": outcome,
        "kernel": kernel,
        "step_mismatch_calls": 0,
        "probe_errors": 0,
    }


def _step_digests(family: str, step: int, diverge_from: int | None) -> dict[str, Any]:
    def tag(first_changed: int | None) -> str:
        return "dense" if first_changed is None or step < first_changed else "approx"

    if family == "wan2_2":
        # Wan records the latent that enters a step. It changes one step after the noise prediction does.
        latent_changes = None if diverge_from is None else diverge_from + 1
        return {
            "latent_in": _digest(f"latent-{tag(latent_changes)}-{step}"),
            "noise_pred": _digest(f"noise-{tag(diverge_from)}-{step}"),
        }
    return {
        "video": _digest(f"video-{tag(diverge_from)}-{step}"),
        "audio": _digest(f"audio-{tag(diverge_from)}-{step}"),
    }


def _make_output(directory: Path, session: str, name: str, tag: str) -> dict[str, Any]:
    frames_path = directory / f"{session}-{name}.frames.npy"
    video_path = directory / f"{session}-{name}.mp4"
    frames_path.write_bytes(f"frames-{tag}".encode())
    video_path.write_bytes(f"video-{tag}".encode())
    shape = [_FRAMES, 4, 4, 3]
    return {
        "raw": {"sha256": _sha(f"raw-{tag}"), "shape": shape, "dtype": "float32", "nan": 0, "inf": 0},
        "shape": shape,
        "dtype": "uint8",
        "source_dtype": "float32",
        "nan_count": 0,
        "inf_count": 0,
        "frames_file": frames_path.name,
        "frames_sha256": bench.sha256_file(frames_path),
        "video_file": video_path.name,
        "video_sha256": bench.sha256_file(video_path),
        "video_error": None,
    }


def make_request(
    config: dict[str, Any],
    session: str,
    name: str,
    directory: Path,
    *,
    schedule: list[dict[str, Any]] | None,
    select_from: int | None = None,
    select_to: int | None = None,
    diverge_from: int | None = None,
    compiled: tuple[int, int] = (5, 5),
) -> dict[str, Any]:
    """Return one request entry of a session record and write its artifact files.

    ``select_from`` is the first step whose attention block selects the profile and ``select_to`` is
    the first step after it that is dense again. ``diverge_from`` is the first step whose digests differ
    from an all-dense run. None means no such step.
    """
    family, mode, kind = config["model_family"], config["execution_mode"], config["approximation"]
    total = config["expected_total_steps"]
    steps = []
    for step in range(total):
        approximate = select_from is not None and select_from <= step and (select_to is None or step < select_to)
        transformer = None
        if family == "wan2_2":
            transformer = "transformer" if step < total // 2 else "transformer_2"
        steps.append(
            {
                "step": step,
                "batch_size": 1,
                "transformer": transformer,
                "seconds": 0.25 if approximate else 0.5,
                "digest_seconds": 0.01,
                "digests": _step_digests(family, step, diverge_from),
                "compile": {
                    "block_calls": 6,
                    "eager_block_calls": 0,
                    "graph_execs": 12,
                    "graphs_compiled": 0,
                    "by_model": {"0": {"block_calls": 6, "eager_block_calls": 0}},
                },
                "attention": _attention(kind if approximate else None) if session == "scheduled" else None,
            }
        )
    tag = "dense" if diverge_from is None else f"approx-{diverge_from}"
    final: dict[str, Any] | None
    if family == "wan2_2":
        initial = {"latents": _digest("initial-latents")}
        final = {"latents": _digest(f"final-latents-{tag}")}
    else:
        initial = {"video": _digest("initial-video"), "audio": _digest("initial-audio")}
        # Step mode ends a sequence at post_decode and records no final state.
        final = None if mode == "step" else {"video": _digest(f"final-video-{tag}"), "audio": _digest(f"audio-{tag}")}
    return {
        "name": name,
        "attention_schedule": copy.deepcopy(schedule),
        "prompt": copy.deepcopy(config["prompt"]),
        "sampling_params": copy.deepcopy(config["sampling_params"]),
        "error": None,
        "wall_seconds": 4.0,
        "graphs_compiled_before": compiled[0],
        "graphs_compiled_after": compiled[1],
        "sequences": [
            {
                "source": bench.SEQUENCE_SOURCES[(family, mode)],
                "total_steps": total,
                "incomplete": False,
                "initial": initial,
                "final": final,
                "steps": steps,
            }
        ],
        "unattributed": {
            "block_calls": 0,
            "eager_block_calls": 0,
            "graph_execs": 0,
            "attention_layer_calls": 0,
            "kernel_calls": 0,
        },
        "output": None if name == "warmup" else _make_output(directory, session, name, tag),
    }


def make_session(
    config: dict[str, Any],
    session: str,
    requests: list[dict[str, Any]],
    *,
    tree: str,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scheduled = session == "scheduled"
    installed = [
        {"target": "torch._TorchCompileInductorWrapper:__call__", "kind": "compile"},
        {"target": f"{bench.RUNNER_MODULE}:regionally_compile", "kind": "block"},
    ]
    kernel = None
    if scheduled:
        installed.append({"target": f"{bench.ATTENTION_MODULE}.Attention:_run_local_attention", "kind": "attention"})
        kernel = {
            "target": f"{bench.TRTLLM_MODULE}:{bench.TRTLLM_KERNEL}",
            "module": "flashinfer.prefill",
            "qualname": bench.TRTLLM_KERNEL,
        }
    setup_call = {
        "index": 0,
        "model_class": "ToyTransformer",
        "compiled_blocks": 2,
        "block_class_names": ["ToyBlock"],
        "kwargs": {"dynamic": "True"},
        "error": None,
    }
    return {
        "schema": bench.SCHEMA,
        "kind": "session",
        "session": session,
        "session_id": f"{session}-session",
        "status": "completed",
        "abort": None,
        "exit_code": 0,
        "argv": ["run", "--session", session],
        "started_utc": "2026-10-04T00:00:00Z",
        "finished_utc": "2026-10-04T00:10:00Z",
        "config": copy.deepcopy(config),
        "config_sha256": "config-sha256",
        "source": {
            "harness_sha256": "harness-sha256",
            "package_file": "/src/vllm_omni/__init__.py",
            "package_tree_sha256": tree,
            "git_head": "head-before",
            "git_diff_sha256": "diff-sha256",
            "git_status_clean": session == "prechange",
        },
        "environment": {"python": "3.12.0"},
        "startup": {
            "omni_kwargs": copy.deepcopy(config["omni_kwargs"]),
            "cold_start_seconds": 30.0,
            "effective": {"enforce_eager": False},
            "effective_missing_reason": None,
        },
        "probes": {
            "installed": installed,
            "compile": {
                "mechanism": bench.COMPILE_MECHANISM,
                "setup_calls": [setup_call],
                "graphs_compiled": 5,
                "compile_errors": 0,
            },
            "kernel": kernel,
            "outside_requests": {"sequences": 0, "steps": 0},
        },
        "requests": requests,
        "comparison": comparison,
        "comparison_to_prior": None,
    }


def build_sessions(directory: Path, combination: str, *, steps: int, switch: int) -> dict[str, Any]:
    """Return scheduled, plain and pre-change session records that satisfy every rule."""
    directory.mkdir(parents=True, exist_ok=True)
    config = bench.validate_config(make_config(combination, steps=steps, switch=switch))
    scheduled_requests = [
        make_request(
            config,
            "scheduled",
            "warmup",
            directory,
            schedule=config["warmup_schedule"],
            select_from=1,
            select_to=steps - 1,
            diverge_from=1,
            compiled=(0, 5),
        ),
        make_request(config, "scheduled", "reference_a", directory, schedule=[]),
        make_request(config, "scheduled", "reference_b", directory, schedule=[]),
        make_request(
            config,
            "scheduled",
            "candidate",
            directory,
            schedule=config["candidate_schedule"],
            select_from=switch,
            diverge_from=switch,
        ),
        make_request(
            config,
            "scheduled",
            "boundary_shift",
            directory,
            schedule=config["boundary_shift_schedule"],
            select_from=switch + 1,
            diverge_from=switch + 1,
        ),
    ]
    side_by_side = directory / "scheduled-side_by_side.mp4"
    side_by_side.write_bytes(b"side-by-side")
    comparison = {
        "lpips": {
            "value": 0.12,
            "error": None,
            "per_frame": [0.1, 0.14],
            "frames": _FRAMES,
            "net": "alex",
            "package_version": "0.1.4",
            "weights_sha256": "weights-sha256",
            "frame_alignment": "index",
            "resize": "none",
            "input_scaling": "uint8/127.5-1",
            "aggregation": "mean_over_frames",
        },
        "side_by_side": {
            "file": side_by_side.name,
            "sha256": bench.sha256_file(side_by_side),
            "labels": ["all-dense reference", f"{combination} scheduled"],
            "fps": 8.0,
            "frames": _FRAMES,
        },
        "side_by_side_error": None,
    }
    sessions = {
        "scheduled": make_session(config, "scheduled", scheduled_requests, tree=_TREE_AFTER, comparison=comparison)
    }
    for session, tree in (("plain", _TREE_AFTER), ("prechange", _TREE_BEFORE)):
        requests = [
            make_request(config, session, "warmup", directory, schedule=None, compiled=(0, 5)),
            make_request(config, session, "plain", directory, schedule=None),
        ]
        sessions[session] = make_session(config, session, requests, tree=tree)
    return sessions


@pytest.fixture
def make_sessions(tmp_path):
    """Build passing session records of one combination and their artifact files in its own directory."""

    def build(combination: str = "C1", *, steps: int = 8, switch: int = 3) -> tuple[dict[str, Any], Path]:
        directory = tmp_path / combination.lower()
        return build_sessions(directory, combination, steps=steps, switch=switch), directory

    return build


def _evaluate(sessions: dict[str, Any], directory: Path) -> dict[str, Any]:
    return bench.evaluate_combination(sessions, directory=directory)


def _not_passing(result: dict[str, Any]) -> dict[str, Any]:
    """Return the status and the reasons of every item that did not pass, for assertion messages."""
    return {
        name: (item["status"], item["reasons"]) for name, item in result["items"].items() if item["status"] != "pass"
    }


def _summary(result: dict[str, Any]) -> str:
    """Return the summary of a matrix in which the other five combinations are proven."""
    passing = {"status": "success", "items": {"AE5_boundary": {"status": "pass"}}, "report": {}}
    combinations: dict[str, Any] = dict.fromkeys(bench.COMBINATIONS, passing)
    combinations[result["combination"] or "C1"] = result
    return bench.render_summary(combinations, bench.evaluate_matrix(combinations))


def _put(document: Any, pointer: str, value: Any) -> None:
    """Set the node a JSON pointer names, or delete it when value is _DROP."""
    parent_pointer, _, key = pointer.rpartition("/")
    parent = bench.resolve_pointer(document, parent_pointer)
    index: Any = int(key) if isinstance(parent, list) else key
    if value is _DROP:
        del parent[index]
    else:
        parent[index] = value


def _steps(sessions: dict[str, Any], index: int, session: str = "scheduled") -> list[dict[str, Any]]:
    return sessions[session]["requests"][index]["sequences"][0]["steps"]


def _rebuild(sessions: dict[str, Any], directory: Path, index: int, **kwargs: Any) -> None:
    """Replace one request of the scheduled session by one with another selection or another output."""
    scheduled = sessions["scheduled"]
    old = scheduled["requests"][index]
    scheduled["requests"][index] = make_request(
        scheduled["config"], "scheduled", old["name"], directory, schedule=old["attention_schedule"], **kwargs
    )


def _make_fallback(step: dict[str, Any], reason: str) -> None:
    """Rewrite a step so that every layer call fell back for one reason and no approximate call ran."""
    outcome = step["attention"]["outcome"]
    outcome.update(approx_quantized=0, approx_sparse=0, approx_calls=0)
    outcome["fallback"][reason] = _LAYER_CALLS
    kernel_calls = 0 if reason == "layer_fallback" else _LAYER_CALLS
    step["attention"]["kernel"].update(
        calls=kernel_calls,
        sage_calls=0,
        skip_calls=0,
        dense_calls=kernel_calls,
        skip_factor_min=None,
        skip_factor_max=None,
    )


# ---------------------------------------------------------------------------
# Config and small helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schedule", "total"),
    [
        pytest.param([], 4, id="empty"),
        pytest.param(_from(2), 6, id="gap-at-start-open-end"),
        pytest.param([{"start": 0, "end": 3, "profile": "a"}], 5, id="gap-at-end"),
        pytest.param(
            [{"start": 1, "end": 3, "profile": "a"}, {"start": 4, "end": None, "profile": "b"}], 8, id="two-ranges"
        ),
        pytest.param(
            [{"start": 1, "end": 2, "profile": "a"}, {"start": 2, "end": 5, "profile": "b"}], 5, id="adjacent-ranges"
        ),
        pytest.param([{"start": 0, "end": None, "profile": "a"}], 1, id="one-step"),
    ],
)
def test_expected_profiles_matches_production_selector(schedule, total):
    parsed = parse_attention_schedule(schedule)
    production = [select_attention_profile(parsed, step, total_steps=total) for step in range(total)]

    assert bench.expected_profiles(schedule, total) == production


@pytest.mark.parametrize(
    ("schedule", "total"),
    [
        pytest.param([{"start": True, "end": None, "profile": "a"}], 4, id="bool-step"),
        pytest.param([{"start": 0, "end": 3, "profile": "a"}, {"start": 2, "end": 4, "profile": "b"}], 4, id="overlap"),
        pytest.param(
            [{"start": 0, "end": None, "profile": "a"}, {"start": 2, "end": 4, "profile": "b"}],
            4,
            id="range-after-open-end",
        ),
        pytest.param([{"start": 2, "end": 2, "profile": "a"}], 4, id="end-not-above-start"),
        pytest.param([{"start": 1, "end": 5, "profile": "a"}], 4, id="end-beyond-total"),
        pytest.param([{"start": 4, "end": None, "profile": "a"}], 4, id="start-beyond-total"),
        pytest.param([{"start": 0, "end": None, "profile": "a", "extra": 1}], 4, id="extra-key"),
        pytest.param([{"start": 0, "end": None, "profile": "a.b"}], 4, id="profile-name-with-a-dot"),
        pytest.param([{"start": 0, "end": None, "profile": "1a"}], 4, id="profile-name-starts-with-a-digit"),
        pytest.param([{"start": 0, "end": None, "profile": ""}], 4, id="empty-profile-name"),
        pytest.param([], 0, id="no-steps"),
    ],
)
def test_expected_profiles_rejects_what_production_rejects(schedule, total):
    with pytest.raises(bench.ConfigError):
        bench.expected_profiles(schedule, total)
    with pytest.raises((TypeError, ValueError)):
        select_attention_profile(parse_attention_schedule(schedule), 0, total_steps=total)


def test_first_switch():
    assert bench.first_switch([None, None, "a", None, "b"]) == 2
    assert bench.first_switch([None, None]) is None


@pytest.mark.parametrize("combination", list(bench.COMBINATIONS))
def test_validate_config_accepts_each_combination(combination):
    raw = make_config(combination)
    snapshot = copy.deepcopy(raw)

    config = bench.validate_config(raw, session="scheduled")

    assert raw == snapshot
    assert config["combination"] == combination
    assert config["expected_sequences"] == 1


def test_validate_config_fills_defaults():
    raw = make_config("C1")
    for key in ("boundary_shift_schedule", "expected_frames"):
        del raw[key]
    del raw["schedule_config"]["default"]

    config = bench.validate_config(raw)

    assert config["boundary_shift_schedule"] is None
    assert config["expected_frames"] is None
    assert config["expected_sequences"] == 1
    assert config["no_schedule_tolerance"] is None
    assert config["lpips"] == {"net": "alex"}
    assert config["adapter"] is None
    assert config["schedule_config"]["default"] == []
    assert config["prechange_source"] == {"package_tree_sha256": _TREE_BEFORE, "git_head": None}


@pytest.mark.parametrize(
    ("combination", "pointer", "value", "message"),
    [
        pytest.param("C1", "/omni_kwargs/enforce_eager", True, "omni_kwargs.enforce_eager", id="enforce-eager"),
        pytest.param(
            "C1",
            "/omni_kwargs/diffusion_attention_schedule",
            {},
            "omni_kwargs.diffusion_attention_schedule",
            id="schedule-in-omni-kwargs",
        ),
        pytest.param("C1", "/omni_kwargs/num_gpus", 2, "omni_kwargs.num_gpus", id="two-gpus"),
        pytest.param(
            "C1",
            "/omni_kwargs/distributed_executor_backend",
            "mp",
            "omni_kwargs.distributed_executor_backend",
            id="mp-executor",
        ),
        pytest.param(
            "C1",
            "/omni_kwargs/diffusion_compile_granularity",
            "full",
            "omni_kwargs.diffusion_compile_granularity",
            id="full-compile",
        ),
        pytest.param(
            "C3", "/omni_kwargs/step_execution", False, "omni_kwargs.step_execution must be True", id="step-mode-off"
        ),
        pytest.param(
            "C1", "/omni_kwargs/step_execution", True, "omni_kwargs.step_execution must be False", id="step-mode-on"
        ),
        pytest.param("C1", "/model_family", "wan2_2", "model_family must be 'minimax_h3'", id="family-row"),
        pytest.param("C1", "/execution_mode", "step", "execution_mode must be 'request'", id="mode-row"),
        pytest.param("C1", "/approximation", "sparse", "approximation must be 'quantized'", id="approximation-row"),
        pytest.param("C1", "/candidate_schedule", _from(0), "candidate_schedule: the first switch", id="switch-at-0"),
        pytest.param(
            "C1", "/candidate_schedule", _from(3, "other"), "candidate_schedule: profile(s) other", id="unknown-profile"
        ),
        pytest.param("C1", "/candidate_schedule", _from(8), "candidate_schedule: range 0", id="beyond-total"),
        pytest.param(
            "C1",
            f"/approx_profiles/{_PROFILE}/kind",
            "sparse",
            "candidate_schedule: the profile at the first switch must be quantized",
            id="kind-at-switch",
        ),
        pytest.param(
            "C1",
            "/approx_profiles",
            {"other": {"kind": "quantized", "backend": "TRTLLM_ATTN"}},
            "approx_profiles['other']",
            id="approx-profile-not-configured",
        ),
        pytest.param("C1", "/warmup_schedule", [], "warmup_schedule: profile(s) approx", id="warmup-omits-profile"),
        pytest.param("C1", "/warmup_schedule", _from(0), "warmup_schedule: at least one step", id="warmup-not-dense"),
        pytest.param(
            "C1", "/boundary_shift_schedule", _from(3), "boundary_shift_schedule: the first switch", id="same-switch"
        ),
        pytest.param("C1", "/sampling_params", {"num_inference_steps": 8}, "sampling_params.seed", id="no-seed"),
        pytest.param(
            "C1", "/sampling_params/attention_schedule", [], "sampling_params.attention_schedule", id="own-schedule"
        ),
        pytest.param("C1", "/sampling_params/generator", 1, "sampling_params.generator", id="generator"),
        pytest.param(
            "C1",
            "/sampling_params/extra_args",
            {"attention_schedule": []},
            "sampling_params.extra_args.attention_schedule",
            id="schedule-in-extra-args",
        ),
        pytest.param("C1", "/sampling_params/output_type", "pt", "sampling_params.output_type", id="tensor-output"),
        pytest.param(
            "C1",
            "/sampling_params/emit_request_lifecycle",
            True,
            "sampling_params.emit_request_lifecycle",
            id="lifecycle-output",
        ),
        pytest.param(
            "C1",
            "/sampling_params/extra_args",
            {"preencode_mp4": True},
            "sampling_params.extra_args.preencode_mp4",
            id="encoded-output",
        ),
        pytest.param(
            "C1",
            "/schedule_config/profiles",
            {"a.b": {"backend": "TRTLLM_ATTN"}},
            "schedule_config.profiles: the name 'a.b'",
            id="profile-name",
        ),
        pytest.param("C1", "/schedule_config/default", _from(1), "schedule_config.default", id="default-schedule"),
        pytest.param("C1", "/prechange_source", _DROP, "missing config key(s): prechange_source", id="no-prechange"),
        pytest.param("C1", "/prechange_source", {}, "prechange_source must state", id="empty-prechange"),
        pytest.param("C1", "/extra", 1, "unknown config key(s): extra", id="unknown-key"),
        pytest.param("C1", "/schema", "other/1", "schema must be", id="schema"),
        pytest.param("C1", "/expected_total_steps", 1, "expected_total_steps", id="one-step"),
        pytest.param("C1", "/expected_sequences", 0, "expected_sequences", id="no-sequence"),
        pytest.param("C1", "/fps", 0, "fps must be a positive number", id="fps"),
        pytest.param("C1", "/lpips", {"net": "resnet"}, "lpips.net", id="lpips-net"),
        pytest.param(
            "C1",
            "/no_schedule_tolerance",
            {"max_abs_diff_uint8": -1},
            "no_schedule_tolerance.max_abs_diff_uint8",
            id="negative-tolerance",
        ),
        pytest.param("C1", "/adapter", {"module": "x"}, "adapter must be an object", id="adapter-keys"),
    ],
)
def test_validate_config_rejects(combination, pointer, value, message):
    raw = make_config(combination)
    _put(raw, pointer, value)

    with pytest.raises(bench.ConfigError, match=re.escape(message)):
        bench.validate_config(raw)


def test_validate_config_rejects_unknown_session():
    with pytest.raises(bench.ConfigError, match="session must be one of"):
        bench.validate_config(make_config("C1"), session="other")


def test_validate_config_accepts_numpy_output():
    raw = make_config("C1")
    raw["sampling_params"].update(output_type="np", emit_request_lifecycle=False, extra_args={"preencode_mp4": False})

    assert bench.validate_config(raw)["sampling_params"]["output_type"] == "np"


def test_load_config_adds_the_file_hash(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**make_config("C5"), "_sha256": "stale"}), encoding="utf-8")

    config = bench.load_config(path)

    assert config["_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config["combination"] == "C5"

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(bench.ConfigError, match="cannot read config"):
        bench.load_config(path)
    with pytest.raises(bench.ConfigError, match="cannot read config"):
        bench.load_config(tmp_path / "absent.json")
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(bench.ConfigError, match="the config must be a JSON object"):
        bench.load_config(path)


def test_resolve_pointer():
    document = {"a": [{"b/c": 1, "d~e": 2}], "": 3}

    assert bench.resolve_pointer(document, "") is document
    assert bench.resolve_pointer(document, "/a/0/b~1c") == 1
    assert bench.resolve_pointer(document, "/a/0/d~0e") == 2
    assert bench.resolve_pointer(document, "/") == 3
    for pointer in ("a", "/a/1", "/a/x", "/a/-1", "/a/0/missing", "/a/0/b~1c/deeper"):
        with pytest.raises(KeyError):
            bench.resolve_pointer(document, pointer)


def test_tensor_digest():
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    digest = bench.tensor_digest(base)

    assert digest == bench.tensor_digest(base.clone())
    assert (digest["shape"], digest["dtype"], digest["nan"], digest["inf"]) == ([2, 3], "torch.float32", 0, 0)

    changed = base.clone()
    changed[0, 0] = 9.0
    assert bench.tensor_digest(changed)["sha256"] != digest["sha256"]
    assert bench.tensor_digest(base.to(torch.float64))["sha256"] != digest["sha256"]
    assert bench.tensor_digest(base.reshape(3, 2))["sha256"] != digest["sha256"]
    # A view that is not contiguous digests like its contiguous copy.
    assert bench.tensor_digest(base.t()) == bench.tensor_digest(base.t().contiguous())

    counted = bench.tensor_digest(torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0]))
    assert (counted["nan"], counted["inf"]) == (1, 2)
    assert bench.tensor_digest(torch.ones(2, 2, dtype=torch.bfloat16))["dtype"] == "torch.bfloat16"
    assert bench.tensor_digest(torch.tensor(1.5))["shape"] == []
    assert bench.tensor_digest(torch.empty(0, 3))["shape"] == [0, 3]
    assert bench.tensor_digest(torch.arange(3))["nan"] == 0

    with pytest.raises(bench.EvidenceError, match="a tensor is required"):
        bench.tensor_digest([1.0, 2.0])


def test_package_tree_sha256_covers_paths_and_content(tmp_path):
    package = tmp_path / "package"
    (package / "sub").mkdir(parents=True)
    (package / "a.py").write_text("A = 1\n", encoding="utf-8")
    (package / "sub" / "b.py").write_text("B = 2\n", encoding="utf-8")
    (package / "notes.txt").write_text("ignored", encoding="utf-8")
    first = bench.package_tree_sha256(package)

    (package / "notes.txt").write_text("still ignored", encoding="utf-8")
    assert bench.package_tree_sha256(package) == first

    (package / "sub" / "b.py").write_text("B = 3\n", encoding="utf-8")
    changed = bench.package_tree_sha256(package)
    assert changed != first

    (package / "sub" / "b.py").rename(package / "sub" / "c.py")
    assert bench.package_tree_sha256(package) not in (first, changed)


def test_write_json_replaces_values_json_cannot_hold(tmp_path):
    path = tmp_path / "record.json"
    data = {"nan": float("nan"), "inf": float("inf"), "tuple": (1, 2), 3: "integer key", "object": Path("x")}

    bench._write_json(path, data)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "nan": None,
        "inf": None,
        "tuple": [1, 2],
        "3": "integer key",
        "object": repr(Path("x")),
    }
    assert list(tmp_path.iterdir()) == [path]


def test_environment_has_no_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub_secret_value")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor-cache")

    environment = bench.collect_environment()

    text = json.dumps(bench._json_safe(environment))
    for secret in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_secret_value", "hub_secret_value"):
        assert secret not in text
    assert environment["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["env"]["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/inductor-cache"
    assert environment["torch"] == str(torch.__version__)


# ---------------------------------------------------------------------------
# Frames, LPIPS and the side-by-side video
# ---------------------------------------------------------------------------


def _outputs(image: Any, error: Any = None) -> list[Any]:
    return [SimpleNamespace(images=[image], error=error)]


def test_extract_frames_layouts():
    video = torch.rand(1, 3, 5, 4, 6) * 2.0 - 1.0
    expected = (video[0].permute(1, 2, 3, 0) * 0.5 + 0.5).numpy()

    np.testing.assert_allclose(bench.extract_frames(_outputs(video)), expected)
    np.testing.assert_allclose(bench.extract_frames(_outputs(video[0])), expected)
    np.testing.assert_allclose(bench.extract_frames(_outputs((video, torch.zeros(1, 16)))), expected)
    np.testing.assert_allclose(bench.extract_frames(_outputs({"video": video})), expected)
    np.testing.assert_allclose(bench.extract_frames(_outputs({"frames": video})), expected)

    frames_last = torch.zeros(1, 5, 4, 6, 3, dtype=torch.uint8)
    extracted = bench.extract_frames(_outputs(frames_last))
    assert (extracted.shape, extracted.dtype) == ((5, 4, 6, 3), np.uint8)
    assert bench.extract_frames(_outputs(np.zeros((1, 5, 4, 6, 3), dtype=np.uint8))).shape == (5, 4, 6, 3)
    assert bench.extract_frames(_outputs(np.zeros((5, 4, 6, 3), dtype=np.uint8))).shape == (5, 4, 6, 3)
    # A numpy output is already in [0, 1] and is not rescaled.
    unit_range = np.linspace(0.0, 1.0, 5 * 4 * 6 * 3, dtype=np.float32).reshape(1, 5, 4, 6, 3)
    np.testing.assert_array_equal(bench.extract_frames(_outputs(unit_range)), unit_range[0])

    # Three RGB frames (frames first) and a channels-first RGB video of five frames have the same shape.
    with pytest.raises(bench.EvidenceError, match="is ambiguous"):
        bench.extract_frames(_outputs(torch.zeros(1, 3, 5, 4, 3)))


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        pytest.param([SimpleNamespace(images=[], error=None)], "has no images", id="no-images"),
        pytest.param(_outputs(torch.zeros(1), error="out of memory"), "returned an error", id="request-error"),
        pytest.param(_outputs({"audio": torch.zeros(1, 16)}), "holds no video", id="audio-only"),
        pytest.param(_outputs(torch.zeros(4, 6, 3)), "got an array of shape", id="one-image"),
    ],
)
def test_extract_frames_rejects_outputs_without_video(outputs, message):
    with pytest.raises(bench.EvidenceError, match=message):
        bench.extract_frames(outputs)


def test_frames_to_uint8_counts_nan_and_inf():
    frames = np.zeros((2, 2, 2, 3), dtype=np.float32)
    frames[0, 0, 0, 0] = np.nan
    frames[1, 0, 0, 1] = np.inf
    frames[1, 1, 1, 2] = 2.0
    frames[0, 1, 1, 1] = 0.5

    array, info = bench.frames_to_uint8(frames)

    assert (array.dtype, array.shape) == (np.uint8, (2, 2, 2, 3))
    assert (array[0, 0, 0, 0], array[1, 0, 0, 1], array[1, 1, 1, 2], array[0, 1, 1, 1]) == (0, 0, 255, 128)
    assert (info["nan_count"], info["inf_count"]) == (1, 1)
    assert (info["shape"], info["dtype"], info["source_dtype"]) == ([2, 2, 2, 3], "uint8", "float32")
    assert info["raw"]["shape"] == [2, 2, 2, 3]
    assert info["raw"]["sha256"] != bench.frames_to_uint8(np.zeros((2, 2, 2, 3), dtype=np.float32))[1]["raw"]["sha256"]


def test_frames_to_uint8_channel_layouts():
    rgb = np.full((1, 2, 2, 3), 7, dtype=np.uint8)
    rgba = np.full((1, 2, 2, 4), 7, dtype=np.uint8)
    gray = np.full((1, 2, 2, 1), 7, dtype=np.uint8)

    for frames in (rgb, rgba, gray):
        array, info = bench.frames_to_uint8(frames)
        assert np.array_equal(array, rgb)
        assert info["shape"] == [1, 2, 2, 3]

    for frames in (np.zeros((1, 2, 2, 2), dtype=np.uint8), np.zeros((1, 2, 2, 3), dtype=np.int16), np.zeros((2, 2, 3))):
        with pytest.raises(bench.EvidenceError):
            bench.frames_to_uint8(frames)


def test_check_frame_pair():
    clean = {"shape": [4, 8, 8, 3], "nan_count": 0, "inf_count": 0}

    assert bench.check_frame_pair(clean, clean, 4) == []
    assert bench.check_frame_pair(clean, clean, None) == []
    assert bench.check_frame_pair(clean, {**clean, "shape": [3, 8, 8, 3]}, None) == ["missing_frames"]
    assert bench.check_frame_pair(clean, clean, 5) == ["missing_frames"]
    assert bench.check_frame_pair(clean, {**clean, "shape": [4, 8, 16, 3]}, 4) == ["shape_mismatch"]
    assert bench.check_frame_pair({**clean, "nan_count": 2}, clean, 4) == ["nan"]
    assert bench.check_frame_pair(clean, {**clean, "inf_count": 1}, 4) == ["inf"]
    assert bench.check_frame_pair({}, clean, None) == ["missing_frames", "shape_mismatch"]
    broken = {"shape": [3, 8, 16, 3], "nan_count": 1, "inf_count": 1}
    assert bench.check_frame_pair(clean, broken, 4) == ["missing_frames", "shape_mismatch", "nan", "inf"]


def test_lpips_video_with_fake_loss():
    reference = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    candidate = np.full((3, 4, 4, 3), 255, dtype=np.uint8)
    values = iter([0.1, 0.2, 0.6])
    seen = []

    def loss_fn(left, right):
        seen.append((tuple(left.shape), float(left.min()), float(right.max())))
        return torch.tensor([[[[next(values)]]]])

    result = bench.lpips_video(reference, candidate, net="vgg", loss_fn=loss_fn)

    assert (result["frames"], result["net"], result["device"]) == (3, "vgg", "cpu")
    assert result["per_frame"] == pytest.approx([0.1, 0.2, 0.6])
    assert result["value"] == pytest.approx(0.3)
    assert seen == [((1, 3, 4, 4), -1.0, 1.0)] * 3


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        pytest.param(np.zeros((3, 4, 4, 3), np.uint8), np.zeros((2, 4, 4, 3), np.uint8), id="frame-count"),
        pytest.param(np.zeros((3, 4, 4, 3), np.uint8), np.zeros((3, 4, 8, 3), np.uint8), id="size"),
        pytest.param(np.zeros((3, 4, 4, 3), np.float32), np.zeros((3, 4, 4, 3), np.float32), id="dtype"),
        pytest.param(np.zeros((0, 4, 4, 3), np.uint8), np.zeros((0, 4, 4, 3), np.uint8), id="no-frames"),
    ],
)
def test_lpips_video_rejects_inputs_that_are_not_aligned(reference, candidate):
    calls = []

    with pytest.raises(bench.EvidenceError, match="lpips inputs are not aligned"):
        bench.lpips_video(reference, candidate, loss_fn=lambda left, right: calls.append(1))
    assert calls == []


def test_lpips_video_scores_on_the_named_device(monkeypatch):
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    moves = []

    class FakeNetwork:
        def __init__(self, net):
            self.net = net

        def eval(self):
            return self

        def to(self, device):
            moves.append(device)
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return lambda left, right: torch.zeros(1)

    _fake_module(monkeypatch, "lpips", LPIPS=FakeNetwork)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # A named device is the only one tried, also when a GPU is available.
    assert bench.lpips_video(frames, frames, device="cpu")["device"] == "cpu"
    assert moves == ["cpu"]
    with pytest.raises(bench.EvidenceError, match="lpips_failed: RuntimeError: CUDA out of memory"):
        bench.lpips_video(frames, frames, device="cuda")
    assert moves == ["cpu", "cuda"]

    # Without a device the GPU comes first, and the CPU gives the scores when the GPU fails.
    result = bench.lpips_video(frames, frames)
    assert (result["device"], result["per_frame"], result["value"]) == ("cpu", [0.0, 0.0], 0.0)
    assert moves == ["cpu", "cuda", "cuda", "cpu"]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bench.lpips_video(frames, frames)["device"] == "cpu"
    assert moves == ["cpu", "cuda", "cuda", "cpu", "cpu"]


def test_side_by_side():
    reference = np.full((2, 6, 40, 3), 10, dtype=np.uint8)
    candidate = np.full((2, 6, 40, 3), 200, dtype=np.uint8)

    combined = bench.side_by_side(reference, candidate, ["dense", "scheduled"])

    band = bench.LABEL_BAND
    assert (combined.shape, combined.dtype) == ((2, 6 + band, 80, 3), np.uint8)
    assert np.array_equal(combined[:, band:, :40], reference)
    assert np.array_equal(combined[:, band:, 40:], candidate)
    # Both labels are drawn, and every frame carries the same band.
    assert combined[0, :band, :40].max() > 0
    assert combined[0, :band, 40:].max() > 0
    assert np.array_equal(combined[0, :band], combined[1, :band])

    with pytest.raises(bench.EvidenceError):
        bench.side_by_side(reference, candidate[:1], ["dense", "scheduled"])
    with pytest.raises(bench.EvidenceError):
        bench.side_by_side(reference, candidate, ["dense"])
    with pytest.raises(bench.EvidenceError):
        bench.side_by_side(reference.astype(np.float32), candidate.astype(np.float32), ["dense", "scheduled"])


def test_compare_to_prior_measures_the_difference(tmp_path):
    prior_frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    frames_path = tmp_path / "prechange-plain.frames.npy"
    np.save(frames_path, prior_frames)
    output = {"frames_file": frames_path.name, "frames_sha256": bench.sha256_file(frames_path)}
    prior = tmp_path / "prechange.json"
    prior_record = {"session_id": "prechange-session", "requests": [{"name": "plain", "output": output}]}
    prior.write_text(json.dumps(prior_record), encoding="utf-8")
    current = prior_frames.copy()
    current[1, 0, 0, 0] = 3

    block = bench._compare_to_prior(prior, current)

    assert block["error"] is None
    assert block["prior_session_id"] == "prechange-session"
    assert block["prior_frames_sha256"] == output["frames_sha256"]
    assert (block["bit_identical"], block["max_abs_diff_uint8"], block["differing_frames"]) == (False, 3, 1)
    assert block["mean_abs_diff_uint8"] == pytest.approx(3 / current.size)
    assert block["prior_sha256"] == bench.sha256_file(prior)
    assert bench._compare_to_prior(prior, prior_frames)["bit_identical"] is True
    assert bench._compare_to_prior(prior, current[:1])["error"] == "shape_mismatch"
    assert bench._compare_to_prior(tmp_path / "absent.json", current)["error"].startswith("FileNotFoundError")

    frames_path.write_bytes(b"changed after the run")
    changed = bench._compare_to_prior(prior, current)
    assert "does not match its recorded sha256" in changed["error"]
    assert changed["bit_identical"] is False


# ---------------------------------------------------------------------------
# Verdict rules
# ---------------------------------------------------------------------------


def test_complete_evidence_is_success(make_sessions):
    results = {}
    for combination in bench.COMBINATIONS:
        sessions, directory = make_sessions(combination)
        result = _evaluate(sessions, directory)

        assert result["combination"] == combination
        assert result["status"] == "success", _not_passing(result)
        for name in bench.REQUIRED_ITEMS:
            assert result["items"][name]["status"] == "pass", name
            assert result["items"][name]["evidence"], name
        assert result["items"]["AE5_boundary"]["status"] == "pass"
        assert result["report"]["no_schedule"] == {"bit_identical": True}
        assert result["report"]["reference_vs_plain"] == {"bit_identical": True}
        results[combination] = result

    matrix = bench.evaluate_matrix(results)
    summary = bench.render_summary(results, matrix)

    assert matrix == {
        "status": "success",
        "missing": [],
        "ae5": {"status": "pass", "combinations": list(bench.COMBINATIONS), "failed": []},
    }
    assert summary.splitlines()[0] == "R7-R10: SUCCESS"
    assert summary.count("SUCCESS") == 1
    assert "  LPIPS 0.12 (alex, 2 frames)" in summary


@pytest.mark.parametrize("combination", ["C1", "C3", "C5"])
def test_every_evidence_pointer_is_load_bearing(make_sessions, combination):
    sessions, directory = make_sessions(combination, steps=5, switch=2)
    baseline = _evaluate(sessions, directory)
    assert baseline["status"] == "success", _not_passing(baseline)
    session_of = {file_name: name for name, file_name in bench.SESSION_FILES.items()}

    checked = 0
    for item in bench.REQUIRED_ITEMS:
        for reference in baseline["items"][item]["evidence"]:
            file_name, _, pointer = reference.partition("#")
            changed = copy.deepcopy(sessions)
            parent = bench.resolve_pointer(changed[session_of[file_name]], pointer.rpartition("/")[0])
            assert isinstance(parent, dict), reference
            _put(changed[session_of[file_name]], pointer, _DROP)

            result = _evaluate(changed, directory)

            assert result["items"][item]["status"] != "pass", (item, reference)
            assert result["status"] != "success", (item, reference)
            checked += 1
    assert checked > 10 * len(bench.REQUIRED_ITEMS)

    # The shifted-boundary item is not required of every combination, so only the item is checked.
    shifted_evidence = baseline["items"]["AE5_boundary"]["evidence"]
    assert len(shifted_evidence) > 10
    for reference in shifted_evidence:
        file_name, _, pointer = reference.partition("#")
        changed = copy.deepcopy(sessions)
        _put(changed[session_of[file_name]], pointer, _DROP)

        assert _evaluate(changed, directory)["items"]["AE5_boundary"]["status"] != "pass", reference


@pytest.mark.parametrize(
    ("select_shift", "diverge_shift", "status", "reason"),
    [
        pytest.param(1, 1, "fail", "selection_mismatch", id="layers-switch-one-step-late"),
        pytest.param(0, -1, "fail", "diverged_before_switch", id="output-changes-one-step-early"),
        pytest.param(0, 1, "not_proven", "no_divergence_at_switch", id="output-changes-one-step-late"),
    ],
)
def test_shifted_boundary_is_detected(make_sessions, select_shift, diverge_shift, status, reason):
    switch = 3
    sessions, directory = make_sessions("C1", switch=switch)
    _rebuild(sessions, directory, _CA, select_from=switch + select_shift, diverge_from=switch + diverge_shift)

    result = _evaluate(sessions, directory)

    item = result["items"]["R7_switch"]
    assert (item["status"], item["reasons"]) == (status, [reason])
    assert result["status"] == "not_success"
    assert "SUCCESS" not in _summary(result)


@pytest.mark.parametrize(("schedule", "switch"), [(_from(0), 0), ([], None)], ids=["switch-at-0", "no-profile"])
def test_dense_prefix_rule_needs_a_dense_step_before_the_switch(make_sessions, schedule, switch):
    sessions, _ = make_sessions("C1")
    # validate_config rejects these schedules, so only a record that was edited reaches the rule with one.
    _put(sessions["scheduled"], "/config/candidate_schedule", schedule)

    item = bench._rule_r9_dense_prefix(sessions["scheduled"])

    assert (item["status"], item["reasons"]) == ("not_proven", ["no_dense_prefix"])
    assert item["values"] == {"first_switch": switch}


def test_output_change_before_the_switch_fails_the_dense_prefix(make_sessions):
    switch = 3
    sessions, directory = make_sessions("C1", switch=switch)
    _rebuild(sessions, directory, _CA, select_from=switch, diverge_from=switch - 1)

    result = _evaluate(sessions, directory)

    prefix = result["items"]["R9_dense_prefix"]
    assert (prefix["status"], prefix["reasons"]) == ("fail", ["prefix_differs"])
    assert prefix["values"] == {"first_switch": switch, "first_differing_step": switch - 1}
    assert result["items"]["R7_switch"]["values"]["first_differing_step"] == switch - 1


def test_late_layer_switch_names_the_step(make_sessions):
    switch = 3
    sessions, directory = make_sessions("C1", switch=switch)
    _rebuild(sessions, directory, _CA, select_from=switch + 1, diverge_from=switch + 1)

    item = _evaluate(sessions, directory)["items"]["R7_switch"]

    assert item["values"]["steps"] == [{"sequence": 0, "step": switch, "problem": "expected_profile"}]


@pytest.mark.parametrize(
    ("index", "select_from", "status", "reason"),
    [
        pytest.param(_CA, None, "fail", "selection_mismatch", id="candidate-all-baseline"),
        pytest.param(_CA, 3, "not_proven", "no_divergence_at_switch", id="profile-selected-output-unchanged"),
        pytest.param(_RA, 3, "fail", "selection_mismatch", id="reference-selects-a-profile"),
    ],
)
def test_all_dense_candidate_is_detected(make_sessions, index, select_from, status, reason):
    sessions, directory = make_sessions("C1", switch=3)
    _rebuild(sessions, directory, index, select_from=select_from, diverge_from=None)

    result = _evaluate(sessions, directory)

    item = result["items"]["R7_switch"]
    assert (item["status"], item["reasons"]) == (status, [reason])
    assert result["status"] == "not_success"
    assert "SUCCESS" not in _summary(result)


def test_early_layer_switch_names_the_step(make_sessions):
    switch = 3
    sessions, directory = make_sessions("C1", switch=switch)
    _rebuild(sessions, directory, _CA, select_from=switch - 1, diverge_from=switch)

    item = _evaluate(sessions, directory)["items"]["R7_switch"]

    assert (item["status"], item["reasons"]) == ("fail", ["selection_mismatch"])
    assert item["values"]["steps"] == [{"sequence": 0, "step": switch - 1, "problem": "expected_baseline"}]


def test_approximate_kernel_call_on_a_dense_step_is_a_mismatch(make_sessions):
    sessions, directory = make_sessions("C1")
    # Every layer of the reference reports its own implementation, and one kernel call is quantized.
    _put(sessions["scheduled"], f"/requests/{_RA}/sequences/0/steps/1/attention/kernel/sage_calls", 1)

    result = _evaluate(sessions, directory)

    item = result["items"]["R7_switch"]
    assert (item["status"], item["reasons"]) == ("fail", ["selection_mismatch"])
    assert item["values"]["steps"] == [{"sequence": 0, "step": 1, "problem": "approximate_kernel_on_dense_step"}]
    assert result["status"] == "not_success"


@pytest.mark.parametrize(
    ("step", "edits", "problem"),
    [
        pytest.param(4, {"selection": {_PROFILE: 4, "<unscheduled>": 2}}, None, id="layers-without-a-schedule"),
        pytest.param(4, {"selection": {f"{_PROFILE}|other": 6}}, None, id="profiles-that-share-an-implementation"),
        pytest.param(1, {"selection": {"<baseline>": 4, "<unscheduled>": 2}}, None, id="dense-step"),
        pytest.param(4, {"selection": {"other": 6}}, "other_profile", id="other-profile"),
        pytest.param(4, {"selection": {_PROFILE: 3, "<baseline>": 3}}, "expected_profile", id="some-layers-dense"),
        pytest.param(4, {"selection": {"<unscheduled>": 6}}, "expected_profile", id="no-scheduled-layer"),
        pytest.param(4, {"selection": {"<unknown>": 6}}, "expected_profile", id="unknown-implementation"),
        pytest.param(1, {"selection": {"<unscheduled>": 6}}, "expected_baseline", id="dense-step-not-scheduled"),
        pytest.param(4, {"step_mismatch_calls": 1}, "step_mismatch", id="step-index-of-another-step"),
        pytest.param(4, {"layer_calls": 0}, "no_layer_calls", id="no-layer-call"),
        pytest.param(4, {"probe_errors": 1}, "probe_errors", id="probe-error"),
    ],
)
def test_switch_rule_reads_the_recorded_selection(make_sessions, step, edits, problem):
    sessions, directory = make_sessions("C1", switch=3)
    for key, value in edits.items():
        _put(sessions["scheduled"], f"/requests/{_CA}/sequences/0/steps/{step}/attention/{key}", value)

    item = _evaluate(sessions, directory)["items"]["R7_switch"]

    if problem is None:
        assert (item["status"], item["reasons"]) == ("pass", [])
    else:
        assert (item["status"], item["reasons"]) == ("fail", ["selection_mismatch"])
        assert item["values"]["steps"] == [{"sequence": 0, "step": step, "problem": problem}]


def test_switch_rule_needs_an_attention_record_on_every_step(make_sessions):
    sessions, directory = make_sessions("C1", switch=3)
    _put(sessions["scheduled"], f"/requests/{_CA}/sequences/0/steps/4/attention", None)

    item = _evaluate(sessions, directory)["items"]["R7_switch"]

    assert (item["status"], item["reasons"]) == ("fail", ["selection_mismatch"])
    assert item["values"]["steps"] == [{"sequence": 0, "step": 4, "problem": "attention_not_recorded"}]


def test_switch_rule_reads_every_sequence(make_sessions):
    sessions, directory = make_sessions("C1", switch=3)
    for session in bench.SESSIONS:
        sessions[session]["config"]["expected_sequences"] = 2
        for request in sessions[session]["requests"]:
            request["sequences"].append(copy.deepcopy(request["sequences"][0]))
    baseline = _evaluate(sessions, directory)
    assert baseline["status"] == "success", _not_passing(baseline)

    # One sequence of the candidate switches one step late: sequence 0 first, then sequence 1.
    _steps(sessions, _CA)[3]["attention"] = _attention(None)
    first = _evaluate(sessions, directory)["items"]["R7_switch"]
    sessions["scheduled"]["requests"][_CA]["sequences"].reverse()
    second = _evaluate(sessions, directory)["items"]["R7_switch"]

    assert first["values"]["steps"] == [{"sequence": 0, "step": 3, "problem": "expected_profile"}]
    assert second["values"]["steps"] == [{"sequence": 1, "step": 3, "problem": "expected_profile"}]


@pytest.mark.parametrize("reason", ["short_kv", "ignored_layer", "private_gate", "layer_fallback"])
def test_all_fallback_is_not_proven(make_sessions, reason):
    steps, switch = 8, 3
    sessions, directory = make_sessions("C2", steps=steps, switch=switch)
    for step in _steps(sessions, _CA)[switch:]:
        _make_fallback(step, reason)

    result = _evaluate(sessions, directory)

    target = result["items"]["R10_target"]
    assert (target["status"], target["reasons"]) == ("not_proven", ["all_fallback"])
    assert target["values"]["fallback_totals"][reason] == _LAYER_CALLS * (steps - switch)
    assert target["values"]["approximate"] == {"approx_quantized": 0, "approx_sparse": 0, "approx_calls": 0}
    assert target["values"]["steps_all_fallback"] == list(range(switch, steps))
    assert result["items"]["R7_switch"]["reasons"] == ["target_kind_not_observed_at_switch"]
    assert result["status"] == "not_success"
    assert "SUCCESS" not in _summary(result)


def test_partial_fallback_is_reported(make_sessions):
    steps, switch = 8, 3
    sessions, directory = make_sessions("C1", steps=steps, switch=switch)
    for step in _steps(sessions, _CA)[switch + 1 :]:
        _make_fallback(step, "short_kv")

    result = _evaluate(sessions, directory)

    target = result["items"]["R10_target"]
    assert target["status"] == "pass"
    assert target["values"]["target_calls"] == _LAYER_CALLS
    assert target["values"]["fallback_totals"]["short_kv"] == _LAYER_CALLS * (steps - switch - 1)
    assert target["values"]["steps_all_fallback"] == list(range(switch + 1, steps))
    assert result["status"] == "success", _not_passing(result)
    assert result["report"]["backend"]["steps_all_fallback"] == list(range(switch + 1, steps))


@pytest.mark.parametrize(
    ("pointer", "value", "reason"),
    [
        pytest.param("/probes/kernel/module", "tests.fake_kernel", "kernel_not_real", id="fake-kernel"),
        pytest.param("/probes/kernel", None, "kernel_not_real", id="no-kernel-probe"),
        pytest.param(
            f"/config/approx_profiles/{_PROFILE}/backend",
            "FASTVIDEO_VSA",
            "no_probe_for_backend",
            id="backend-without-probe",
        ),
        pytest.param(
            f"/requests/{_CA}/sequences/0/steps/1/attention/probe_errors", 1, "probe_errors", id="probe-error"
        ),
    ],
)
def test_target_kernel_must_be_real(make_sessions, pointer, value, reason):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    target = result["items"]["R10_target"]
    assert (target["status"], target["reasons"]) == ("not_proven", [reason])
    assert result["status"] == "not_success"
    assert "SUCCESS" not in _summary(result)


def _measured_steps(scheduled: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the steps of the two requests the compile rule reads."""
    return [step for index in (_RA, _CA) for step in scheduled["requests"][index]["sequences"][0]["steps"]]


def _no_compile_call(scheduled: dict[str, Any]) -> None:
    scheduled["probes"]["compile"]["setup_calls"] = []


def _compile_call_without_blocks(scheduled: dict[str, Any]) -> None:
    scheduled["probes"]["compile"]["setup_calls"][0]["compiled_blocks"] = 0


def _no_graph_compiled(scheduled: dict[str, Any]) -> None:
    scheduled["probes"]["compile"]["graphs_compiled"] = 0


def _every_block_call_eager(scheduled: dict[str, Any]) -> None:
    for step in _measured_steps(scheduled):
        counts = step["compile"]
        counts.update(eager_block_calls=counts["block_calls"], graph_execs=0)
        counts["by_model"]["0"]["eager_block_calls"] = counts["block_calls"]


def _one_eager_block_call(scheduled: dict[str, Any]) -> None:
    scheduled["requests"][_CA]["sequences"][0]["steps"][4]["compile"]["eager_block_calls"] = 1


def _fewer_graph_runs_than_block_calls(scheduled: dict[str, Any]) -> None:
    scheduled["requests"][_RA]["sequences"][0]["steps"][2]["compile"]["graph_execs"] = 5


def _no_block_call_captured(scheduled: dict[str, Any]) -> None:
    for step in _measured_steps(scheduled):
        step["compile"].update(block_calls=0, eager_block_calls=0, graph_execs=0, by_model={})


def _service_ran_eager(scheduled: dict[str, Any]) -> None:
    scheduled["startup"]["effective"]["enforce_eager"] = True


def _compile_error_recorded(scheduled: dict[str, Any]) -> None:
    scheduled["probes"]["compile"]["compile_errors"] = 1


def _second_model_not_compiled(scheduled: dict[str, Any]) -> None:
    setup_calls = scheduled["probes"]["compile"]["setup_calls"]
    setup_calls.append({**setup_calls[0], "index": 1, "compiled_blocks": 0})
    first_step = scheduled["requests"][_CA]["sequences"][0]["steps"][0]
    first_step["compile"]["by_model"]["1"] = {"block_calls": 2, "eager_block_calls": 0}


@pytest.mark.parametrize(
    ("change", "status", "reason"),
    [
        pytest.param(_no_compile_call, "fail", "not_compiled", id="no-compile-call"),
        pytest.param(_compile_call_without_blocks, "fail", "not_compiled", id="compile-call-without-blocks"),
        pytest.param(_no_graph_compiled, "fail", "not_compiled", id="no-graph-compiled"),
        pytest.param(_every_block_call_eager, "fail", "eager_block_calls", id="every-block-call-eager"),
        pytest.param(_one_eager_block_call, "fail", "eager_block_calls", id="one-eager-block-call"),
        pytest.param(_fewer_graph_runs_than_block_calls, "fail", "eager_block_calls", id="fewer-graph-runs"),
        pytest.param(_no_block_call_captured, "not_proven", "no_block_calls_captured", id="no-block-call"),
        pytest.param(_service_ran_eager, "fail", "enforce_eager", id="enforce-eager"),
        pytest.param(_compile_error_recorded, "fail", "compile_errors", id="compile-error"),
        pytest.param(_second_model_not_compiled, "fail", "uncompiled_model_used", id="second-model-not-compiled"),
    ],
)
def test_all_eager_is_compile_failure(make_sessions, change, status, reason):
    sessions, directory = make_sessions("C1")
    change(sessions["scheduled"])

    result = _evaluate(sessions, directory)

    compiled = result["items"]["R10_compiled"]
    assert compiled["status"] == status
    assert reason in compiled["reasons"]
    assert result["status"] == "not_success"
    assert "SUCCESS" not in _summary(result)


def test_compile_rule_names_the_eager_step(make_sessions):
    sessions, directory = make_sessions("C1")
    _one_eager_block_call(sessions["scheduled"])

    compiled = _evaluate(sessions, directory)["items"]["R10_compiled"]

    assert compiled["reasons"] == ["eager_block_calls"]
    assert compiled["values"] == {"steps": [{"request": "candidate", "sequence": 0, "step": 4}]}


@pytest.mark.parametrize(
    ("pointer", "value"),
    [
        pytest.param("/startup/effective", None, id="no-read-back"),
        pytest.param("/startup/effective/enforce_eager", None, id="setting-unknown"),
        pytest.param("/startup/effective/enforce_eager", _DROP, id="setting-absent"),
    ],
)
def test_compile_rule_accepts_a_missing_read_back(make_sessions, pointer, value):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    compiled = result["items"]["R10_compiled"]
    assert compiled["status"] == "pass"
    # The counts are the proof. The read-back is not listed as evidence when it does not decide.
    assert not any("/startup/effective" in reference for reference in compiled["evidence"])
    assert result["status"] == "success", _not_passing(result)


def test_compile_rule_lists_the_read_back_that_says_eager(make_sessions):
    sessions, directory = make_sessions("C1")
    sessions["scheduled"]["startup"]["effective"]["enforce_eager"] = True

    compiled = _evaluate(sessions, directory)["items"]["R10_compiled"]

    assert (compiled["status"], compiled["reasons"]) == ("fail", ["enforce_eager"])
    assert "scheduled.json#/startup/effective/enforce_eager" in compiled["evidence"]


@pytest.mark.parametrize(
    ("pointer", "value", "status", "reason"),
    [
        pytest.param(
            f"/requests/{_CA}/output/shape",
            [_FRAMES - 1, 4, 4, 3],
            "not_proven",
            "depends:conditions",
            id="one-frame-fewer",
        ),
        pytest.param(
            f"/requests/{_CA}/output/shape", [_FRAMES, 8, 4, 3], "not_proven", "depends:conditions", id="other-size"
        ),
        pytest.param("/config/expected_frames", _FRAMES + 1, "fail", "missing_frames", id="expected-frames"),
        pytest.param(f"/requests/{_CA}/output/nan_count", 3, "fail", "nan", id="nan"),
        pytest.param(f"/requests/{_RA}/output/inf_count", 1, "fail", "inf", id="inf"),
        pytest.param("/comparison/lpips/error", "shape_mismatch", "fail", "shape_mismatch", id="stored-error"),
        pytest.param("/comparison/lpips/frames", _FRAMES - 1, "fail", "misaligned", id="misaligned"),
        pytest.param("/comparison/lpips/value", None, "fail", "lpips_value_missing", id="no-value"),
        pytest.param("/comparison/lpips/per_frame", [0.1], "fail", "lpips_value_missing", id="per-frame-count"),
        pytest.param(
            "/comparison/lpips/weights_sha256", None, "not_proven", "lpips_metadata_incomplete", id="no-weights-hash"
        ),
        pytest.param(
            "/comparison/lpips/error",
            "lpips_unavailable: ModuleNotFoundError: No module named 'lpips'",
            "not_measured",
            "lpips_unavailable",
            id="lpips-not-installed",
        ),
        pytest.param(
            "/comparison/lpips/error",
            "lpips_failed: RuntimeError: CUDA out of memory",
            "not_measured",
            "lpips_failed",
            id="lpips-did-not-run",
        ),
        pytest.param("/comparison", None, "not_measured", "comparison_absent", id="no-comparison"),
    ],
)
def test_output_problems_block_success(make_sessions, pointer, value, status, reason):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    item = result["items"]["R8_lpips"]
    assert (item["status"], item["reasons"]) == (status, [reason])
    assert result["status"] == "not_success"
    assert "SUCCESS" not in _summary(result)


@pytest.mark.parametrize(
    ("pointer", "value", "reason"),
    [
        pytest.param(f"/requests/{_CA}/sampling_params/seed", 8, "sampling_params_differ", id="seed"),
        pytest.param(f"/requests/{_CA}/prompt", {"prompt": "a dog"}, "prompt_differs", id="prompt"),
        pytest.param(
            f"/requests/{_CA}/sequences/0/initial/video",
            _digest("other initial noise"),
            "initial_state_differs",
            id="initial-state",
        ),
        pytest.param(
            f"/requests/{_RA}/attention_schedule", _from(1), "reference_schedule_not_empty", id="reference-schedule"
        ),
        pytest.param(
            f"/requests/{_CA}/attention_schedule", _from(4), "candidate_schedule_differs", id="candidate-schedule"
        ),
    ],
)
def test_reference_condition_mismatch_blocks_success(make_sessions, pointer, value, reason):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    conditions = result["items"]["conditions"]
    assert (conditions["status"], conditions["reasons"]) == ("fail", [reason])
    for name in ("R7_switch", "R9_dense_prefix", "R8_lpips", "R8_video"):
        item = result["items"][name]
        assert item["status"] == "not_proven", name
        assert "depends:conditions" in item["reasons"], name
    assert result["status"] == "not_success"
    assert "SUCCESS" not in _summary(result)


def test_conditions_rule_compares_total_steps(make_sessions):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], f"/requests/{_CA}/sequences/0/total_steps", 9)

    # The capture rule reports this record first, so the conditions rule is called on its own here.
    item = bench._rule_conditions(sessions["scheduled"])
    result = _evaluate(sessions, directory)

    assert (item["status"], item["reasons"]) == ("fail", ["total_steps_differ"])
    assert result["items"]["capture"]["reasons"] == ["total_steps:candidate"]
    assert result["items"]["conditions"]["reasons"] == ["depends:capture"]


def test_conditions_rule_needs_the_initial_state(make_sessions):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], f"/requests/{_CA}/sequences/0/initial", {})

    conditions = _evaluate(sessions, directory)["items"]["conditions"]

    assert (conditions["status"], conditions["reasons"]) == ("not_proven", ["initial_state_not_captured"])


@pytest.mark.parametrize(
    ("pointer", "value", "values"),
    [
        pytest.param(
            f"/requests/{_RB}/sequences/0/steps/5/digests/video",
            _digest("drift"),
            {"first_differing_step": 5},
            id="step-digest",
        ),
        pytest.param(f"/requests/{_RB}/sequences/0/final/video", _digest("drift"), {}, id="final-state"),
        pytest.param(f"/requests/{_RB}/output/raw/sha256", _sha("drift"), {}, id="output"),
    ],
)
def test_reference_not_repeatable_gates_prefix(make_sessions, pointer, value, values):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    repeatable = result["items"]["dense_repeatable"]
    assert (repeatable["status"], repeatable["reasons"]) == ("fail", ["reference_not_repeatable"])
    assert repeatable["values"] == values
    prefix = result["items"]["R9_dense_prefix"]
    assert (prefix["status"], prefix["reasons"]) == ("not_proven", ["depends:dense_repeatable"])
    assert result["status"] == "not_success"


@pytest.mark.parametrize("requests", [(_RB,), (_RA, _RB)], ids=["one-reference", "both-references"])
def test_empty_step_digests_are_not_proof_of_a_repeatable_reference(make_sessions, requests):
    sessions, directory = make_sessions("C1")
    for index in requests:
        _put(sessions["scheduled"], f"/requests/{index}/sequences/0/steps/5/digests", {})

    result = _evaluate(sessions, directory)

    repeatable = result["items"]["dense_repeatable"]
    assert (repeatable["status"], repeatable["reasons"]) == ("not_proven", ["digests_not_captured"])
    prefix = result["items"]["R9_dense_prefix"]
    assert (prefix["status"], prefix["reasons"]) == ("not_proven", ["depends:dense_repeatable"])
    assert result["status"] == "not_success"


def test_empty_step_digests_do_not_hide_a_later_difference(make_sessions):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], f"/requests/{_RB}/sequences/0/steps/2/digests", {})
    _put(sessions["scheduled"], f"/requests/{_RB}/sequences/0/steps/5/digests/video", _digest("drift"))

    repeatable = _evaluate(sessions, directory)["items"]["dense_repeatable"]

    assert (repeatable["status"], repeatable["reasons"]) == ("fail", ["reference_not_repeatable"])
    assert repeatable["values"] == {"first_differing_step": 5}


def test_empty_step_digests_before_the_switch_are_not_proof_of_the_prefix(make_sessions):
    switch = 3
    sessions, directory = make_sessions("C1", switch=switch)
    _put(sessions["scheduled"], f"/requests/{_CA}/sequences/0/steps/1/digests", {})

    result = _evaluate(sessions, directory)

    for name in ("R7_switch", "R9_dense_prefix"):
        item = result["items"][name]
        assert (item["status"], item["reasons"]) == ("not_proven", ["digests_not_captured"]), name
    # The steps that were captured still show where the candidate leaves the reference.
    assert result["items"]["R7_switch"]["values"]["first_differing_step"] == switch
    assert result["status"] == "not_success"


@pytest.mark.parametrize(
    ("step_offset", "name", "status"),
    [
        pytest.param(-1, "noise_pred", "fail", id="prediction-before-the-switch"),
        pytest.param(0, "latent_in", "fail", id="input-of-the-switch-step"),
        pytest.param(0, "noise_pred", "pass", id="prediction-at-the-switch"),
    ],
)
def test_dense_prefix_rule_for_wan(make_sessions, step_offset, name, status):
    switch = 3
    sessions, directory = make_sessions("C5", switch=switch)
    step = switch + step_offset
    _put(sessions["scheduled"], f"/requests/{_CA}/sequences/0/steps/{step}/digests/{name}", _digest("changed"))

    prefix = _evaluate(sessions, directory)["items"]["R9_dense_prefix"]

    assert prefix["status"] == status
    assert prefix["reasons"] == ([] if status == "pass" else ["prefix_differs"])
    assert prefix["values"].get("first_differing_step") == (None if status == "pass" else step)


def _drift_plain_output(sessions: dict[str, Any]) -> None:
    """Make the no-schedule run differ from the pre-change run at one step and store a measured difference."""
    plain = sessions["plain"]
    _put(plain, "/requests/1/sequences/0/steps/2/digests/video", _digest("drift"))
    plain["comparison_to_prior"] = {
        "prior_file": "prechange.json",
        "prior_sha256": "prior-sha256",
        "prior_session_id": sessions["prechange"]["session_id"],
        "prior_frames_sha256": sessions["prechange"]["requests"][1]["output"]["frames_sha256"],
        "error": None,
        "bit_identical": False,
        "max_abs_diff_uint8": 3,
        "mean_abs_diff_uint8": 0.25,
        "differing_frames": 1,
    }


@pytest.mark.parametrize(
    ("tolerance", "status", "reasons"),
    [
        pytest.param(None, "fail", ["output_changed"], id="no-tolerance"),
        pytest.param({"max_abs_diff_uint8": 3}, "pass", ["within_stated_tolerance"], id="within-tolerance"),
        pytest.param({"max_abs_diff_uint8": 2}, "fail", ["output_changed"], id="beyond-tolerance"),
    ],
)
def test_no_schedule_regression_applies_the_stated_tolerance(make_sessions, tolerance, status, reasons):
    sessions, directory = make_sessions("C1")
    _drift_plain_output(sessions)
    sessions["plain"]["config"]["no_schedule_tolerance"] = tolerance

    item = _evaluate(sessions, directory)["items"]["R9_no_schedule"]

    assert (item["status"], item["reasons"]) == (status, reasons)
    assert item["values"] == {
        "bit_identical": False,
        "max_abs_diff_uint8": 3,
        "mean_abs_diff_uint8": 0.25,
        "differing_frames": 1,
        "tolerance": tolerance,
    }


def test_no_schedule_tolerance_needs_a_measured_difference(make_sessions):
    sessions, directory = make_sessions("C1")
    _drift_plain_output(sessions)
    sessions["plain"]["config"]["no_schedule_tolerance"] = {"max_abs_diff_uint8": 3}
    sessions["plain"]["comparison_to_prior"] = None

    item = _evaluate(sessions, directory)["items"]["R9_no_schedule"]

    assert (item["status"], item["reasons"]) == ("fail", ["output_changed"])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("prior_session_id", "another-session", id="another-session"),
        pytest.param("prior_session_id", None, id="no-session-id"),
        pytest.param("prior_frames_sha256", _sha("other frames"), id="other-frames"),
        pytest.param("prior_frames_sha256", None, id="no-frames-hash"),
    ],
)
def test_no_schedule_tolerance_needs_the_same_prior_record(make_sessions, key, value):
    sessions, directory = make_sessions("C1")
    _drift_plain_output(sessions)
    sessions["plain"]["config"]["no_schedule_tolerance"] = {"max_abs_diff_uint8": 3}
    sessions["plain"]["comparison_to_prior"][key] = value

    result = _evaluate(sessions, directory)

    item = result["items"]["R9_no_schedule"]
    assert (item["status"], item["reasons"]) == ("not_proven", ["prior_record_differs"])
    assert item["values"]["max_abs_diff_uint8"] == 3
    assert result["status"] == "not_success"


_DECLARED_HEAD = {"git_head": "head-before", "package_tree_sha256": None}
_UNCONFIRMED = ["prechange_identity_unconfirmed"]
_REQUEST_ERROR = {"error_type": "RuntimeError", "message": "generation failed"}


@pytest.mark.parametrize(
    ("edits", "status", "reasons"),
    [
        pytest.param([], "pass", [], id="identical"),
        pytest.param([("prechange", "/config_sha256", "other")], "not_proven", ["config_differs"], id="config"),
        pytest.param(
            [("scheduled", "/config_sha256", "other")], "not_proven", ["config_differs"], id="scheduled-config"
        ),
        pytest.param(
            [("prechange", "/source/package_tree_sha256", "tree-unknown")],
            "not_proven",
            _UNCONFIRMED,
            id="tree-not-the-declared-one",
        ),
        pytest.param(
            [
                ("plain", "/config/prechange_source/package_tree_sha256", _TREE_AFTER),
                ("prechange", "/source/package_tree_sha256", _TREE_AFTER),
            ],
            "not_proven",
            _UNCONFIRMED,
            id="same-tree-in-both-sessions",
        ),
        pytest.param(
            [("scheduled", "/source/package_tree_sha256", "tree-other")],
            "not_proven",
            _UNCONFIRMED,
            id="scheduled-ran-other-source",
        ),
        pytest.param([("plain", "/config/prechange_source", _DECLARED_HEAD)], "pass", [], id="declared-head"),
        pytest.param(
            [("plain", "/config/prechange_source", _DECLARED_HEAD), ("prechange", "/source/git_head", "other-head")],
            "not_proven",
            _UNCONFIRMED,
            id="head-differs",
        ),
        pytest.param(
            [("plain", "/config/prechange_source", _DECLARED_HEAD), ("prechange", "/source/git_status_clean", False)],
            "not_proven",
            _UNCONFIRMED,
            id="head-with-local-changes",
        ),
        pytest.param([("prechange", "/status", "aborted")], "not_proven", ["capture"], id="prechange-aborted"),
        pytest.param([("plain", "/requests/1/error", _REQUEST_ERROR)], "not_proven", ["capture"], id="plain-error"),
        pytest.param(
            [("plain", "/requests/1/sequences/0/incomplete", True)], "not_proven", ["capture"], id="plain-incomplete"
        ),
        pytest.param(
            [
                ("plain", "/requests/1/sequences/0/steps/2/digests", {}),
                ("prechange", "/requests/1/sequences/0/steps/2/digests", {}),
            ],
            "not_proven",
            ["digests_not_captured"],
            id="no-step-digests",
        ),
        pytest.param(
            [("prechange", "/requests/1/sequences/0/steps/2/digests", {})],
            "not_proven",
            ["digests_not_captured"],
            id="no-step-digests-before-the-change",
        ),
    ],
)
def test_no_schedule_regression(make_sessions, edits, status, reasons):
    sessions, directory = make_sessions("C1")
    for session, pointer, value in edits:
        _put(sessions[session], pointer, value)

    result = _evaluate(sessions, directory)

    item = result["items"]["R9_no_schedule"]
    assert (item["status"], item["reasons"]) == (status, reasons)
    assert item["values"] == ({"bit_identical": True} if status == "pass" else {})
    assert (result["status"] == "success") == (status == "pass")


@pytest.mark.parametrize("absent", ["plain", "prechange"])
def test_no_schedule_regression_is_not_measured_without_both_sessions(make_sessions, absent):
    sessions, directory = make_sessions("C1")
    sessions[absent] = None

    result = _evaluate(sessions, directory)

    item = result["items"]["R9_no_schedule"]
    assert (item["status"], item["reasons"]) == ("not_measured", ["plain_or_prechange_session_absent"])
    assert result["status"] == "not_success"


def test_ae5_boundary_passes_when_the_shifted_schedule_adds_no_graph(make_sessions):
    sessions, directory = make_sessions("C1", switch=3)

    item = _evaluate(sessions, directory)["items"]["AE5_boundary"]

    assert (item["status"], item["reasons"]) == ("pass", [])
    assert item["values"] == {"first_switch": 4, "candidate_first_switch": 3, "delta": 0}


@pytest.mark.parametrize(
    ("pointer", "value", "status", "reasons"),
    [
        pytest.param(f"/requests/{_BS}/graphs_compiled_after", 7, "fail", ["graphs_grew"], id="graphs-grew"),
        pytest.param(f"/requests/{_BS}/sampling_params/seed", 8, "fail", ["sampling_params_differ"], id="other-seed"),
        pytest.param(
            "/config/boundary_shift_schedule",
            _from(3),
            "fail",
            ["boundary_not_shifted", "selection_mismatch"],
            id="boundary-not-shifted",
        ),
        pytest.param(f"/requests/{_BS}/error", _REQUEST_ERROR, "not_proven", ["capture"], id="request-error"),
        pytest.param(
            f"/requests/{_BS}/sequences/0/steps/4/compile/eager_block_calls",
            1,
            "fail",
            ["eager_block_calls"],
            id="eager-block-call",
        ),
        pytest.param(
            f"/requests/{_BS}/sequences/0/steps/4/compile/graph_execs",
            5,
            "fail",
            ["eager_block_calls"],
            id="fewer-graph-runs",
        ),
        pytest.param(
            f"/requests/{_BS}/sequences/0/steps/4/compile/block_calls",
            0,
            "not_proven",
            ["no_block_calls_captured"],
            id="no-block-call",
        ),
        pytest.param(f"/requests/{_BS}", _DROP, "not_measured", ["boundary_shift_not_run"], id="not-run"),
        pytest.param(
            "/config/boundary_shift_schedule", None, "not_measured", ["boundary_shift_not_run"], id="not-configured"
        ),
    ],
)
def test_ae5_boundary(make_sessions, pointer, value, status, reasons):
    sessions, directory = make_sessions("C1", switch=3)
    _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    item = result["items"]["AE5_boundary"]
    assert (item["status"], item["reasons"]) == (status, reasons)
    # The shifted-boundary check is decided per matrix, so one combination still succeeds without it.
    assert result["status"] == "success", _not_passing(result)


def test_ae5_boundary_reports_the_added_graphs_and_the_wrong_steps(make_sessions):
    sessions, directory = make_sessions("C1", switch=3)
    _rebuild(sessions, directory, _BS, select_from=3, diverge_from=3, compiled=(5, 7))

    item = _evaluate(sessions, directory)["items"]["AE5_boundary"]

    assert (item["status"], item["reasons"]) == ("fail", ["graphs_grew", "selection_mismatch"])
    assert item["values"]["delta"] == 2
    assert item["values"]["steps"] == [{"sequence": 0, "step": 3, "problem": "expected_baseline"}]


def test_timing_report_holds_raw_samples_and_medians(make_sessions):
    steps, switch = 8, 3
    sessions, directory = make_sessions("C1", steps=steps, switch=switch)

    result = _evaluate(sessions, directory)

    assert result["items"]["R8_timing"]["status"] == "pass"
    timing = result["report"]["seconds_per_step"]
    assert timing["cold_start_seconds"] == 30.0
    assert timing["compile_warmup"] == [0.5] + [0.25] * (steps - 2) + [0.5]
    assert timing["reference"] == {"samples": [[0.5] * steps, [0.5] * steps], "median": 0.5}
    assert timing["candidate"] == {
        "samples": [0.5] * switch + [0.25] * (steps - switch),
        "median": 0.25,
        "dense_median": 0.5,
        "approx_median": 0.25,
    }
    assert timing["plain"] == {"samples": [0.5] * steps, "median": 0.5}
    assert (timing["batch_size"], timing["probes_active"]) == (1, True)


@pytest.mark.parametrize(
    ("pointer", "value", "reason"),
    [
        pytest.param(
            f"/requests/{_CA}/graphs_compiled_after",
            7,
            "compiled_during_measured_request:candidate",
            id="compiled-during-candidate",
        ),
        pytest.param(f"/requests/{_RA}/sequences/0/steps/1/batch_size", 2, "batch_size", id="batch-of-two"),
        pytest.param(f"/requests/{_WU}/sequences", [], "warmup_samples_missing", id="warmup-without-steps"),
        pytest.param("/startup/cold_start_seconds", None, "cold_start_missing", id="no-cold-start"),
        pytest.param(
            f"/requests/{_RB}/sequences/0/steps/2/seconds",
            None,
            "step_samples_missing:reference_b",
            id="step-without-time",
        ),
    ],
)
def test_timing_requires_recorded_steady_phases(make_sessions, pointer, value, reason):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    timing = result["items"]["R8_timing"]
    assert (timing["status"], timing["reasons"]) == ("not_proven", [reason])
    assert result["report"]["seconds_per_step"] == {}
    assert result["status"] == "not_success"


@pytest.mark.parametrize(
    ("warmup", "gaps"),
    [
        pytest.param({"select_from": 1, "select_to": 7}, [], id="both-paths-on-both-transformers"),
        pytest.param({"select_from": 1}, ["transformer_2:<baseline>"], id="no-dense-step-on-the-second-transformer"),
        pytest.param(
            {"select_from": 4, "select_to": 7}, ["transformer:approx"], id="no-profile-step-on-the-first-transformer"
        ),
        pytest.param({}, ["transformer:approx", "transformer_2:approx"], id="warmup-selected-no-profile"),
    ],
)
def test_warmup_must_run_what_the_measured_requests_run(make_sessions, warmup, gaps):
    steps, switch = 8, 3
    sessions, directory = make_sessions("C5", steps=steps, switch=switch)
    _rebuild(sessions, directory, _WU, compiled=(0, 5), **warmup)
    scheduled = sessions["scheduled"]
    later = [request["attention_schedule"] for request in scheduled["requests"][1:]]

    result = _evaluate(sessions, directory)

    # run_session stops after the warm-up request on the same gaps the verdict reports.
    assert bench._warmup_coverage_gaps(scheduled["requests"][_WU], later, steps) == gaps
    timing = result["items"]["R8_timing"]
    if gaps:
        assert (timing["status"], timing["reasons"]) == ("not_proven", ["warmup_coverage"])
        assert result["status"] == "not_success"
    else:
        assert timing["status"] == "pass"
        assert result["status"] == "success", _not_passing(result)


def test_warmup_coverage_is_counted_per_transformer(make_sessions):
    sessions, directory = make_sessions("C1")
    warmup = sessions["scheduled"]["requests"][_WU]

    # MiniMax-H3 has one transformer, so the default warm-up covers both paths.
    assert bench._selection_pairs(warmup["sequences"]) == {("None", "<baseline>"), ("None", _PROFILE)}
    assert bench._warmup_coverage_gaps(warmup, [[], _from(3), _from(4)], 8) == []
    assert bench._warmup_coverage_gaps(warmup, [_from(3, "other")], 8) == ["None:other"]
    # Profiles that share one implementation are recorded under a joined name.
    for step in warmup["sequences"][0]["steps"][1:7]:
        step["attention"]["selection"] = {f"{_PROFILE}|other": _LAYER_CALLS}
    assert bench._warmup_coverage_gaps(warmup, [_from(3, "other")], 8) == []


def test_video_rule_checks_the_artifact_files(make_sessions):
    sessions, directory = make_sessions("C1")
    assert _evaluate(sessions, directory)["items"]["R8_video"]["status"] == "pass"

    without_directory = bench.evaluate_combination(sessions)["items"]["R8_video"]
    assert (without_directory["status"], without_directory["reasons"]) == ("not_proven", ["artifact_missing"])

    (directory / "scheduled-candidate.mp4").write_bytes(b"replaced after the run")
    changed = _evaluate(sessions, directory)["items"]["R8_video"]
    assert (changed["status"], changed["reasons"]) == ("fail", ["artifact_changed"])

    (directory / "scheduled-side_by_side.mp4").unlink()
    missing = _evaluate(sessions, directory)["items"]["R8_video"]
    assert (missing["status"], missing["reasons"]) == ("fail", ["artifact_changed"])
    (directory / "scheduled-candidate.mp4").unlink()
    missing = _evaluate(sessions, directory)["items"]["R8_video"]
    assert (missing["status"], missing["reasons"]) == ("not_proven", ["artifact_missing"])


@pytest.mark.parametrize(
    ("pointer", "value", "status", "reason"),
    [
        pytest.param("/comparison/side_by_side", None, "not_proven", "side_by_side_missing", id="no-side-by-side"),
        pytest.param("/comparison/side_by_side/labels", ["dense", ""], "fail", "labels", id="empty-label"),
        pytest.param("/comparison/side_by_side/frames", _FRAMES - 1, "fail", "frame_count", id="frame-count"),
    ],
)
def test_video_rule_checks_the_side_by_side_entry(make_sessions, pointer, value, status, reason):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    video = result["items"]["R8_video"]
    assert (video["status"], video["reasons"]) == (status, [reason])
    assert result["status"] == "not_success"


@pytest.mark.parametrize(
    ("pointer", "value"),
    [
        pytest.param("/probes/installed", [{"target": "torch:compile", "kind": "compile"}], id="no-attention-probe"),
        pytest.param(f"/requests/{_CA}/sequences/0/steps/2/attention/kernel", _DROP, id="step-without-kernel-counts"),
    ],
)
def test_backend_report_must_be_complete(make_sessions, pointer, value):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    report = result["items"]["R8_backend_report"]
    assert (report["status"], report["reasons"]) == ("not_proven", ["backend_report_incomplete"])
    assert result["report"]["backend"] == {}
    assert result["status"] == "not_success"


def test_backend_report_lists_each_candidate_step(make_sessions):
    steps, switch = 8, 3
    sessions, directory = make_sessions("C6", steps=steps, switch=switch)

    report = _evaluate(sessions, directory)["report"]["backend"]

    assert [entry["step"] for entry in report["per_step"]] == list(range(steps))
    assert [entry["expected"] for entry in report["per_step"]] == [None] * switch + [_PROFILE] * (steps - switch)
    at_switch = report["per_step"][switch]
    assert at_switch["transformer"] == "transformer"
    assert at_switch["selection"] == {_PROFILE: _LAYER_CALLS}
    assert at_switch["backends"] == {"TRTLLM_ATTN": _LAYER_CALLS}
    assert at_switch["kernel"]["skip_factor_min"] == 0.5
    assert report["steps_all_fallback"] == []


@pytest.mark.parametrize(
    ("edits", "reason"),
    [
        pytest.param([("/status", "aborted")], "status", id="aborted-session"),
        pytest.param([("/session", "plain")], "session", id="record-of-another-session"),
        pytest.param([("/schema", "other/1")], "schema", id="other-schema"),
        pytest.param([("/config/approximation", "sparse")], "combination_row", id="combination-row"),
        pytest.param([(f"/requests/{_CA}/error", _REQUEST_ERROR)], "request_error:candidate", id="request-error"),
        pytest.param(
            [(f"/requests/{_RA}/sequences/0/incomplete", True)],
            "sequence_incomplete:reference_a",
            id="incomplete-sequence",
        ),
        pytest.param(
            [(f"/requests/{_RB}/sequences/0/steps/4/step", 5)], "step_indices:reference_b", id="gap-in-step-indices"
        ),
        pytest.param([(f"/requests/{_CA}/sequences", [])], "sequence_count:candidate", id="no-sequence"),
        pytest.param(
            [(f"/requests/{_CA}/sequences/0/source", "h3_step_scheduler")],
            "sequence_source:candidate",
            id="source-of-another-mode",
        ),
        pytest.param(
            [(f"/requests/{_CA}/unattributed/block_calls", 1)],
            "unattributed_block_calls:candidate",
            id="block-call-outside-a-step",
        ),
        pytest.param(
            [(f"/requests/{_RA}/unattributed/attention_layer_calls", 2)],
            "unattributed_attention_layer_calls:reference_a",
            id="attention-call-outside-a-step",
        ),
        pytest.param(
            [(f"/requests/{_CA}/unattributed/kernel_calls", 1)],
            "unattributed_kernel_calls:candidate",
            id="kernel-call-outside-a-step",
        ),
        pytest.param(
            [(f"/requests/{_RB}/name", "candidate"), (f"/requests/{_CA}/name", "reference_b")],
            "request_order",
            id="request-order",
        ),
    ],
)
def test_capture_rule(make_sessions, edits, reason):
    sessions, directory = make_sessions("C1")
    for pointer, value in edits:
        _put(sessions["scheduled"], pointer, value)

    result = _evaluate(sessions, directory)

    capture = result["items"]["capture"]
    assert (capture["status"], capture["reasons"]) == ("not_proven", [reason])
    for name, dependencies in bench.ITEM_DEPENDENCIES.items():
        if name == "capture":
            continue
        item = result["items"][name]
        blocked_by = "depends:capture" if "capture" in dependencies else "depends:conditions"
        assert item["status"] == "not_proven", name
        assert blocked_by in item["reasons"], name
    assert result["status"] == "not_success"
    assert "SUCCESS" not in _summary(result)


def test_capture_rule_checks_the_source_of_the_execution_mode(make_sessions):
    sessions, directory = make_sessions("C3")
    _put(sessions["scheduled"], f"/requests/{_CA}/sequences/0/source", "h3_request_loop")

    capture = _evaluate(sessions, directory)["items"]["capture"]

    assert (capture["status"], capture["reasons"]) == ("not_proven", ["sequence_source:candidate"])


def test_missing_scheduled_session_is_not_measured(make_sessions):
    sessions, directory = make_sessions("C1")
    sessions["scheduled"] = None

    result = _evaluate(sessions, directory)

    assert (result["combination"], result["status"]) == ("C1", "not_measured")
    for name in bench.ITEM_DEPENDENCIES:
        item = result["items"][name]
        assert (item["status"], item["reasons"]) == ("not_measured", ["scheduled_session_absent"]), name
    assert result["items"]["R9_no_schedule"]["status"] == "pass"


@pytest.mark.parametrize("scheduled", [{}, [], {"requests": "x"}, {"schema": bench.SCHEMA, "config": 3}])
def test_incomplete_record_is_not_proven(scheduled):
    result = bench.evaluate_combination({"scheduled": scheduled, "plain": None, "prechange": None})

    assert result["status"] == "not_success"
    capture = result["items"]["capture"]
    assert capture["status"] == "not_proven"
    assert capture["reasons"][0].startswith("missing:scheduled.json#/")
    assert "SUCCESS" not in _summary(result)


def test_wrong_type_in_a_record_is_not_proven(make_sessions):
    sessions, directory = make_sessions("C1")
    _put(sessions["scheduled"], f"/requests/{_RA}/unattributed", None)

    capture = _evaluate(sessions, directory)["items"]["capture"]

    assert (capture["status"], capture["reasons"]) == ("not_proven", ["malformed:AttributeError"])


def test_matrix_requires_all_six(make_sessions):
    results = {combination: _evaluate(*make_sessions(combination)) for combination in bench.COMBINATIONS}
    del results["C4"]

    matrix = bench.evaluate_matrix(results)
    summary = bench.render_summary(results, matrix)

    assert (matrix["status"], matrix["missing"]) == ("not_success", ["C4"])
    assert summary.splitlines()[0] == "R7-R10: NOT PROVEN"
    assert "C4: not_measured (no session records)" in summary
    assert "SUCCESS" not in summary


def test_matrix_rejects_one_unproven_combination(make_sessions):
    results = {combination: _evaluate(*make_sessions(combination)) for combination in bench.COMBINATIONS}
    sessions, directory = make_sessions("C6")
    for step in _steps(sessions, _CA)[3:]:
        _make_fallback(step, "private_gate")
    results["C6"] = _evaluate(sessions, directory)

    matrix = bench.evaluate_matrix(results)
    summary = bench.render_summary(results, matrix)

    assert (matrix["status"], matrix["missing"]) == ("not_success", [])
    assert summary.splitlines()[0] == "R7-R10: NOT PROVEN"
    assert "  R10_target: not_proven [all_fallback]" in summary.splitlines()
    # The summary names reasons. The evidence pointers are in the verdict file.
    assert "scheduled.json#" not in summary
    assert "SUCCESS" not in summary


def test_matrix_requires_one_shifted_boundary_pass(make_sessions):
    results = {}
    for combination in bench.COMBINATIONS:
        sessions, directory = make_sessions(combination)
        del sessions["scheduled"]["requests"][_BS]
        results[combination] = _evaluate(sessions, directory)
        assert results[combination]["status"] == "success", _not_passing(results[combination])

    matrix = bench.evaluate_matrix(results)
    summary = bench.render_summary(results, matrix)

    assert matrix == {
        "status": "not_success",
        "missing": [],
        "ae5": {"status": "not_measured", "combinations": [], "failed": []},
    }
    assert summary.splitlines()[1] == "AE5 shifted boundary: not_measured; passed on none; failed on none"
    assert "SUCCESS" not in summary


def test_matrix_rejects_one_failed_shifted_boundary(make_sessions):
    results = {}
    for combination in bench.COMBINATIONS:
        sessions, directory = make_sessions(combination)
        if combination == "C5":
            # The shifted schedule compiled two more graphs on this combination.
            _put(sessions["scheduled"], f"/requests/{_BS}/graphs_compiled_after", 7)
        results[combination] = _evaluate(sessions, directory)
        assert results[combination]["status"] == "success", _not_passing(results[combination])

    matrix = bench.evaluate_matrix(results)
    summary = bench.render_summary(results, matrix)

    passed = [name for name in bench.COMBINATIONS if name != "C5"]
    assert matrix == {
        "status": "not_success",
        "missing": [],
        "ae5": {"status": "fail", "combinations": passed, "failed": ["C5"]},
    }
    assert summary.splitlines()[:2] == [
        "R7-R10: NOT PROVEN",
        f"AE5 shifted boundary: fail; passed on {', '.join(passed)}; failed on C5",
    ]
    assert "  AE5_boundary: fail [graphs_grew]" in summary.splitlines()
    assert "SUCCESS" not in summary


# ---------------------------------------------------------------------------
# Capture store
# ---------------------------------------------------------------------------


class _Clock:
    """A clock the test sets by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _open_step(recorder: Any, step: int = 0) -> None:
    recorder.begin_request("candidate")
    recorder.begin_sequence("test", None, {})
    recorder.begin_step(step)


def test_recorder_step_and_block_accounting():
    clock = _Clock()
    syncs: list[float] = []
    recorder = bench.Recorder(clock=clock, sync=lambda: syncs.append(clock.now))

    # Before the first request, as during the start of the service: counted outside requests.
    recorder.begin_sequence("startup", 1, {})
    recorder.begin_step(0)
    recorder.block_enter(0)
    recorder.block_exit(0)
    recorder.end_sequence(None)
    assert recorder.probes_snapshot()["outside_requests"] == {
        "sequences": 1,
        "steps": 1,
        "block_calls": 1,
        "eager_block_calls": 1,
        "graph_execs": 0,
        "attention_layer_calls": 0,
        "kernel_calls": 0,
    }
    assert syncs == []

    recorder.begin_request("candidate")
    recorder.begin_sequence("source", 3, {"x": {"sha256": "initial"}})
    clock.now = 10.0
    recorder.begin_step(0)
    assert syncs == [10.0]
    recorder.block_enter(0)
    recorder.note_graph_executed(0)
    recorder.block_exit(0)
    recorder.block_enter(0)
    recorder.block_exit(0)
    recorder.add_digests({"x": {"sha256": "step-0"}}, 0.5)
    clock.now = 13.0
    recorder.end_step()
    assert syncs == [10.0, 13.0]

    # Between two steps: counted on the request.
    recorder.block_enter(0)
    recorder.note_graph_executed(0)
    recorder.block_exit(0)

    # A new step closes the open one, as the Wan adapter relies on.
    clock.now = 20.0
    recorder.begin_step(1, transformer="transformer")
    assert recorder.step_open()
    assert recorder.steps_in_sequence() == 2
    clock.now = 21.0
    recorder.begin_step(2, batch_size=2)
    clock.now = 24.0
    captured = recorder.end_request()

    assert syncs == [10.0, 13.0, 20.0, 21.0, 21.0, 24.0]
    [sequence] = captured["sequences"]
    assert (sequence["source"], sequence["total_steps"], sequence["final"]) == ("source", 3, None)
    assert sequence["incomplete"] is True
    assert sequence["initial"] == {"x": {"sha256": "initial"}}
    first, second, third = sequence["steps"]
    assert first == {
        "step": 0,
        "batch_size": 1,
        "transformer": None,
        "seconds": 2.5,
        "digest_seconds": 0.5,
        "digests": {"x": {"sha256": "step-0"}},
        "compile": {
            "block_calls": 2,
            "eager_block_calls": 1,
            "graph_execs": 1,
            "graphs_compiled": 0,
            "by_model": {"0": {"block_calls": 2, "eager_block_calls": 1}},
        },
        "attention": None,
    }
    assert (second["step"], second["seconds"], second["transformer"]) == (1, 1.0, "transformer")
    assert (third["step"], third["seconds"], third["batch_size"]) == (2, 3.0, 2)
    assert captured["unattributed"] == {
        "block_calls": 1,
        "eager_block_calls": 0,
        "graph_execs": 1,
        "attention_layer_calls": 0,
        "kernel_calls": 0,
    }
    assert not recorder.request_open()


def test_recorder_sequence_totals_and_graph_counts():
    recorder = bench.Recorder(clock=_Clock())
    assert recorder.note_graph_compiled() == 0

    recorder.begin_request("warmup")
    recorder.begin_sequence("source", None, {})
    recorder.begin_step(0)
    assert recorder.note_graph_compiled() == 1
    recorder.end_step()
    recorder.end_sequence({"x": 1})
    recorder.note_compile_error()
    assert recorder.note_setup("Model", 2, ["Block"], {"dynamic": "True"}) == 0
    captured = recorder.end_request()

    assert (captured["graphs_compiled_before"], captured["graphs_compiled_after"]) == (1, 2)
    [sequence] = captured["sequences"]
    assert (sequence["total_steps"], sequence["incomplete"], sequence["final"]) == (1, False, {"x": 1})
    assert sequence["steps"][0]["compile"]["graphs_compiled"] == 1
    assert recorder.totals() == {"setup_calls": 1, "graphs_compiled": 2, "graph_execs": 0, "compile_errors": 1}
    assert recorder.probes_snapshot()["setup_calls"] == [
        {
            "index": 0,
            "model_class": "Model",
            "compiled_blocks": 2,
            "block_class_names": ["Block"],
            "kwargs": {"dynamic": "True"},
            "error": None,
        }
    ]

    with pytest.raises(bench.EvidenceError, match="no request is open"):
        recorder.end_request()
    recorder.begin_request("first")
    with pytest.raises(bench.EvidenceError, match="request first is still open"):
        recorder.begin_request("second")


def test_recorder_is_shared_across_threads():
    recorder = bench.Recorder(clock=_Clock())
    recorder.enable_attention()
    _open_step(recorder)

    def model_thread() -> None:
        recorder.block_enter(0)
        recorder.note_graph_executed(0)
        call = recorder.attention_enter({"selection": _PROFILE, "ctx_step": 0})
        recorder.note_kernel_call(skip_factor=None, sage=True)
        recorder.attention_exit(call)
        recorder.block_exit(0)

    worker = threading.Thread(target=model_thread)
    worker.start()
    worker.join()
    recorder.end_step()
    step = recorder.end_request()["sequences"][0]["steps"][0]

    assert step["compile"]["block_calls"] == 1
    assert step["compile"]["eager_block_calls"] == 0
    assert step["compile"]["graph_execs"] == 1
    assert step["attention"]["layer_calls"] == 1
    assert step["attention"]["outcome"]["approx_quantized"] == 1


_FLAGS_OFF = {"quant": False, "skip_enabled": False, "skip_configured": False}


@pytest.mark.parametrize(
    ("info", "kernel_calls", "counters"),
    [
        pytest.param({"selection": "<baseline>"}, [], {"baseline": 1}, id="baseline"),
        pytest.param({"selection": "<unscheduled>"}, [], {"unscheduled": 1}, id="unscheduled"),
        pytest.param({"selection": "p"}, [(None, True)], {"approx_quantized": 1, "approx_calls": 1}, id="sage"),
        pytest.param({"selection": "p"}, [(0.5, False)], {"approx_sparse": 1, "approx_calls": 1}, id="skip"),
        pytest.param(
            {"selection": "p"},
            [(0.5, True)],
            {"approx_quantized": 1, "approx_sparse": 1, "approx_calls": 1},
            id="sage-and-skip",
        ),
        pytest.param(
            {"selection": "p", **_FLAGS_OFF, "quant": True}, [(None, False)], {"fallback.short_kv": 1}, id="short-kv"
        ),
        pytest.param(
            {"selection": "p", **_FLAGS_OFF, "skip_configured": True},
            [(None, False)],
            {"fallback.ignored_layer": 1},
            id="ignored-layer",
        ),
        pytest.param(
            {"selection": "p", **_FLAGS_OFF, "skip_enabled": True, "skip_configured": True},
            [(None, False)],
            {"fallback.private_gate": 1},
            id="private-gate",
        ),
        pytest.param(
            {"selection": "p", **_FLAGS_OFF},
            [(None, False)],
            {"fallback.profile_not_approximate": 1},
            id="profile-not-approximate",
        ),
        pytest.param(
            {"selection": "p", **_FLAGS_OFF, "quant": True}, [], {"fallback.layer_fallback": 1}, id="no-kernel-call"
        ),
        pytest.param({"selection": "p"}, [(None, False)], {"fallback.unclassified": 1}, id="flags-unknown"),
        # A threshold of 0.0 gives the factor 0.0, which skips nothing.
        pytest.param(
            {"selection": "p", **_FLAGS_OFF, "skip_enabled": True, "skip_configured": True},
            [(0.0, False)],
            {"fallback.private_gate": 1},
            id="skip-factor-zero",
        ),
        pytest.param(
            {"selection": "p", **_FLAGS_OFF, "skip_enabled": True, "skip_configured": True},
            [(float("nan"), False)],
            {"fallback.private_gate": 1},
            id="skip-factor-not-readable",
        ),
        pytest.param({"selection": "<error>"}, [], {}, id="probe-error"),
    ],
)
def test_attention_outcome_classification(info, kernel_calls, counters):
    recorder = bench.Recorder(clock=_Clock())
    recorder.enable_attention()
    _open_step(recorder)

    call = recorder.attention_enter(info)
    for factor, sage in kernel_calls:
        recorder.note_kernel_call(skip_factor=factor, sage=sage)
    recorder.attention_exit(call)
    recorder.end_step()
    attention = recorder.end_request()["sequences"][0]["steps"][0]["attention"]

    outcome = attention["outcome"]
    counted = {name: value for name, value in outcome.items() if name != "fallback" and value}
    counted.update({f"fallback.{name}": value for name, value in outcome["fallback"].items() if value})
    assert counted == counters
    assert attention["layer_calls"] == 1
    assert attention["selection"] == {info["selection"]: 1}
    assert attention["kernel"]["calls"] == len(kernel_calls)
    assert attention["probe_errors"] == (1 if info["selection"] == "<error>" else 0)


def test_attention_step_mismatch_and_kernel_calls_outside_a_layer_call():
    recorder = bench.Recorder(clock=_Clock())
    recorder.enable_attention()
    _open_step(recorder, step=2)

    for ctx_step in (3, 2, None):
        recorder.attention_exit(recorder.attention_enter({"selection": "<baseline>", "ctx_step": ctx_step}))
    recorder.note_kernel_call(skip_factor=0.25, sage=False)
    recorder.end_step()

    # After the step: the events are counted on the request.
    call = recorder.attention_enter({"selection": "<baseline>"})
    recorder.note_kernel_call(skip_factor=None, sage=False)
    recorder.attention_exit(call)
    recorder.note_kernel_call(skip_factor=None, sage=False)
    captured = recorder.end_request()

    attention = captured["sequences"][0]["steps"][0]["attention"]
    assert (attention["layer_calls"], attention["step_mismatch_calls"]) == (3, 1)
    assert attention["kernel"] == {
        "calls": 1,
        "sage_calls": 0,
        "skip_calls": 1,
        "dense_calls": 0,
        "orphan_calls": 1,
        "skip_factor_min": 0.25,
        "skip_factor_max": 0.25,
    }
    assert captured["unattributed"]["attention_layer_calls"] == 1
    assert captured["unattributed"]["kernel_calls"] == 2


def test_attention_exit_closes_its_own_call():
    recorder = bench.Recorder(clock=_Clock())
    recorder.enable_attention()
    _open_step(recorder)

    # Two open layer calls hold equal data. The kernel call after the inner one belongs to the outer one.
    outer = recorder.attention_enter({"selection": _PROFILE})
    inner = recorder.attention_enter({"selection": _PROFILE})
    recorder.attention_exit(inner)
    recorder.note_kernel_call(skip_factor=None, sage=True)
    recorder.attention_exit(outer)
    recorder.end_step()
    attention = recorder.end_request()["sequences"][0]["steps"][0]["attention"]

    assert attention["layer_calls"] == 2
    assert (attention["kernel"]["calls"], attention["kernel"]["sage_calls"]) == (1, 1)
    assert attention["kernel"]["orphan_calls"] == 0
    assert attention["outcome"]["approx_quantized"] == 1
    assert attention["outcome"]["fallback"]["layer_fallback"] == 1


# ---------------------------------------------------------------------------
# Probes on fake modules
# ---------------------------------------------------------------------------


def _fake_module(monkeypatch, name: str, **attributes: Any) -> ModuleType:
    """Register a module that importlib.import_module finds by name for the length of one test."""
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_require_attr_reports_missing_name():
    module = ModuleType("fake_owner_module")

    class Owner:
        present = 1

    assert bench.require_attr(Owner, "present", where="test") == 1
    with pytest.raises(bench.EvidenceError, match="step adapter: fake_owner_module has no attribute missing"):
        bench.require_attr(module, "missing", where="step adapter")
    with pytest.raises(bench.EvidenceError) as raised:
        bench.require_attr(Owner, "missing", where="step adapter")
    assert f"{Owner.__module__}.{Owner.__qualname__} has no attribute missing" in str(raised.value)


def test_patchset_restores():
    module = ModuleType("fake_patch_module")
    setattr(module, "value", "original")

    class Base:
        def inherited(self) -> str:
            return "base"

    class Child(Base):
        def own(self) -> str:
            return "own"

    own_before = vars(Child)["own"]
    removed = []
    patches = bench.PatchSet()

    assert patches.add(module, "value", "patched", where="test", kind="module") == "original"
    assert patches.add(Child, "own", lambda self: "patched own", where="test", kind="class") is own_before
    patches.add(Child, "inherited", lambda self: "patched inherited", where="test", kind="class")
    patches.add_handle(SimpleNamespace(remove=lambda: removed.append("hook")))
    with pytest.raises(bench.EvidenceError, match="has no attribute absent"):
        patches.add(Child, "absent", None, where="test", kind="class")

    assert (module.value, Child().own(), Child().inherited(), Base().inherited()) == (
        "patched",
        "patched own",
        "patched inherited",
        "base",
    )
    assert [entry["kind"] for entry in patches.installed] == ["module", "class", "class"]
    assert patches.installed[0] == {"target": "fake_patch_module:value", "kind": "module"}
    assert "absent" not in vars(Child)

    patches.restore()

    assert module.value == "original"
    assert vars(Child)["own"] is own_before
    assert "inherited" not in vars(Child)
    assert Child().inherited() == "base"
    assert removed == ["hook"]

    with bench.PatchSet() as scoped:
        scoped.add(module, "value", "patched again", where="test", kind="module")
        assert module.value == "patched again"
    assert module.value == "original"


def _h3_loop(*, initial_video_rows, initial_audio_rows, sigmas_video, on_step=None, step_profiler=None):
    """A loop with the parameters and the callback order of minimax_h3_denoise_loop."""
    video, audio = initial_video_rows, initial_audio_rows
    for step in range(len(sigmas_video) - 1):
        with step_profiler(step) if step_profiler is not None else contextlib.nullcontext():
            video, audio = video + 1.0, audio + 2.0
            if on_step is not None:
                on_step(step, video, audio)
    return video, audio


def test_h3_request_adapter(monkeypatch):
    steps = 3
    module = _fake_module(monkeypatch, "_fake_bench_h3_loop", minimax_h3_denoise_loop=_h3_loop)
    recorder = bench.Recorder()
    video, audio = torch.zeros(2, 4), torch.ones(1, 4)
    caller_steps: list[int] = []
    profiled: list[tuple[str, int]] = []

    @contextlib.contextmanager
    def caller_profiler(step):
        profiled.append(("enter", step))
        yield
        profiled.append(("exit", step))

    with bench.PatchSet() as patches:
        bench.install_h3_request_adapter(patches, recorder, module_path=module.__name__)
        assert patches.installed == [{"target": "_fake_bench_h3_loop:minimax_h3_denoise_loop", "kind": "step"}]

        # Outside a request, as in the start of the service, nothing is stored.
        module.minimax_h3_denoise_loop(initial_video_rows=video, initial_audio_rows=audio, sigmas_video=[1.0, 0.0])
        outside = recorder.probes_snapshot()["outside_requests"]
        assert (outside["sequences"], outside["steps"]) == (1, 1)

        recorder.begin_request("candidate")
        result = module.minimax_h3_denoise_loop(
            initial_video_rows=video,
            initial_audio_rows=audio,
            sigmas_video=[1.0, 0.6, 0.3, 0.0],
            on_step=lambda step, video_rows, audio_rows: caller_steps.append(step),
            step_profiler=caller_profiler,
        )
        captured = recorder.end_request()

    assert module.minimax_h3_denoise_loop is _h3_loop
    assert torch.equal(result[0], video + 3.0)
    assert torch.equal(result[1], audio + 6.0)
    assert caller_steps == [0, 1, 2]
    assert profiled == [(event, step) for step in range(steps) for event in ("enter", "exit")]
    [sequence] = captured["sequences"]
    assert (sequence["source"], sequence["total_steps"], sequence["incomplete"]) == ("h3_request_loop", steps, False)
    assert sequence["initial"] == {"video": bench.tensor_digest(video), "audio": bench.tensor_digest(audio)}
    assert sequence["final"] == {"video": bench.tensor_digest(result[0]), "audio": bench.tensor_digest(result[1])}
    assert [step["step"] for step in sequence["steps"]] == [0, 1, 2]
    for index, step in enumerate(sequence["steps"]):
        assert step["digests"] == {
            "video": bench.tensor_digest(video + (index + 1.0)),
            "audio": bench.tensor_digest(audio + 2.0 * (index + 1)),
        }
        assert step["batch_size"] == 1


def test_h3_request_adapter_requires_the_loop_parameters(monkeypatch):
    def minimax_h3_denoise_loop(*, initial_video_rows, initial_audio_rows, sigmas_video, on_step=None):
        return initial_video_rows, initial_audio_rows

    module = _fake_module(monkeypatch, "_fake_bench_h3_old_loop", minimax_h3_denoise_loop=minimax_h3_denoise_loop)
    patches = bench.PatchSet()

    with pytest.raises(bench.EvidenceError, match=re.escape("lacks parameter(s) step_profiler")):
        bench.install_h3_request_adapter(patches, bench.Recorder(), module_path=module.__name__)
    assert patches.installed == []
    assert module.minimax_h3_denoise_loop is minimax_h3_denoise_loop


class _FakeH3StepPipeline:
    """A pipeline with the three step-mode methods the adapter patches."""

    def denoise_step(self, input_batch, states=None):
        return "noise"

    def step_scheduler(self, state, noise_pred=None):
        state.latents = state.latents + 1.0
        state.extra["audio_rows"] = state.extra["audio_rows"] + 2.0
        state.step_index += 1
        return "scheduled"

    def post_decode(self, state):
        return "decoded"


def _h3_state() -> SimpleNamespace:
    return SimpleNamespace(latents=torch.zeros(2, 4), extra={"audio_rows": torch.ones(1, 4)}, step_index=0)


def test_h3_step_adapter(monkeypatch):
    steps = 3
    module = _fake_module(
        monkeypatch, "_fake_bench_h3_step", Pipeline=_FakeH3StepPipeline, _STEP_AUDIO_ROWS="audio_rows"
    )
    originals = dict(vars(_FakeH3StepPipeline))
    recorder = bench.Recorder()
    pipeline = _FakeH3StepPipeline()

    def run_request(name: str) -> dict[str, Any]:
        state = _h3_state()
        recorder.begin_request(name)
        for _ in range(steps):
            assert pipeline.denoise_step(SimpleNamespace(states=[state])) == "noise"
            assert pipeline.step_scheduler(state) == "scheduled"
        assert pipeline.post_decode(state) == "decoded"
        return recorder.end_request()

    with bench.PatchSet() as patches:
        bench.install_h3_step_adapter(patches, recorder, module_path=module.__name__, class_name="Pipeline")
        assert [entry["kind"] for entry in patches.installed] == ["step", "step", "step"]
        first = run_request("reference_a")
        second = run_request("candidate")

        # A state at step 0 opens a new sequence even when post_decode did not close the last one.
        recorder.begin_request("interrupted")
        state = _h3_state()
        pipeline.denoise_step(SimpleNamespace(states=[state]))
        pipeline.step_scheduler(state)
        other = _h3_state()
        pipeline.denoise_step(None, states=[other, _h3_state()])
        pipeline.step_scheduler(other)
        interrupted = recorder.end_request()

    for name in ("denoise_step", "step_scheduler", "post_decode"):
        assert vars(_FakeH3StepPipeline)[name] is originals[name]
    video, audio = torch.zeros(2, 4), torch.ones(1, 4)
    for captured in (first, second):
        [sequence] = captured["sequences"]
        assert (sequence["source"], sequence["total_steps"]) == ("h3_step_scheduler", steps)
        assert (sequence["incomplete"], sequence["final"]) == (False, None)
        assert sequence["initial"] == {"video": bench.tensor_digest(video), "audio": bench.tensor_digest(audio)}
        assert [step["step"] for step in sequence["steps"]] == [0, 1, 2]
        for index, step in enumerate(sequence["steps"]):
            # The digests are taken after step_scheduler updated the state.
            assert step["digests"] == {
                "video": bench.tensor_digest(video + (index + 1.0)),
                "audio": bench.tensor_digest(audio + 2.0 * (index + 1)),
            }
            assert step["batch_size"] == 1
    assert [sequence["incomplete"] for sequence in interrupted["sequences"]] == [True, True]
    assert [len(sequence["steps"]) for sequence in interrupted["sequences"]] == [1, 1]
    assert interrupted["sequences"][1]["steps"][0]["batch_size"] == 2


def test_h3_step_adapter_requires_its_targets(monkeypatch):
    class Partial:
        def denoise_step(self, input_batch):
            return None

        def step_scheduler(self, state):
            return None

    module = _fake_module(monkeypatch, "_fake_bench_h3_partial", Pipeline=Partial, _STEP_AUDIO_ROWS="audio_rows")
    patches = bench.PatchSet()

    with pytest.raises(bench.EvidenceError, match="has no attribute post_decode"):
        bench.install_h3_step_adapter(patches, bench.Recorder(), module_path=module.__name__, class_name="Pipeline")
    assert patches.installed == []

    del module._STEP_AUDIO_ROWS
    with pytest.raises(bench.EvidenceError, match="has no attribute _STEP_AUDIO_ROWS"):
        bench.install_h3_step_adapter(patches, bench.Recorder(), module_path=module.__name__, class_name="Pipeline")


class _FakeCfgMixin:
    """Holds the noise prediction, which Wan22Pipeline also inherits from a mixin."""

    def predict_noise_maybe_with_cfg(self, do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs=None):
        return (positive_kwargs["hidden_states"] * 0.5,)


class _FakeWanPipeline(_FakeCfgMixin):
    def __init__(self) -> None:
        self.transformer = object()
        self.transformer_2 = object()

    def diffuse(self, latents, timesteps, as_tensor=True):
        for index in range(len(timesteps)):
            model = self.transformer if index < 2 else self.transformer_2
            positive = {"hidden_states": latents, "current_model": model}
            latents = latents - self.predict_noise_maybe_with_cfg(False, 1.0, positive_kwargs=positive)[0]
        return latents if as_tensor else SimpleNamespace(latents=latents)


def test_wan_request_adapter(monkeypatch):
    module = _fake_module(monkeypatch, "_fake_bench_wan", Pipeline=_FakeWanPipeline)
    diffuse_before = vars(_FakeWanPipeline)["diffuse"]
    recorder = bench.Recorder()
    pipeline = _FakeWanPipeline()
    latents = torch.full((1, 4), 8.0)

    with bench.PatchSet() as patches:
        bench.install_wan_request_adapter(patches, recorder, module_path=module.__name__, class_name="Pipeline")
        assert [entry["kind"] for entry in patches.installed] == ["step", "step"]
        assert "predict_noise_maybe_with_cfg" in vars(_FakeWanPipeline)

        recorder.begin_request("candidate")
        # A prediction outside diffuse records nothing.
        positive = {"hidden_states": latents, "current_model": pipeline.transformer}
        pipeline.predict_noise_maybe_with_cfg(False, 1.0, positive_kwargs=positive)
        result = pipeline.diffuse(latents, torch.arange(4))
        pipeline.diffuse(latents, torch.arange(2), as_tensor=False)
        captured = recorder.end_request()

    assert "predict_noise_maybe_with_cfg" not in vars(_FakeWanPipeline)
    assert vars(_FakeWanPipeline)["diffuse"] is diffuse_before
    assert torch.equal(result, latents * 0.5**4)
    first, second = captured["sequences"]
    assert (first["source"], first["total_steps"], first["incomplete"]) == ("wan_request_diffuse", 4, False)
    assert first["initial"] == {"latents": bench.tensor_digest(latents)}
    assert first["final"] == {"latents": bench.tensor_digest(result)}
    assert [step["step"] for step in first["steps"]] == [0, 1, 2, 3]
    assert [step["transformer"] for step in first["steps"]] == ["transformer"] * 2 + ["transformer_2"] * 2
    entering = latents
    for step in first["steps"]:
        assert step["digests"] == {
            "latent_in": bench.tensor_digest(entering),
            "noise_pred": bench.tensor_digest(entering * 0.5),
        }
        entering = entering - entering * 0.5
    # A return value that is not a tensor leaves the final state empty.
    assert (second["total_steps"], len(second["steps"]), second["final"]) == (2, 2, None)


def test_wan_request_adapter_requires_the_parameters(monkeypatch):
    class Pipeline(_FakeCfgMixin):
        def diffuse(self, latents):
            return latents

    module = _fake_module(monkeypatch, "_fake_bench_wan_old", Pipeline=Pipeline)
    patches = bench.PatchSet()

    with pytest.raises(bench.EvidenceError, match="lacks parameter"):
        bench.install_wan_request_adapter(patches, bench.Recorder(), module_path=module.__name__, class_name="Pipeline")
    assert patches.installed == []


def _backend(name: str) -> SimpleNamespace:
    return SimpleNamespace(get_name=lambda: name)


class _FakeAttentionLayer:
    """A layer with the attributes the attention probe reads."""

    def __init__(self, *, configured: bool = True) -> None:
        self.attention = SimpleNamespace(name="dense")
        self.attn_backend = _backend("SDPA")
        self._schedule_configured = configured
        self._schedule_candidates: dict[str, Any] = {}
        self.selected: Any = (self.attention, self.attn_backend, None)
        self.effective_calls = 0

    def effective_attention(self):
        self.effective_calls += 1
        if isinstance(self.selected, Exception):
            raise self.selected
        return self.selected

    def _run_local_attention(self, query, key=None):
        return ("ran", query, key)


def _fake_attention_modules(monkeypatch) -> SimpleNamespace:
    """Register a fake attention layer module and a fake forward-context module; return the context."""
    context = SimpleNamespace(denoise_step_idx=0)
    layer_module = _fake_module(monkeypatch, "_fake_bench_attention_layer", Attention=_FakeAttentionLayer)
    context_module = _fake_module(
        monkeypatch,
        "_fake_bench_forward_context",
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: context,
    )
    monkeypatch.setattr(bench, "ATTENTION_MODULE", layer_module.__name__)
    monkeypatch.setattr(bench, "FORWARD_CONTEXT_MODULE", context_module.__name__)
    return context


def test_attention_probe(monkeypatch):
    context = _fake_attention_modules(monkeypatch)
    original = vars(_FakeAttentionLayer)["_run_local_attention"]
    recorder = bench.Recorder()
    approximate = SimpleNamespace(
        quant=SimpleNamespace(enabled=True), skip=SimpleNamespace(enabled=False, configured=False)
    )

    with bench.PatchSet() as patches:
        bench.install_attention_probe(patches, recorder)
        assert [entry["kind"] for entry in patches.installed] == ["attention"]
        _open_step(recorder)

        baseline = _FakeAttentionLayer()
        assert baseline._run_local_attention("q", key="k") == ("ran", "q", "k")

        shared = _FakeAttentionLayer()
        shared._schedule_candidates = {
            "late": SimpleNamespace(impl=approximate),
            "early": SimpleNamespace(impl=approximate),
            "other": SimpleNamespace(impl=object()),
        }
        shared.selected = (approximate, _backend("TRTLLM_ATTN"), None)
        shared._run_local_attention("q")

        unscheduled = _FakeAttentionLayer(configured=False)
        unscheduled._run_local_attention("q")
        assert unscheduled.effective_calls == 0

        failing = _FakeAttentionLayer()
        failing.selected = RuntimeError("no candidate for this step")
        assert failing._run_local_attention("q") == ("ran", "q", None)

        unknown = _FakeAttentionLayer()
        unknown.selected = (object(), _backend("OTHER"), None)
        unknown._run_local_attention("q")

        context.denoise_step_idx = 5
        _FakeAttentionLayer()._run_local_attention("q")

        # While Dynamo traces, the probe only calls the original. This checks the pass-through only.
        traced = _FakeAttentionLayer()
        with monkeypatch.context() as patch:
            patch.setattr(torch.compiler, "is_compiling", lambda: True)
            assert traced._run_local_attention("q") == ("ran", "q", None)
        assert traced.effective_calls == 0

        recorder.end_step()
        captured = recorder.end_request()

    assert vars(_FakeAttentionLayer)["_run_local_attention"] is original
    attention = captured["sequences"][0]["steps"][0]["attention"]
    assert attention["layer_calls"] == 6
    assert attention["selection"] == {
        "<baseline>": 2,
        "early|late": 1,
        "<unscheduled>": 1,
        "<error>": 1,
        "<unknown>": 1,
    }
    assert attention["backends"] == {"SDPA": 3, "TRTLLM_ATTN": 1, "OTHER": 1}
    assert (attention["probe_errors"], attention["step_mismatch_calls"]) == (1, 1)
    assert (attention["outcome"]["baseline"], attention["outcome"]["unscheduled"]) == (2, 1)
    assert attention["outcome"]["fallback"]["layer_fallback"] == 2


def test_attention_probe_requires_effective_attention(monkeypatch):
    class Attention:
        def _run_local_attention(self, query):
            return query

    layer_module = _fake_module(monkeypatch, "_fake_bench_old_layer", Attention=Attention)
    monkeypatch.setattr(bench, "ATTENTION_MODULE", layer_module.__name__)
    _fake_module(monkeypatch, "_fake_bench_context", is_forward_context_available=bool, get_forward_context=dict)
    monkeypatch.setattr(bench, "FORWARD_CONTEXT_MODULE", "_fake_bench_context")
    patches = bench.PatchSet()

    with pytest.raises(bench.EvidenceError, match="has no attribute effective_attention"):
        bench.install_attention_probe(patches, bench.Recorder())
    assert patches.installed == []


def _fake_kernel(query, *, skip_softmax_threshold_scale_factor=None, sage_attn_sfs=None):
    return query


def test_kernel_probe(monkeypatch):
    module = _fake_module(monkeypatch, "_fake_bench_trtllm", **{bench.TRTLLM_KERNEL: _fake_kernel})
    monkeypatch.setattr(bench, "TRTLLM_MODULE", module.__name__)
    recorder = bench.Recorder()
    recorder.enable_attention()

    with bench.PatchSet() as patches:
        info = bench.install_trtllm_kernel_probe(patches, recorder)
        probed = getattr(module, bench.TRTLLM_KERNEL)
        assert probed is not _fake_kernel
        parameters = inspect.signature(probed).parameters
        assert {"sage_attn_sfs", "skip_softmax_threshold_scale_factor"} <= set(parameters)

        _open_step(recorder)
        call = recorder.attention_enter({"selection": _PROFILE})
        assert probed("q", skip_softmax_threshold_scale_factor=0.5) == "q"
        probed("q", sage_attn_sfs=(1.0, 1.0, 1.0, 1.0))
        probed("q")
        # Neither a factor of 0.0 nor a factor the probe cannot read counts as a sparse call.
        probed("q", skip_softmax_threshold_scale_factor=0.0)
        probed("q", skip_softmax_threshold_scale_factor="x")
        with monkeypatch.context() as patch:
            patch.setattr(torch.compiler, "is_compiling", lambda: True)
            assert probed("q", sage_attn_sfs=(1.0, 1.0, 1.0, 1.0)) == "q"
        recorder.attention_exit(call)
        recorder.end_step()
        captured = recorder.end_request()

    assert getattr(module, bench.TRTLLM_KERNEL) is _fake_kernel
    assert info == {
        "target": f"_fake_bench_trtllm:{bench.TRTLLM_KERNEL}",
        "module": _fake_kernel.__module__,
        "qualname": "_fake_kernel",
    }
    # The verdict accepts only a kernel from flashinfer, so this fake would give kernel_not_real.
    assert not info["module"].startswith("flashinfer")
    attention = captured["sequences"][0]["steps"][0]["attention"]
    assert attention["kernel"] == {
        "calls": 5,
        "sage_calls": 1,
        "skip_calls": 1,
        "dense_calls": 3,
        "orphan_calls": 0,
        "skip_factor_min": 0.5,
        "skip_factor_max": 0.5,
    }
    assert attention["outcome"]["approx_quantized"] == 1
    assert attention["outcome"]["approx_sparse"] == 1
    assert attention["outcome"]["approx_calls"] == 1


def test_kernel_probe_requires_the_kernel(monkeypatch):
    module = _fake_module(monkeypatch, "_fake_bench_trtllm_without_kernel")
    monkeypatch.setattr(bench, "TRTLLM_MODULE", module.__name__)

    with pytest.raises(bench.EvidenceError, match=f"has no attribute {bench.TRTLLM_KERNEL}"):
        bench.install_trtllm_kernel_probe(bench.PatchSet(), bench.Recorder())


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, value):
        return self.linear(value)


class _ToyTail(nn.Module):
    def forward(self, value):
        return value


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock(), _ToyBlock()])
        self.tail = _ToyTail()

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return self.tail(value)


def test_block_probe(monkeypatch):
    received = []

    def regionally_compile(model, *args, **kwargs):
        received.append((model, args, kwargs))
        if kwargs.get("fail"):
            raise RuntimeError("compile failed")
        for module in model.modules():
            if not isinstance(module, _ToyBlock):
                continue
            if vars(module).get("_omni_original_forward") is not None:
                # A block with a hook dispatcher keeps its forward. The production helper replaces
                # only the callable the dispatcher calls.
                module._omni_original_forward = module.forward
            else:
                # The production helper stores the compiled callable as an instance attribute.
                setattr(module, "forward", module.forward)
        return model

    runner = _fake_module(monkeypatch, "_fake_bench_runner", regionally_compile=regionally_compile)
    monkeypatch.setattr(bench, "RUNNER_MODULE", runner.__name__)
    recorder = bench.Recorder()
    model, tail, hooked = _ToyModel(), _ToyTail(), _ToyModel()
    for block in hooked.blocks:
        block._omni_original_forward = block.forward

    with bench.PatchSet() as patches:
        bench.install_block_probe(patches, recorder)
        assert runner.regionally_compile(model, "positional", dynamic=False) is model
        assert runner.regionally_compile(tail) is tail
        with pytest.raises(RuntimeError, match="compile failed"):
            runner.regionally_compile(_ToyModel(), fail=True)
        assert runner.regionally_compile(hooked) is hooked
        assert "forward" not in vars(hooked.blocks[0])

        _open_step(recorder)
        model(torch.ones(1, 2))
        recorder.end_step()
        captured = recorder.end_request()
        assert len(model.blocks[0]._forward_hooks) == 1

    assert runner.regionally_compile is regionally_compile
    assert received[0] == (model, ("positional",), {"dynamic": False})
    for block in model.blocks:
        assert not block._forward_hooks
        assert not block._forward_pre_hooks
    setup_calls = recorder.probes_snapshot()["setup_calls"]
    assert setup_calls[0] == {
        "index": 0,
        "model_class": "_ToyModel",
        "compiled_blocks": 2,
        "block_class_names": ["_ToyBlock"],
        "kwargs": {"dynamic": "False"},
        "error": None,
    }
    assert (setup_calls[1]["model_class"], setup_calls[1]["compiled_blocks"]) == ("_ToyTail", 0)
    assert (setup_calls[2]["compiled_blocks"], setup_calls[2]["error"]) == (0, "RuntimeError: compile failed")
    assert setup_calls[3] == {
        "index": 3,
        "model_class": "_ToyModel",
        "compiled_blocks": 2,
        "block_class_names": ["_ToyBlock"],
        "kwargs": {},
        "error": None,
    }
    # No compiled graph ran in the two block calls, so both count as eager.
    assert captured["sequences"][0]["steps"][0]["compile"] == {
        "block_calls": 2,
        "eager_block_calls": 2,
        "graph_execs": 0,
        "graphs_compiled": 0,
        "by_model": {"0": {"block_calls": 2, "eager_block_calls": 2}},
    }


def _fake_probe_targets(monkeypatch) -> ModuleType:
    """Register fake modules for every name install_probes patches outside torch; return the pipeline module."""

    def regionally_compile(model, *args, **kwargs):
        return model

    class Attention:
        def effective_attention(self):
            return None

        def _run_local_attention(self, query):
            return query

    targets = {
        "RUNNER_MODULE": _fake_module(monkeypatch, "_fake_bench_runner", regionally_compile=regionally_compile),
        "ATTENTION_MODULE": _fake_module(monkeypatch, "_fake_bench_layer", Attention=Attention),
        "FORWARD_CONTEXT_MODULE": _fake_module(
            monkeypatch, "_fake_bench_context", is_forward_context_available=bool, get_forward_context=dict
        ),
        "TRTLLM_MODULE": _fake_module(monkeypatch, "_fake_bench_trtllm", **{bench.TRTLLM_KERNEL: _fake_kernel}),
    }
    for constant, module in targets.items():
        monkeypatch.setattr(bench, constant, module.__name__)
    return _fake_module(
        monkeypatch,
        "_fake_bench_h3",
        minimax_h3_denoise_loop=_h3_loop,
        MiniMaxH3Pipeline=_FakeH3StepPipeline,
        _STEP_AUDIO_ROWS="audio_rows",
    )


def test_install_probes_reports_a_module_that_cannot_be_imported(monkeypatch):
    _fake_probe_targets(monkeypatch)
    runner = sys.modules[bench.RUNNER_MODULE]
    compile_before = runner.regionally_compile
    config = bench.validate_config(make_config("C1", adapter={"pipeline_module": "_fake_bench_absent_module"}))

    with pytest.raises(bench.EvidenceError, match="cannot be imported: ModuleNotFoundError"):
        bench.install_probes(config, "plain", bench.Recorder())

    # The probes installed before the failure are removed again.
    assert runner.regionally_compile is compile_before


@pytest.mark.parametrize(
    ("session", "backend", "extra_kinds", "has_kernel"),
    [
        pytest.param("scheduled", "TRTLLM_ATTN", ["attention", "kernel"], True, id="scheduled-with-kernel-probe"),
        pytest.param("scheduled", "FASTVIDEO_VSA", ["attention"], False, id="scheduled-other-backend"),
        pytest.param("plain", "TRTLLM_ATTN", [], False, id="plain"),
        pytest.param("prechange", "TRTLLM_ATTN", [], False, id="prechange"),
    ],
)
def test_install_probes_selects_probes_by_session(monkeypatch, session, backend, extra_kinds, has_kernel):
    pipeline = _fake_probe_targets(monkeypatch)
    raw = make_config("C1", adapter={"pipeline_module": pipeline.__name__})
    raw["approx_profiles"][_PROFILE]["backend"] = backend
    config = bench.validate_config(raw)

    patches, kernel_info = bench.install_probes(config, session, bench.Recorder())
    try:
        kinds = [entry["kind"] for entry in patches.installed]
        assert kinds == ["compile", "block", "step", "step", "step", "step", *extra_kinds]
        assert (kernel_info is not None) == has_kernel
        assert pipeline.minimax_h3_denoise_loop is not _h3_loop
    finally:
        patches.restore()
    assert pipeline.minimax_h3_denoise_loop is _h3_loop


def test_install_probes_fails_before_startup_when_a_name_is_missing(monkeypatch):
    pipeline = _fake_probe_targets(monkeypatch)
    del pipeline.minimax_h3_denoise_loop
    runner = sys.modules[bench.RUNNER_MODULE]
    compile_before = runner.regionally_compile
    wrapper_call = torch._TorchCompileInductorWrapper.__call__
    config = bench.validate_config(make_config("C1", adapter={"pipeline_module": pipeline.__name__}))

    with pytest.raises(
        bench.EvidenceError, match="h3 request adapter: _fake_bench_h3 has no attribute minimax_h3_denoise_loop"
    ):
        bench.install_probes(config, "scheduled", bench.Recorder())

    # The probes installed before the failure are removed again.
    assert runner.regionally_compile is compile_before
    assert torch._TorchCompileInductorWrapper.__call__ is wrapper_call


# ---------------------------------------------------------------------------
# Probes on production code
# ---------------------------------------------------------------------------


def test_real_probe_targets_exist():
    where = "test"
    layer = importlib.import_module(bench.ATTENTION_MODULE)
    attention_cls = bench.require_attr(layer, "Attention", where=where)
    for name in ("_run_local_attention", "effective_attention"):
        bench.require_attr(attention_cls, name, where=where)
    init_source = inspect.getsource(attention_cls.__init__)
    assert "_schedule_candidates" in init_source
    assert "_schedule_configured" in init_source

    context = importlib.import_module(bench.FORWARD_CONTEXT_MODULE)
    for name in ("is_forward_context_available", "get_forward_context"):
        bench.require_attr(context, name, where=where)
    # The attention probe reads this field to compare it with the step the adapter opened.
    assert "denoise_step_idx" in {field.name for field in dataclasses.fields(context.ForwardContext)}

    runner = importlib.import_module(bench.RUNNER_MODULE)
    regionally_compile = bench.require_attr(runner, "regionally_compile", where=where)
    # The block probe finds the compiled blocks through the two slots this helper writes.
    compile_source = inspect.getsource(regionally_compile)
    assert "submod.forward = compiled_forward" in compile_source
    assert "submod._omni_original_forward = compiled_forward" in compile_source

    h3 = importlib.import_module(bench.H3_PIPELINE_MODULE)
    loop = bench.require_attr(h3, "minimax_h3_denoise_loop", where=where)
    assert set(bench.H3_LOOP_PARAMETERS) <= set(inspect.signature(loop).parameters)
    bench.require_attr(h3, "_STEP_AUDIO_ROWS", where=where)
    h3_cls = bench.require_attr(h3, bench.H3_PIPELINE_CLASS, where=where)
    for name in ("denoise_step", "step_scheduler", "post_decode"):
        bench.require_attr(h3_cls, name, where=where)

    wan = importlib.import_module(bench.WAN_PIPELINE_MODULE)
    wan_cls = bench.require_attr(wan, bench.WAN_PIPELINE_CLASS, where=where)
    assert {"latents", "timesteps"} <= set(inspect.signature(wan_cls.diffuse).parameters)
    assert "positive_kwargs" in inspect.signature(wan_cls.predict_noise_maybe_with_cfg).parameters

    wrapper_cls = bench.require_attr(torch, "_TorchCompileInductorWrapper", where=where)
    bench.require_attr(wrapper_cls, "__call__", where=where)

    trtllm = importlib.import_module(bench.TRTLLM_MODULE)
    if trtllm.HAS_FLASHINFER:
        kernel = bench.require_attr(trtllm, bench.TRTLLM_KERNEL, where=where)
        assert kernel.__module__.startswith("flashinfer")


@pytest.mark.parametrize(("combination", "step_patches"), [("C1", 4), ("C5", 2)])
def test_install_probes_on_the_real_modules(combination, step_patches):
    config = bench.validate_config(make_config(combination))
    runner = importlib.import_module(bench.RUNNER_MODULE)
    compile_before = runner.regionally_compile

    patches, kernel_info = bench.install_probes(config, "plain", bench.Recorder())
    try:
        assert [entry["kind"] for entry in patches.installed] == ["compile", "block"] + ["step"] * step_patches
        assert kernel_info is None
        assert runner.regionally_compile is not compile_before
    finally:
        patches.restore()
    assert runner.regionally_compile is compile_before


def test_attention_probe_installs_on_the_real_layer():
    layer = importlib.import_module(bench.ATTENTION_MODULE)
    original = vars(layer.Attention)["_run_local_attention"]

    with bench.PatchSet() as patches:
        bench.install_attention_probe(patches, bench.Recorder())
        assert vars(layer.Attention)["_run_local_attention"] is not original
    assert vars(layer.Attention)["_run_local_attention"] is original


def _inductor_skip_reason() -> str | None:
    """Return why the default torch.compile backend cannot run on this host, or None when it can."""

    def toy(value):
        return torch.sin(value) + 1.0

    try:
        torch.compile(toy)(torch.ones(2))
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else 'no message'}"
    finally:
        torch._dynamo.reset()
    return None


def test_compile_probe_counts_real_inductor():
    reason = _inductor_skip_reason()
    if reason is not None:
        pytest.skip(f"the default torch.compile backend does not run on this host: {reason}")
    wrapper_cls = torch._TorchCompileInductorWrapper
    original = wrapper_cls.__call__
    recorder = bench.Recorder()

    def toy(value):
        return torch.sin(value) * 2.0 + 1.0

    try:
        with bench.PatchSet() as patches:
            bench.install_compile_probe(patches, recorder)
            assert patches.installed == [{"target": "torch._TorchCompileInductorWrapper:__call__", "kind": "compile"}]
            compiled = torch.compile(toy)
            sample = torch.ones(4)
            recorder.begin_request("probe")
            results = [compiled(sample) for _ in range(3)]
            captured = recorder.end_request()
    finally:
        torch._dynamo.reset()

    assert wrapper_cls.__call__ is original
    for result in results:
        assert torch.allclose(result, toy(sample))
    totals = recorder.totals()
    assert totals["graphs_compiled"] >= 1
    assert totals["compile_errors"] == 0
    assert captured["unattributed"]["graph_execs"] == 3
    assert captured["graphs_compiled_after"] - captured["graphs_compiled_before"] == totals["graphs_compiled"]


def test_preflight_self_tests_run_on_cpu():
    reason = _inductor_skip_reason()
    if reason is not None:
        pytest.skip(f"the default torch.compile backend does not run on this host: {reason}")

    try:
        compile_detail = bench._compile_self_test("cpu")
        block_detail = bench._block_self_test("cpu")
    finally:
        torch._dynamo.reset()

    # The block self test runs the production regionally_compile on a toy model with two blocks.
    assert compile_detail.endswith("3 executions")
    assert block_detail == "2 compiled blocks, 6 block calls, 0 eager"


# ---------------------------------------------------------------------------
# run_session against a fake service
# ---------------------------------------------------------------------------


class _FakeOmni:
    """Stands in for the object Omni(...) returns."""

    def __init__(self, service: _FakeService, omni_kwargs: dict[str, Any]) -> None:
        self.service = service
        self.omni_kwargs = omni_kwargs
        self.closed = False
        self.num_stages = service.num_stages
        od_config = SimpleNamespace(enforce_eager=False, step_execution=False, num_gpus=1)
        self.engine = SimpleNamespace(stage_clients=[SimpleNamespace(stage_type="diffusion", od_config=od_config)])
        for compiled_blocks, error in service.setup_calls:
            names = ["FakeBlock"] if compiled_blocks else []
            service.recorder.note_setup("FakeTransformer", compiled_blocks, names, {"dynamic": "True"}, error)

    def generate(self, prompt: dict[str, Any], params: Any) -> list[Any]:
        return self.service.generate(prompt, params)

    def close(self) -> None:
        self.closed = True


class _FakeService:
    """Replaces what run_session uses from outside the harness: the service, the probes and the file writers.

    generate() sends the capture store the events the probes would send for one denoise sequence. A step
    that selects a profile adds 2.0 to the state and a dense step adds 1.0, so the digests and the frames
    of a scheduled request differ from the all-dense reference from the first switch on.

    setup_calls holds (compiled blocks, error) of each regionally_compile call the service start records.
    transformers, when set, names the transformer of each step.
    """

    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        self.config = make_config("C1", steps=4, switch=2)
        self.out_dir = tmp_path / "evidence"
        self.repository = tmp_path / "repository"
        self.repository.mkdir()
        self.setup_calls: list[tuple[int, str | None]] = [(2, None)]
        self.num_stages = 1
        self.transformers: list[str] | None = None
        self.emit_steps = True
        self.fail_on_call: int | None = None
        self.frames_by_call: dict[int, int] = {}
        self.self_test_error: str | None = None
        self.lpips_error: Exception | None = None
        self.recorder: Any = None
        self.session: str | None = None
        self.generate_calls = 0
        self.install_calls: list[str] = []
        self.schedules: list[Any] = []
        self.prompts: list[dict[str, Any]] = []
        self.omnis: list[_FakeOmni] = []
        # Each call with the device it was given: the run path names none, so the scoring function chooses it.
        self.self_test_calls: list[tuple[str, str | None]] = []
        self.lpips_calls: list[tuple[tuple[int, ...], str | None]] = []
        ticks = itertools.count(1)
        recorder_cls = bench.Recorder

        # Every clock read advances by one second, so each captured step lasts one second.
        monkeypatch.setattr(bench, "Recorder", lambda sync=None: recorder_cls(clock=lambda: float(next(ticks))))
        monkeypatch.setattr(bench, "_repository_root", lambda: self.repository)
        monkeypatch.setattr(bench, "_comparison_self_test", self._comparison_self_test)
        monkeypatch.setattr(bench, "install_probes", self._install_probes)
        monkeypatch.setattr(bench, "_create_omni", self._create_omni)
        monkeypatch.setattr(bench, "_build_sampling_params", self._build_sampling_params)
        monkeypatch.setattr(bench, "collect_source", self._collect_source)
        monkeypatch.setattr(bench, "collect_environment", lambda: {"python": "test"})
        monkeypatch.setattr(bench, "lpips_metadata", lambda net: {"package_version": "0.1.4", "weights_sha256": "w"})
        monkeypatch.setattr(bench, "lpips_video", self._lpips_video)
        monkeypatch.setattr(bench, "write_video", self._write_video)

    def run(self, session: str, **kwargs: Any) -> int:
        self.session = session
        self.generate_calls = 0
        return bench.run_session(self.config, session=session, out_dir=self.out_dir, argv=["run"], **kwargs)

    def record(self, session: str) -> dict[str, Any]:
        return json.loads((self.out_dir / bench.SESSION_FILES[session]).read_text(encoding="utf-8"))

    def _comparison_self_test(self, net: str, device: str | None = None) -> str:
        self.self_test_calls.append((net, device))
        if self.self_test_error is not None:
            raise bench.EvidenceError(self.self_test_error)
        return "compared"

    def _install_probes(self, config: dict[str, Any], session: str, recorder: Any) -> tuple[Any, Any]:
        self.install_calls.append(session)
        self.recorder = recorder
        patches = bench.PatchSet()
        patches.installed.append({"target": "fake:compile", "kind": "compile"})
        if session != "scheduled":
            return patches, None
        recorder.enable_attention()
        patches.installed.append({"target": "fake:attention", "kind": "attention"})
        return patches, {"target": "fake:kernel", "module": "flashinfer.prefill", "qualname": bench.TRTLLM_KERNEL}

    def _create_omni(self, omni_kwargs: dict[str, Any]) -> _FakeOmni:
        omni = _FakeOmni(self, omni_kwargs)
        self.omnis.append(omni)
        return omni

    def _build_sampling_params(self, sampling_params: dict[str, Any], schedule: Any) -> SimpleNamespace:
        self.schedules.append(schedule)
        return SimpleNamespace(attention_schedule=schedule, **sampling_params)

    def _collect_source(self) -> dict[str, Any]:
        return {
            "harness_sha256": "harness-sha256",
            "package_file": "/src/vllm_omni/__init__.py",
            "package_tree_sha256": _TREE_BEFORE if self.session == "prechange" else _TREE_AFTER,
            "git_head": "head-before",
            "git_diff_sha256": None,
            "git_status_clean": self.session == "prechange",
        }

    def _lpips_video(
        self, reference: Any, candidate: Any, *, net: str = "alex", loss_fn: Any = None, device: str | None = None
    ) -> dict:
        self.lpips_calls.append((tuple(reference.shape), device))
        if self.lpips_error is not None:
            raise self.lpips_error
        frames = int(reference.shape[0])
        return {"value": 0.25, "per_frame": [0.25] * frames, "frames": frames, "net": net, "device": "cpu"}

    def _write_video(self, frames: Any, path: Any, fps: float) -> None:
        Path(path).write_bytes(np.asarray(frames).tobytes())

    def generate(self, prompt: dict[str, Any], params: Any) -> list[Any]:
        index = self.generate_calls
        self.generate_calls += 1
        self.prompts.append(prompt)
        if index == self.fail_on_call:
            raise RuntimeError("generation failed")
        value = torch.zeros(1, 4)
        if self.emit_steps:
            value = self._denoise(params.attention_schedule or [])
        frames = self.frames_by_call.get(index, _FRAMES)
        # Numpy frames with a batch axis: [1, frames, height, width, channels], values in [0, 1].
        video = np.full((1, frames, 4, 4, 3), float(value.mean()) / 100.0, dtype=np.float32)
        return [SimpleNamespace(images=[video], error=None)]

    def _denoise(self, schedule: list[dict[str, Any]]) -> torch.Tensor:
        recorder, config = self.recorder, self.config
        total = config["expected_total_steps"]
        sparse = config["approximation"] == "sparse"
        source = bench.SEQUENCE_SOURCES[(config["model_family"], config["execution_mode"])]
        value = torch.zeros(1, 4)

        def digests() -> dict[str, Any]:
            return {"video": bench.tensor_digest(value), "audio": bench.tensor_digest(value + 0.5)}

        recorder.begin_sequence(source, total, digests())
        for step, profile in enumerate(bench.expected_profiles(schedule, total)):
            recorder.begin_step(step, transformer=self.transformers[step] if self.transformers else None)
            if recorder.totals()["graphs_compiled"] == 0:
                recorder.note_graph_compiled()
            recorder.block_enter(0)
            recorder.note_graph_executed(0)
            if self.session == "scheduled":
                info = {"selection": profile or "<baseline>", "backend": "TRTLLM_ATTN" if profile else "SDPA"}
                call = recorder.attention_enter({**info, "ctx_step": step})
                if profile is not None:
                    recorder.note_kernel_call(skip_factor=0.5 if sparse else None, sage=not sparse)
                recorder.attention_exit(call)
            recorder.block_exit(0)
            value = value + (2.0 if profile else 1.0)
            recorder.add_digests(digests(), 0.0)
            recorder.end_step()
        recorder.end_sequence(digests())
        return value


@pytest.fixture
def fake_service(monkeypatch, tmp_path):
    return _FakeService(monkeypatch, tmp_path)


def test_run_session_with_fake_omni(fake_service, capsys):
    out_dir = fake_service.out_dir

    assert fake_service.run("prechange") == 0
    assert fake_service.run("plain", prior=out_dir / "prechange.json") == 0
    assert fake_service.run("scheduled") == 0

    stdout = capsys.readouterr().out
    lines = [json.loads(line) for line in stdout.splitlines()]
    assert [line["exit_code"] for line in lines] == [0, 0, 0]
    assert lines[2]["session_file"] == str((out_dir / "scheduled.json").resolve())
    # A run states no result; only the verdict command does.
    assert "SUCCESS" not in stdout

    record = fake_service.record("scheduled")
    config = bench.validate_config(fake_service.config)
    assert (record["status"], record["exit_code"], record["abort"]) == ("completed", 0, None)
    assert [request["name"] for request in record["requests"]] == [*bench.SCHEDULED_REQUESTS, "boundary_shift"]
    assert record["requests"][_RA]["attention_schedule"] == []
    assert record["requests"][_RB]["attention_schedule"] == []
    assert record["requests"][_CA]["attention_schedule"] == config["candidate_schedule"]
    assert record["requests"][_WU]["output"] is None
    assert record["startup"]["omni_kwargs"]["diffusion_attention_schedule"] == config["schedule_config"]
    assert record["startup"]["omni_kwargs"]["enforce_eager"] is False
    assert record["startup"]["omni_kwargs"]["distributed_executor_backend"] == "uni"
    assert record["startup"]["effective"]["enforce_eager"] is False
    assert record["probes"]["kernel"]["module"] == "flashinfer.prefill"
    assert record["comparison"]["lpips"]["value"] == 0.25
    assert record["comparison"]["lpips"]["device"] == "cpu"
    assert record["comparison"]["side_by_side"]["file"] == "scheduled-side_by_side.mp4"
    assert record["comparison"]["side_by_side"]["frames"] == _FRAMES
    assert fake_service.lpips_calls == [((_FRAMES, 4, 4, 3), None)]
    # Only the scheduled session compares outputs, so only it checks the comparison tools.
    assert fake_service.self_test_calls == [("alex", None)]
    assert all(omni.closed for omni in fake_service.omnis)
    assert len(fake_service.omnis) == 3

    plain = fake_service.record("plain")
    prechange = fake_service.record("prechange")
    assert plain["comparison_to_prior"]["error"] is None
    assert plain["comparison_to_prior"]["bit_identical"] is True
    assert plain["comparison_to_prior"]["prior_session_id"] == prechange["session_id"]
    assert plain["comparison_to_prior"]["prior_frames_sha256"] == prechange["requests"][1]["output"]["frames_sha256"]

    sessions, inputs = bench._load_sessions(out_dir)
    result = bench.evaluate_combination(sessions, directory=out_dir)
    assert len(inputs) == 3
    assert result["status"] == "success", _not_passing(result)
    assert result["items"]["AE5_boundary"]["status"] == "pass"
    assert result["report"]["reference_vs_plain"] == {"bit_identical": True}
    assert result["report"]["seconds_per_step"]["candidate"]["samples"] == [1.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("session", ["plain", "prechange"])
def test_run_session_without_schedule_sends_no_schedule(fake_service, session):
    assert fake_service.run(session) == 0

    record = fake_service.record(session)
    assert fake_service.install_calls == [session]
    assert fake_service.self_test_calls == []
    assert fake_service.schedules == [None, None]
    assert fake_service.prompts == [fake_service.config["prompt"]] * 2
    assert [request["name"] for request in record["requests"]] == ["warmup", "plain"]
    assert [request["attention_schedule"] for request in record["requests"]] == [None, None]
    assert "diffusion_attention_schedule" not in record["startup"]["omni_kwargs"]
    assert "diffusion_attention_schedule" not in fake_service.omnis[0].omni_kwargs
    assert record["probes"]["kernel"] is None
    assert (record["comparison"], record["comparison_to_prior"]) == (None, None)
    assert all(step["attention"] is None for step in record["requests"][1]["sequences"][0]["steps"])


def test_run_session_checks_the_comparison_tools_before_the_service_starts(fake_service):
    fake_service.self_test_error = "the video writer wrote no file"

    assert fake_service.run("scheduled") == 3

    record = fake_service.record("scheduled")
    assert (record["status"], record["exit_code"]) == ("aborted", 3)
    assert record["abort"] == {
        "stage": "comparison_self_test",
        "error_type": "EvidenceError",
        "message": "the video writer wrote no file",
    }
    assert fake_service.self_test_calls == [("alex", None)]
    assert fake_service.install_calls == []
    assert fake_service.omnis == []
    assert record["probes"]["installed"] == []
    assert record["requests"] == []


def test_run_session_stops_when_no_compile_call_is_seen(fake_service):
    fake_service.setup_calls = []

    assert fake_service.run("scheduled") == 3

    record = fake_service.record("scheduled")
    assert (record["status"], record["exit_code"]) == ("aborted", 3)
    assert (record["abort"]["stage"], record["abort"]["error_type"]) == ("reach_after_startup", "EvidenceError")
    assert "regionally_compile was not called in this process" in record["abort"]["message"]
    assert record["requests"] == []
    assert fake_service.generate_calls == 0
    assert fake_service.omnis[0].closed


@pytest.mark.parametrize(
    ("setup_calls", "reason"),
    [
        pytest.param([(2, None), (0, None)], "no block was compiled", id="no-block-compiled"),
        pytest.param([(0, "RuntimeError: backend failed")], "RuntimeError: backend failed", id="compile-raised"),
    ],
)
def test_run_session_stops_when_a_model_is_not_compiled(fake_service, setup_calls, reason):
    fake_service.setup_calls = setup_calls

    assert fake_service.run("scheduled") == 3

    record = fake_service.record("scheduled")
    assert record["abort"] == {
        "stage": "reach_after_startup",
        "error_type": "EvidenceError",
        "message": f"regionally_compile did not compile FakeTransformer: {reason}",
    }
    assert len(record["probes"]["compile"]["setup_calls"]) == len(setup_calls)
    assert record["requests"] == []
    assert fake_service.generate_calls == 0
    assert fake_service.omnis[0].closed


@pytest.mark.parametrize("session", ["plain", "scheduled"])
def test_run_session_stops_when_the_service_has_several_stages(fake_service, session):
    fake_service.num_stages = 2

    assert fake_service.run(session) == 3

    record = fake_service.record(session)
    assert record["abort"] == {
        "stage": "reach_after_startup",
        "error_type": "EvidenceError",
        "message": "the service has 2 stages: the harness supports one stage",
    }
    assert record["requests"] == []
    assert fake_service.generate_calls == 0
    assert fake_service.omnis[0].closed


def test_run_session_stops_when_the_warmup_reaches_no_step(fake_service):
    fake_service.emit_steps = False

    assert fake_service.run("scheduled") == 3

    record = fake_service.record("scheduled")
    assert (record["status"], record["abort"]["stage"]) == ("aborted", "reach_after_warmup")
    for problem in ("no denoise step was captured", "no graph was compiled", "no kernel call was captured"):
        assert problem in record["abort"]["message"]
    assert [request["name"] for request in record["requests"]] == ["warmup"]
    assert fake_service.generate_calls == 1
    assert fake_service.omnis[0].closed


def test_run_session_stops_when_the_warmup_does_not_cover_the_later_requests(fake_service):
    # Two transformers run two steps each. This warm-up runs the profile only on the first one and a
    # dense step only on the second one. The later requests run a dense step on the first one and the
    # profile on the second one.
    fake_service.transformers = ["transformer", "transformer", "transformer_2", "transformer_2"]
    fake_service.config["warmup_schedule"] = [{"start": 0, "end": 2, "profile": _PROFILE}]

    assert fake_service.run("scheduled") == 3

    record = fake_service.record("scheduled")
    assert (record["status"], record["abort"]["stage"]) == ("aborted", "reach_after_warmup")
    assert record["abort"]["message"] == (
        "the warm-up request did not run every selection of the later requests on the transformer "
        f"that runs it there: transformer:<baseline>, transformer_2:{_PROFILE}; change warmup_schedule"
    )
    assert [request["name"] for request in record["requests"]] == ["warmup"]
    assert fake_service.generate_calls == 1
    assert fake_service.omnis[0].closed


def test_run_session_accepts_a_warmup_that_covers_two_transformers(fake_service):
    fake_service.transformers = ["transformer", "transformer", "transformer_2", "transformer_2"]

    assert fake_service.run("scheduled") == 0

    record = fake_service.record("scheduled")
    assert (record["status"], record["abort"]) == ("completed", None)
    assert [step["transformer"] for step in record["requests"][_CA]["sequences"][0]["steps"]] == [
        "transformer",
        "transformer",
        "transformer_2",
        "transformer_2",
    ]


def test_run_session_keeps_the_record_when_lpips_fails(fake_service):
    fake_service.lpips_error = RuntimeError("CUDA out of memory")

    assert fake_service.run("scheduled") == 0

    record = fake_service.record("scheduled")
    assert (record["status"], record["exit_code"], record["abort"]) == ("completed", 0, None)
    assert record["comparison"]["lpips"]["value"] is None
    assert record["comparison"]["lpips"]["error"] == "lpips_failed: RuntimeError: CUDA out of memory"
    assert record["comparison"]["lpips"]["package_version"] == "0.1.4"
    # The side-by-side video does not depend on the score.
    assert record["comparison"]["side_by_side"]["file"] == "scheduled-side_by_side.mp4"
    assert (fake_service.out_dir / "scheduled-side_by_side.mp4").is_file()

    # Only the scheduled record is given, so the no-schedule item is not measured either.
    result = bench.evaluate_combination({"scheduled": record}, directory=fake_service.out_dir)
    assert _not_passing(result) == {
        "R8_lpips": ("not_measured", ["lpips_failed"]),
        "R9_no_schedule": ("not_measured", ["plain_or_prechange_session_absent"]),
    }


def test_run_session_keeps_record_on_request_error(fake_service):
    fake_service.fail_on_call = _CA

    assert fake_service.run("scheduled") == 4

    record = fake_service.record("scheduled")
    error = {"error_type": "RuntimeError", "message": "generation failed"}
    assert (record["status"], record["exit_code"]) == ("aborted", 4)
    assert record["abort"] == {"stage": "request", **error}
    assert [request["name"] for request in record["requests"]] == list(bench.SCHEDULED_REQUESTS)
    assert record["requests"][_CA]["error"] == error
    assert record["requests"][_CA]["output"] is None
    assert record["requests"][_RB]["output"] is not None
    assert record["comparison"] is None
    assert fake_service.omnis[0].closed

    result = bench.evaluate_combination({"scheduled": record}, directory=fake_service.out_dir)
    assert result["items"]["capture"]["status"] == "not_proven"
    assert "status" in result["items"]["capture"]["reasons"]


def test_run_session_refuses_overwrite_and_repository_paths(fake_service, capsys):
    out_dir = fake_service.out_dir
    assert fake_service.run("plain") == 0
    first = (out_dir / "plain.json").read_bytes()

    assert fake_service.run("plain") == 2
    assert (out_dir / "plain.json").read_bytes() == first
    assert len(fake_service.omnis) == 1

    inside = fake_service.repository / "evidence"
    assert bench.run_session(fake_service.config, session="plain", out_dir=inside) == 2
    assert not inside.exists()

    assert fake_service.run("scheduled", prior=out_dir / "plain.json") == 2
    assert not (out_dir / "scheduled.json").exists()

    errors = capsys.readouterr().err
    for message in ("records of earlier runs are kept", "is inside the repository", "--prior is accepted only"):
        assert message in errors


def test_run_session_reports_an_evidence_directory_that_cannot_be_created(fake_service, capsys):
    fake_service.out_dir.write_text("a file, not a directory", encoding="utf-8")

    assert fake_service.run("plain") == 2

    assert "cannot be created" in capsys.readouterr().err
    assert fake_service.omnis == []


def test_run_session_reports_a_record_that_cannot_be_written(fake_service, monkeypatch, capsys):
    write_json = bench._write_json

    def failing_write(path, data):
        if Path(path).name == bench.SESSION_FILES["plain"]:
            raise OSError("no space left on device")
        write_json(path, data)

    monkeypatch.setattr(bench, "_write_json", failing_write)

    # Every request ran, but a session without its record shows nothing.
    assert fake_service.run("plain") == 4

    captured = capsys.readouterr()
    assert "was not written: no space left on device" in captured.err
    assert captured.out == ""
    assert not (fake_service.out_dir / "plain.json").exists()


def test_run_session_refuses_a_missing_prior_record(fake_service, capsys):
    absent = fake_service.out_dir / "prechange.json"

    assert fake_service.run("plain", prior=absent) == 2

    assert f"--prior {absent} is not a file" in capsys.readouterr().err
    assert not fake_service.out_dir.exists()
    assert fake_service.omnis == []


def test_frame_problem_skips_lpips(fake_service):
    fake_service.frames_by_call = {_CA: _FRAMES - 1}

    assert fake_service.run("scheduled") == 0

    record = fake_service.record("scheduled")
    assert record["comparison"]["lpips"]["value"] is None
    assert record["comparison"]["lpips"]["error"] == "missing_frames"
    assert record["comparison"]["side_by_side"] is None
    assert fake_service.lpips_calls == []
    assert not (fake_service.out_dir / "scheduled-side_by_side.mp4").exists()

    result = bench.evaluate_combination({"scheduled": record}, directory=fake_service.out_dir)
    assert result["items"]["conditions"]["reasons"] == ["output_shape_differs"]
    assert result["items"]["R8_lpips"]["reasons"] == ["depends:conditions"]


def test_comparison_self_test(monkeypatch):
    metadata: dict[str, Any] = {"package_version": "0.1.4", "weights_sha256": "w"}
    scored = []

    def lpips_video(reference, candidate, *, net="alex", loss_fn=None, device=None):
        scored.append((reference.shape, str(reference.dtype), net, device))
        return {"value": 0.0, "per_frame": [0.0, 0.0], "frames": 2, "net": net, "device": device or "cuda"}

    def failing_writer(frames, path, fps):
        raise OSError("no encoder")

    monkeypatch.setattr(bench, "lpips_metadata", lambda net: dict(metadata))
    monkeypatch.setattr(bench, "lpips_video", lpips_video)
    monkeypatch.setattr(bench, "write_video", lambda frames, path, fps: Path(path).write_bytes(b"video"))

    # Without a device, lpips_video chooses it; the text names the device that gave the score.
    assert bench._comparison_self_test("vgg") == "lpips 0.1.4 (vgg) on cuda; side-by-side video written"
    assert bench._comparison_self_test("vgg", "cpu") == "lpips 0.1.4 (vgg) on cpu; side-by-side video written"
    assert scored == [((2, 64, 64, 3), "uint8", "vgg", None), ((2, 64, 64, 3), "uint8", "vgg", "cpu")]

    monkeypatch.setattr(bench, "write_video", lambda frames, path, fps: Path(path).write_bytes(b""))
    with pytest.raises(bench.EvidenceError, match="the video writer wrote no file"):
        bench._comparison_self_test("alex")

    monkeypatch.setattr(bench, "write_video", lambda frames, path, fps: None)
    with pytest.raises(bench.EvidenceError, match="the video writer wrote no file"):
        bench._comparison_self_test("alex")

    monkeypatch.setattr(bench, "write_video", failing_writer)
    with pytest.raises(bench.EvidenceError, match="the comparison tools failed: OSError: no encoder"):
        bench._comparison_self_test("alex")

    metadata["weights_sha256"] = None
    with pytest.raises(bench.EvidenceError, match="lpips metadata is missing: weights_sha256"):
        bench._comparison_self_test("alex")
    assert len(scored) == 5


def test_build_sampling_params_sends_the_schedule_only_when_given():
    sampling = {"seed": 7, "num_inference_steps": 4}
    schedule = [{"start": 2, "end": None, "profile": _PROFILE}]

    without = bench._build_sampling_params(sampling, None)
    empty = bench._build_sampling_params(sampling, [])
    scheduled = bench._build_sampling_params(sampling, schedule)

    # None leaves the field unset, so the service default applies. [] states an all-dense request.
    assert without.attention_schedule is None
    assert empty.attention_schedule == ()
    assert scheduled.attention_schedule == parse_attention_schedule(schedule)
    assert len(scheduled.attention_schedule) == 1
    for params in (without, empty, scheduled):
        assert (params.seed, params.num_inference_steps) == (7, 4)
    assert sampling == {"seed": 7, "num_inference_steps": 4}
    assert schedule == [{"start": 2, "end": None, "profile": _PROFILE}]


def test_read_effective_config_never_raises():
    class Broken:
        @property
        def engine(self):
            raise RuntimeError("engine is gone")

    without_od_config = SimpleNamespace(engine=SimpleNamespace(stage_clients=[SimpleNamespace(stage_type="diffusion")]))

    assert bench.read_effective_config(object()) == (
        None,
        "no diffusion stage client was found on omni.engine.stage_clients",
    )
    assert bench.read_effective_config(Broken()) == (None, "RuntimeError: engine is gone")
    assert bench.read_effective_config(without_od_config) == (None, "the diffusion stage client has no od_config")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _write_sessions(directory: Path, sessions: dict[str, Any]) -> None:
    for name, record in sessions.items():
        (directory / bench.SESSION_FILES[name]).write_text(json.dumps(record), encoding="utf-8")


def test_verdict_cli(make_sessions, tmp_path, capsys):
    directories = []
    for combination in bench.COMBINATIONS:
        sessions, directory = make_sessions(combination)
        _write_sessions(directory, sessions)
        directories.append(str(directory))
    out_file = tmp_path / "verdict.json"

    assert bench.main(["verdict", "--out", str(out_file), *directories]) == 0

    stdout = capsys.readouterr().out
    verdict = json.loads(out_file.read_text(encoding="utf-8"))
    assert stdout.startswith("R7-R10: SUCCESS\n")
    assert verdict["summary_text"] == stdout
    assert (verdict["kind"], verdict["matrix"]["status"]) == ("verdict", "success")
    assert sorted(verdict["combinations"]) == list(bench.COMBINATIONS)
    assert len(verdict["inputs"]) == 3 * len(bench.COMBINATIONS)
    for entry in verdict["inputs"]:
        assert entry["sha256"] == bench.sha256_file(entry["file"])

    sessions, directory = make_sessions("C2")
    for step in _steps(sessions, _CA)[3:]:
        _make_fallback(step, "short_kv")
    _write_sessions(directory, sessions)
    second_file = tmp_path / "verdict-2.json"

    assert bench.main(["verdict", "--out", str(second_file), *directories]) == 1

    stdout = capsys.readouterr().out
    assert stdout.startswith("R7-R10: NOT PROVEN\n")
    assert "  R10_target: not_proven [all_fallback]" in stdout
    assert "SUCCESS" not in stdout
    assert json.loads(second_file.read_text(encoding="utf-8"))["matrix"]["status"] == "not_success"

    assert bench.main(["verdict", "--out", str(tmp_path / "verdict-3.json"), directories[0], directories[0]]) == 2
    assert "combination C1 was given twice" in capsys.readouterr().err
    assert not (tmp_path / "verdict-3.json").exists()


def test_main_usage_errors(tmp_path, capsys):
    out_file = tmp_path / "verdict.json"
    empty = tmp_path / "empty"
    empty.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config("C1")), encoding="utf-8")
    run = ["run", "--out", str(tmp_path / "evidence")]

    assert bench.main([]) == 2
    assert bench.main(["run", "--config", str(config_path)]) == 2
    assert bench.main(["verdict", "--out", str(out_file), str(tmp_path / "absent")]) == 2
    assert bench.main(["verdict", "--out", str(out_file), str(empty)]) == 2
    assert bench.main([*run, "--config", str(tmp_path / "absent.json"), "--session", "plain"]) == 2
    assert bench.main([*run, "--config", str(config_path), "--session", "scheduled", "--prior", "p.json"]) == 2

    errors = capsys.readouterr().err
    for message in ("is not a directory", "holds no session record", "cannot read config", "--prior is accepted only"):
        assert message in errors
    assert not out_file.exists()
    assert not (tmp_path / "evidence").exists()


def test_preflight_reports_missing_target(monkeypatch, tmp_path, capsys):
    def missing_target(config, session, recorder):
        raise bench.EvidenceError("attention probe: layer.Attention has no attribute effective_attention")

    monkeypatch.setattr(bench, "install_probes", missing_target)
    monkeypatch.setattr(bench, "_comparison_self_test", lambda net, device: f"compared with {net} on {device}")
    monkeypatch.setattr(bench, "_compile_self_test", lambda device: "1 graph(s), 3 executions")
    monkeypatch.setattr(bench, "_block_self_test", lambda device: "2 compiled blocks, 6 block calls, 0 eager")
    monkeypatch.setattr(bench, "collect_source", lambda: {"harness_sha256": "harness-sha256"})

    result = bench.preflight(make_config("C1"))

    # The default session is the scheduled one, which also checks the comparison tools.
    assert result["ok"] is False
    assert [(check["name"], check["ok"]) for check in result["checks"]] == [
        ("config", True),
        ("probe_targets", False),
        ("comparison_self_test", True),
        ("compile_self_test", True),
        ("block_self_test", True),
    ]
    assert "has no attribute effective_attention" in result["checks"][1]["detail"]
    # The comparison check scores on the preflight's device, like the two compile self-tests.
    assert result["checks"][2]["detail"] == "compared with alex on cpu"

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config("C1")), encoding="utf-8")
    assert bench.main(["preflight", "--config", str(config_path)]) == 3
    assert json.loads(capsys.readouterr().out)["ok"] is False

    # An invalid config skips the target check, because the targets depend on the config.
    invalid = bench.preflight({"schema": "other/1"})
    assert [(check["name"], check["ok"]) for check in invalid["checks"]] == [
        ("config", False),
        ("compile_self_test", True),
        ("block_self_test", True),
    ]


def test_preflight_passes_and_writes_its_result(monkeypatch, tmp_path, capsys):
    sessions = []

    def install(config, session, recorder):
        sessions.append(session)
        patches = bench.PatchSet()
        patches.installed.append({"target": "fake:compile", "kind": "compile"})
        return patches, None

    monkeypatch.setattr(bench, "install_probes", install)
    monkeypatch.setattr(bench, "_compile_self_test", lambda device: f"compiled on {device}")
    monkeypatch.setattr(bench, "_block_self_test", lambda device: f"blocks on {device}")
    monkeypatch.setattr(bench, "collect_source", lambda: {"harness_sha256": "harness-sha256"})
    monkeypatch.setattr(bench, "_repository_root", lambda: tmp_path / "repository")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config("C5")), encoding="utf-8")
    out_file = tmp_path / "preflight.json"
    command = ["preflight", "--config", str(config_path), "--session", "plain"]

    assert bench.main([*command, "--out", str(out_file)]) == 0

    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == written
    assert written["ok"] is True
    assert sessions == ["plain"]
    assert written["checks"][0]["detail"] == "combination C5, session plain"
    assert written["checks"][2]["detail"] == "compiled on cpu"
    # A session without a schedule compares no outputs, so the comparison tools are not checked.
    assert [check["name"] for check in written["checks"]] == [
        "config",
        "probe_targets",
        "compile_self_test",
        "block_self_test",
    ]

    assert bench.main([*command, "--out", str(tmp_path / "repository" / "preflight.json")]) == 2
    assert "is inside the repository" in capsys.readouterr().err


def test_preflight_reports_failing_comparison_tools(monkeypatch):
    def install(config, session, recorder):
        return bench.PatchSet(), None

    def failing_self_test(net, device):
        raise bench.EvidenceError("the video writer wrote no file")

    monkeypatch.setattr(bench, "install_probes", install)
    monkeypatch.setattr(bench, "_comparison_self_test", failing_self_test)
    monkeypatch.setattr(bench, "_compile_self_test", lambda device: "1 graph(s), 3 executions")
    monkeypatch.setattr(bench, "_block_self_test", lambda device: "2 compiled blocks, 6 block calls, 0 eager")
    monkeypatch.setattr(bench, "collect_source", lambda: {"harness_sha256": "harness-sha256"})

    result = bench.preflight(make_config("C1"), session="scheduled")

    assert result["ok"] is False
    assert [(check["name"], check["ok"]) for check in result["checks"]] == [
        ("config", True),
        ("probe_targets", True),
        ("comparison_self_test", False),
        ("compile_self_test", True),
        ("block_self_test", True),
    ]
    assert result["checks"][2]["detail"] == "EvidenceError: the video writer wrote no file"
