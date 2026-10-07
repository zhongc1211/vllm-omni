# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for sigma-window attention schedules and CUDA graph selection."""

import pytest
import torch

from vllm_omni.diffusion.attention.schedule import (
    parse_attention_sigma_schedule,
    reject_mixed_attention_schedules,
    resolve_attention_sigma_schedule,
    resolve_batch_attention_sigma_schedule,
    select_attention_profile_by_sigma,
)
from vllm_omni.diffusion.attention.sigma_graphs import SigmaCudaGraphTable

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sigma_schedule_inherits_disables_and_replaces():
    default = parse_attention_sigma_schedule([{"low": 0.0, "high": 0.3, "profile": "sparse"}])
    assert resolve_attention_sigma_schedule(None, default, profiles={"sparse"}) == default
    assert resolve_attention_sigma_schedule([], default, profiles={"sparse"}) == ()
    replaced = [{"low": 0.7, "high": 1.0, "profile": "sparse"}]
    assert resolve_attention_sigma_schedule(replaced, default, profiles={"sparse"})[0].low == 0.7


def test_sigma_schedule_rejects_overlap_and_unknown_profile():
    with pytest.raises(ValueError, match="overlap"):
        parse_attention_sigma_schedule(
            [{"low": 0.0, "high": 0.5, "profile": "sparse"}, {"low": 0.5, "high": 1.0, "profile": "sparse"}]
        )
    with pytest.raises(ValueError, match="unknown profile"):
        resolve_attention_sigma_schedule([{"low": 0.0, "high": 0.2, "profile": "missing"}], (), profiles=set())


def test_sigma_selection_uses_normalized_noise_and_gaps():
    schedule = parse_attention_sigma_schedule([{"low": 0.0, "high": 0.3, "profile": "sparse"}])
    assert select_attention_profile_by_sigma(schedule, 0.0) == "sparse"
    assert select_attention_profile_by_sigma(schedule, 0.3) == "sparse"
    assert select_attention_profile_by_sigma(schedule, 0.31) is None


def test_request_and_step_modes_share_batch_resolution():
    service = type("Cfg", (), {"profiles": {"sparse": object()}, "sigma": [{"low": 0.8, "high": 1.0, "profile": "sparse"}]})()
    od_config = type("Od", (), {"diffusion_attention_schedule": service})()
    request = type("Req", (), {"sampling_params": type("S", (), {"attention_sigma_schedule": None})()})()
    state = type("State", (), {"sampling": type("S", (), {"attention_sigma_schedule": []})()})()
    assert resolve_batch_attention_sigma_schedule([request], od_config)[0].profile == "sparse"
    assert resolve_batch_attention_sigma_schedule([state], od_config) == ()


def test_mixed_step_and_sigma_schedules_are_rejected():
    with pytest.raises(ValueError, match="cannot combine"):
        reject_mixed_attention_schedules((object(),), (object(),))


def test_cuda_graph_selection_does_not_require_a_device():
    table = SigmaCudaGraphTable()
    table.register("TRTLLM_ATTN", 0.0, 0.4, "early")
    table.register("TORCH_SDPA", 0.6, 1.0, "late")
    assert table.select("TRTLLM_ATTN", 0.2) == "early"
    assert table.select("TRTLLM_ATTN", 0.9) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA")
def test_cuda_graph_capture_registers_the_backend_window():
    table = SigmaCudaGraphTable()
    static = torch.ones(1, device="cuda")

    def replay():
        static.add_(1)

    table.capture("TRTLLM_ATTN", 0.0, 0.5, replay)
    assert table.select("TRTLLM_ATTN", 0.1) is not None
