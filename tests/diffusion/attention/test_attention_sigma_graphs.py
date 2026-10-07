# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real local dispatch and fixed-signature explicit attention graphs.

CPU tests do not simulate successful CUDA captures. CUDA tests use production
SDPA/Flash implementations, and skip only for absent hardware/dependencies.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.attention.layer as layer_mod
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
    VideoTokenLayout,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.attention.backends.trtllm_attn import TrtllmAttentionBackend, TrtllmAttentionImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.schedule import parse_attention_sigma_schedule
from vllm_omni.diffusion.attention.sigma_graphs import (
    SigmaAttentionGraphCache,
    SigmaCudaGraphTable,
    _CapturedAttention,
    _execution_signature,
    _graph_metadata,
    _signature,
    capture_eligibility,
    enable_sigma_attention_graphs,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionScheduleConfig, AttentionSpec, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import (
    DenoiseProgressMixin,
    bind_attention_sigma_schedule,
    set_forward_context,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def clear_inherited_attention_backend(monkeypatch):
    monkeypatch.delenv("DIFFUSION_ATTENTION_BACKEND", raising=False)


def _sdpa(**kwargs):
    return SDPAImpl(num_heads=2, head_size=64, softmax_scale=0.125, causal=False, **kwargs)


def _config(profiles=None, **kwargs):
    schedule = AttentionScheduleConfig(profiles=profiles or {"dense": AttentionConfig()})
    return OmniDiffusionConfig(diffusion_attention_schedule=schedule, **kwargs)


def _resolve(*, attention_config=None, **kwargs):
    spec = None if attention_config is None else attention_config.default
    backend = {
        "TORCH_SDPA": SDPABackend,
        "FLASH_ATTN": FlashAttentionBackend,
        "TRTLLM_ATTN": TrtllmAttentionBackend,
    }.get(None if spec is None else spec.backend, SDPABackend)
    return backend, spec


def _layer(monkeypatch, config, **kwargs):
    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", _resolve)
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: NoParallelAttention())
    with set_current_diffusion_config(config):
        return Attention(num_heads=2, head_size=64, softmax_scale=0.125, causal=False, **kwargs)


def _windows(*entries):
    return parse_attention_sigma_schedule([{"low": low, "high": high, "profile": name} for low, high, name in entries])


@pytest.mark.cpu
def test_graph_table_boundaries_gaps_and_configuration_keys():
    table = SigmaCudaGraphTable()
    # Separate prepared configuration keys, even when backend names match.
    table.register("TRTLLM_ATTN:threshold=.25", 0.0, 0.4, "quarter")
    table.register("TRTLLM_ATTN:threshold=.5", 0.4, 1.0, "half")
    assert table.select("TRTLLM_ATTN:threshold=.25", 0.399) == "quarter"
    assert table.select("TRTLLM_ATTN:threshold=.25", 0.4) is None
    assert table.select("TRTLLM_ATTN:threshold=.5", 0.4) == "half"
    assert table.select("TRTLLM_ATTN:threshold=.5", 1.0) == "half"
    with pytest.raises(ValueError, match="overlap"):
        table.register("TRTLLM_ATTN:threshold=.25", 0.2, 0.5, object())


@pytest.mark.cpu
def test_local_dispatch_uses_prepared_configuration_identity_including_gaps(monkeypatch):
    quarter = AttentionConfig(default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax={"threshold": 0.25}))
    half = AttentionConfig(default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax={"threshold": 0.5}))
    config = _config({"quarter": quarter, "half": half, "alias": quarter})
    layer = _layer(monkeypatch, config)
    records = layer._schedule_candidates
    assert records["quarter"] is records["alias"]
    assert records["quarter"].impl is not records["half"].impl
    calls = []

    def run(selected, candidates, *inputs):
        calls.append((selected, {id(impl) for impl in candidates}))
        return inputs[0].clone()

    layer._sigma_graph_cache = SimpleNamespace(run=run)
    query = torch.randn(1, 8, 2, 64)
    publisher = DenoiseProgressMixin()
    windows = _windows((0.0, 0.2, "quarter"), (0.4, 0.8, "half"))
    with set_forward_context(omni_diffusion_config=config), bind_attention_sigma_schedule(windows):
        for index, sigma in enumerate((0.1, 0.5, 0.3, 1.0, 0.0, 0.8)):
            publisher.record_denoise_step(index, normalized_sigma=sigma, total_steps=6)
            layer(query, query, query)
        publisher.record_denoise_step(None)
        # No denoise publication: neither selection nor cache dispatch is active.
        monkeypatch.setattr(layer.attention, "forward", lambda *args: query)
        layer(query, query, query)
    assert [selected for selected, _ in calls] == [
        records["quarter"].impl,
        records["half"].impl,
        layer.attention,
        layer.attention,
        records["quarter"].impl,
        layer.attention,
    ]
    assert all(len(identities) == 3 for _, identities in calls)


@pytest.mark.cpu
def test_cpu_cache_keeps_real_sdpa_eager_with_no_entries(monkeypatch):
    # The shared SDPA math is real; only its platform dispatch is adapted to CPU.
    monkeypatch.setattr(SDPAImpl, "forward", SDPAImpl._forward_impl)
    impl = _sdpa()
    cache = SigmaAttentionGraphCache()
    for _ in range(20):
        query, key, value = (torch.randn(1, 8, 2, 64) for _ in range(3))
        mask = torch.rand(1, 8) > 0.3
        metadata = AttentionMetadata(attn_mask=mask)
        out = cache.run(impl, [impl], query, key, value, metadata)
        torch.testing.assert_close(out, impl.forward(query, key, value, metadata))
    assert cache.entries == {}


@pytest.mark.cpu
def test_eligibility_rejects_private_gates_opaque_metadata_and_unaudited_overrides():
    sdpa = _sdpa()
    flash = FlashAttentionImpl(2, 64, 0.125)
    trt = TrtllmAttentionImpl(2, 128, 0.125)
    gated = TrtllmAttentionImpl(2, 128, 0.125, backend_kwargs={"disabled_until_timestep": 0.6})
    assert capture_eligibility(sdpa, AttentionMetadata(attn_mask=torch.ones(1, 8, dtype=torch.bool)))
    assert capture_eligibility(flash, None)
    assert not capture_eligibility(flash, AttentionMetadata(attn_mask=torch.ones(1, 8, dtype=torch.bool)))
    # Ungated TRT also stays eager because its global scratch workspace is shared.
    assert not capture_eligibility(trt, None)
    assert not capture_eligibility(gated, None)
    assert not capture_eligibility(SimpleNamespace(backend="RAINFUSION"), None)
    assert not capture_eligibility(SimpleNamespace(backend="FASTVIDEO_VSA"), None)
    # SDPA consumes only attn_mask; opaque ignored extra is not graph state.
    assert capture_eligibility(sdpa, AttentionMetadata(extra={"opaque": object()}))
    assert not capture_eligibility(flash, AttentionMetadata(full_attn_spans=[[(0, 4)]]))

    class Override(SDPAImpl):
        pass

    assert not capture_eligibility(Override(2, 64, 0.125), None)


@pytest.mark.cpu
def test_metadata_signature_keys_python_values_and_not_tensor_contents():
    first = AttentionMetadata(extra={"max_seqlen_q": 8, "cu_seqlens_q": torch.tensor([0, 8])})
    second = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 4]), "max_seqlen_q": 8})
    third = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 8]), "max_seqlen_q": 4})
    assert _signature(first) == _signature(second)
    assert _signature(first) != _signature(third)
    with pytest.raises(TypeError, match="Unsupported"):
        _signature(AttentionMetadata(extra={"opaque": object()}))


@pytest.mark.cpu
@pytest.mark.parametrize(
    "reason", ["cpu", "no_cuda", "enforce_eager", "no_schedule", "hsdp", "offload", "ring", "paged"]
)
def test_runner_enabling_is_conservative(monkeypatch, reason):
    config = _config(enforce_eager=reason == "enforce_eager", enable_cpu_offload=reason == "offload")
    layer = _layer(monkeypatch, config)
    if reason == "no_schedule":
        config.diffusion_attention_schedule = None
    if reason == "hsdp":
        config.parallel_config.use_hsdp = True
    if reason == "ring":
        layer.use_ring = True
    if reason == "paged":
        layer.paged_kv_cache_role = "self"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: reason != "no_cuda")
    device = "cpu" if reason == "cpu" else "cuda"
    assert not enable_sigma_attention_graphs(layer, config, device)
    assert layer._sigma_graph_cache is None


@pytest.mark.cpu
@pytest.mark.parametrize("enabled", [True, False])
def test_runner_disables_compiler_cudagraphs_only_when_explicit_cache_enabled(monkeypatch, enabled):
    from vllm_omni.diffusion.worker import diffusion_model_runner as runner_mod

    model = nn.Linear(2, 2)
    runner = runner_mod.DiffusionModelRunner.__new__(runner_mod.DiffusionModelRunner)
    runner.pipeline = SimpleNamespace(transformer=model)
    runner.od_config = _config()
    runner.device = torch.device("cuda")
    calls = []
    monkeypatch.setattr(runner_mod, "enable_sigma_attention_graphs", lambda *args: enabled)
    monkeypatch.setattr(runner_mod, "regionally_compile", lambda model, **kw: calls.append(kw) or model)
    runner._compile_transformer("transformer")
    assert len(calls) == 1
    assert calls[0].get("options") == ({"triton.cudagraphs": False} if enabled else None)


@pytest.mark.cpu
def test_candidate_registration_deduplicates_aliases_and_skips_rechecking_types(monkeypatch):
    from vllm_omni.diffusion.attention import sigma_graphs

    eligible, unsupported = object(), object()
    calls = []
    monkeypatch.setattr(
        sigma_graphs,
        "capture_eligibility",
        lambda impl, metadata: calls.append(impl) is None and impl is eligible,
    )
    cache = SigmaAttentionGraphCache()
    candidates = (eligible, eligible, unsupported)
    cache._register_candidates(candidates)
    cache._register_candidates(candidates)
    cache._register_candidates(tuple([eligible, unsupported]))
    assert calls == [eligible, unsupported]
    assert cache._pending == {id(eligible): eligible}
    assert cache.entries == {}


@pytest.mark.cpu
def test_execution_signature_keys_cuda_autocast_state(monkeypatch):
    inputs = (torch.zeros(1, 8, 2, 64),) * 3 + (None,)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: False)
    disabled = _execution_signature(inputs)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: True)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: torch.float16)
    half = _execution_signature(inputs)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: torch.bfloat16)
    bfloat = _execution_signature(inputs)
    assert disabled != half != bfloat


@pytest.mark.cpu
def test_capture_close_waits_once_for_outstanding_replay():
    waits = []
    entry = _CapturedAttention.__new__(_CapturedAttention)
    entry.done = SimpleNamespace(synchronize=lambda: waits.append("wait"))
    entry.close()
    entry.close()
    assert waits == ["wait"]
    assert entry.done is None


@pytest.mark.cuda
@requires_cuda
@torch.no_grad()
@pytest.mark.parametrize("capture_autocast", [False, True])
def test_cuda_autocast_change_uses_eager_without_replaying_old_dtype(capture_autocast):
    impl = _sdpa()
    cache = SigmaAttentionGraphCache()
    inputs = _inputs()
    with torch.autocast("cuda", dtype=torch.float16, enabled=capture_autocast):
        cache.run(impl, [impl], *inputs, None)
    with torch.autocast("cuda", dtype=torch.float16, enabled=not capture_autocast):
        out = cache.run(impl, [impl], *inputs, None)
        expected = impl.forward(*inputs)
        assert out.dtype == expected.dtype
        torch.testing.assert_close(out, expected)
    assert len(cache.entries) == 1
    assert cache.entries[id(impl)].replays == 1


@pytest.mark.cuda
@requires_cuda
@torch.no_grad()
def test_cuda_release_cache_waits_for_queued_nondefault_stream_replay(monkeypatch):
    impl = _sdpa()
    cache = SigmaAttentionGraphCache()
    inputs = _inputs()
    cache.run(impl, [impl], *inputs, None)
    torch.accelerator.synchronize()
    stream = torch.cuda.Stream()
    waits = []
    close = _CapturedAttention.close

    def record_close(entry):
        waits.append(entry.done)
        close(entry)

    monkeypatch.setattr(_CapturedAttention, "close", record_close)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        out = cache.run(impl, [impl], *inputs, None)
    del cache
    # Teardown is a rare synchronization boundary, unlike ordinary replay.
    assert len(waits) == 1
    assert waits[0].query()
    # Overwrite fresh allocations on the initial stream before explicitly
    # synchronizing the replay stream; private/static storage must be safe.
    scratch = [torch.empty_like(inputs[0]).fill_(123) for _ in range(8)]
    stream.synchronize()
    torch.testing.assert_close(out, impl.forward(*inputs))
    assert all(t.shape == inputs[0].shape for t in scratch)


def _inputs(seq=8, dtype=torch.float32):
    return tuple(torch.randn(1, seq, 2, 64, dtype=dtype, device="cuda") for _ in range(3))


@pytest.mark.cuda
@requires_cuda
@torch.no_grad()
def test_cuda_precaptures_all_configurations_copies_inputs_and_retains_prior_outputs():
    baseline = _sdpa()
    quarter = SDPAImpl(2, 64, 0.25)
    half = SDPAImpl(2, 64, 0.5)
    candidates = [baseline, quarter, half, quarter]  # Duplicate profile alias.
    cache = SigmaAttentionGraphCache()
    inputs = _inputs()
    mask = torch.ones(1, 8, dtype=torch.bool, device="cuda")
    metadata = AttentionMetadata(attn_mask=mask)
    prior = cache.run(quarter, candidates, *inputs, metadata)
    assert set(cache.entries) == {id(baseline), id(quarter), id(half)}
    assert [cache.entries[id(impl)].replays for impl in (baseline, quarter, half)] == [0, 1, 0]
    entry = cache.entries[id(quarter)]
    assert entry.inputs[0].data_ptr() != inputs[0].data_ptr()
    assert entry.inputs[3].attn_mask.data_ptr() != metadata.attn_mask.data_ptr()
    assert entry.output.data_ptr() != prior.data_ptr()
    saved = prior.clone()
    original_inputs = tuple(t.clone() for t in inputs)
    for index in range(30):
        selected = candidates[index % 3]
        fresh = _inputs()
        mask = torch.ones(1, 8, dtype=torch.bool, device="cuda")
        mask[:, 4:] = index % 2 == 0
        metadata = AttentionMetadata(attn_mask=mask)
        out = cache.run(selected, candidates, *fresh, metadata)
        torch.testing.assert_close(out, selected.forward(*fresh, metadata))
    assert len(cache.entries) == 3
    assert sum(entry.replays for entry in cache.entries.values()) == 31
    torch.testing.assert_close(prior, saved)
    for before, after in zip(original_inputs, inputs):
        torch.testing.assert_close(before, after)
    assert not torch.allclose(prior, out)
    # Tensor geometry changes are eager, not another graph bucket.
    changed = _inputs(seq=9)
    out = cache.run(half, candidates, *changed, None)
    torch.testing.assert_close(out, half.forward(*changed, None))
    assert len(cache.entries) == 3
    assert sum(entry.replays for entry in cache.entries.values()) == 31


@pytest.mark.cuda
@requires_cuda
@torch.no_grad()
def test_cuda_real_layer_precaptures_baseline_and_profile_before_first_sigma_execution(monkeypatch):
    config = _config({"dense": AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA"))})
    layer = _layer(monkeypatch, config)
    assert enable_sigma_attention_graphs(layer, config, "cuda")
    windows = _windows((0.0, 0.4, "dense"))
    publisher = DenoiseProgressMixin()
    with set_forward_context(omni_diffusion_config=config), bind_attention_sigma_schedule(windows):
        for index, sigma in enumerate((0.2, 0.1, 0.8, 1.0, 0.4, 0.0)):
            publisher.record_denoise_step(index, normalized_sigma=sigma, total_steps=6)
            inputs = _inputs()
            metadata = AttentionMetadata(
                attn_mask=torch.ones(1, 8, dtype=torch.bool, device="cuda"),
                video_layout=VideoTokenLayout(prefix_len=0, latent_grid=(1, 2, 4)),
                extra={"valid_kv_length": 8, "attention_mask_mode": "padding"},
            )
            out = layer(*inputs, metadata)
            impl = layer.effective_attention()[0]
            torch.testing.assert_close(out, impl.forward(*inputs, metadata))
            assert len(layer._sigma_graph_cache.entries) == 2
    assert sum(entry.replays for entry in layer._sigma_graph_cache.entries.values()) == 6


@pytest.mark.cuda
@requires_cuda
@torch.no_grad()
def test_cuda_cache_handles_nondefault_device_stream_and_broadcast_mask():
    device = torch.device("cuda", torch.accelerator.device_count() - 1)
    impl = _sdpa()
    cache = SigmaAttentionGraphCache()
    previous = None
    saved = None
    for _ in range(3):
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            inputs = tuple(torch.randn(1, 8, 2, 64, device=device) for _ in range(3))
            mask = torch.ones(1, 1, 1, 8, dtype=torch.bool, device=device).expand(1, 2, 8, 8)
            metadata = AttentionMetadata(attn_mask=mask)
            out = cache.run(impl, [impl], *inputs, metadata)
            torch.testing.assert_close(out, impl.forward(*inputs, metadata))
            if previous is not None:
                torch.testing.assert_close(previous, saved)
            previous, saved = out, out.clone()
        stream.synchronize()
    assert cache.entries[id(impl)].replays == 3


@pytest.mark.cuda
@requires_cuda
@torch.no_grad()
@pytest.mark.parametrize("packed", [False, True])
def test_cuda_flash_dense_and_packed_refresh_metadata_and_key_static_python_values(packed):
    from vllm_omni.diffusion.attention.backends.utils import fa

    if not fa.HAS_FLASH_ATTN:
        pytest.skip("FlashAttention dependency is absent")
    if packed and fa.flash_attn_varlen_func is None:
        pytest.skip("FlashAttention varlen dependency is absent")
    impl = FlashAttentionImpl(2, 64, 0.125)
    cache = SigmaAttentionGraphCache()
    prior = None
    saved = None
    for index in range(6):
        inputs = _inputs(dtype=torch.float16)
        metadata = AttentionMetadata(
            video_layout=VideoTokenLayout(prefix_len=0, latent_grid=(1, 2, 4)),
            extra={"attention_mask_mode": "none", "kv_cache_dtype": "float", "valid_kv_length": 8},
        )
        if packed:
            # Same geometry and Python maxima, changed document boundaries.
            boundaries = [0, 4, 8] if index % 2 == 0 else [0, 3, 8]
            cu = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
            metadata.extra.update(
                {
                    "cu_seqlens_q": cu,
                    "cu_seqlens_k": cu,
                    "max_seqlen_q": 8,
                    "max_seqlen_k": 8,
                }
            )
            metadata.packed_padding = PackedPaddingMetadata(
                q_length=8,
                kv_length=8,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
            )
        out = cache.run(impl, [impl], *inputs, metadata)
        torch.testing.assert_close(out, impl.forward(*inputs, metadata), atol=1e-3, rtol=1e-3)
        if prior is not None:
            torch.testing.assert_close(prior, saved)
        prior, saved = out, out.clone()
    assert cache.entries[id(impl)].replays == 6
    if packed:
        # Same tensor shapes but a different max length must not replay frozen
        # Python arguments. Only this call is eager; memory stays bounded.
        metadata.extra["max_seqlen_q"] = metadata.extra["max_seqlen_k"] = 5
        out = cache.run(impl, [impl], *inputs, metadata)
        torch.testing.assert_close(out, impl.forward(*inputs, metadata), atol=1e-3, rtol=1e-3)
        assert cache.entries[id(impl)].replays == 6
    assert len(cache.entries) == 1


@pytest.mark.cpu
def test_sdpa_canonical_metadata_is_reachable_with_h3_video_and_padding_fields():
    cu = torch.tensor([0, 8], dtype=torch.int32)
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    metadata = AttentionMetadata(
        attn_mask=mask,
        full_attn_spans=[[(0, 4)]],  # SDPA consumes the explicit 4D mask, not this plan.
        video_layout=VideoTokenLayout(prefix_len=0, latent_grid=(1, 2, 4)),
        packed_padding=PackedPaddingMetadata(8, 8, cu, cu),
        extra={"max_seqlen_q": 8, "valid_kv_length": 8, "opaque_ignored": object()},
    )
    canonical = _graph_metadata(_sdpa(), metadata)
    assert canonical.attn_mask is mask
    assert canonical.video_layout is None
    assert canonical.packed_padding is None
    assert canonical.full_attn_spans is None
    assert canonical.extra == {}
    assert metadata.video_layout is not None  # Never mutate producer-owned metadata.
    assert metadata.extra["valid_kv_length"] == 8


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("capture_inference_mode", [False, True])
def test_cuda_inference_mode_is_part_of_fixed_signature(capture_inference_mode):
    cache = SigmaAttentionGraphCache()
    impl = _sdpa()
    with torch.inference_mode(capture_inference_mode), torch.no_grad():
        inputs = _inputs()
        prior = cache.run(impl, [impl], *inputs, None)
        saved = prior.clone()
    with torch.inference_mode(not capture_inference_mode), torch.no_grad():
        inputs = _inputs()
        out = cache.run(impl, [impl], *inputs, None)
        torch.testing.assert_close(out, impl.forward(*inputs))
        torch.testing.assert_close(prior, saved)
    assert len(cache.entries) == 1
    assert cache.entries[id(impl)].replays == 1  # Opposite mode was eager, no new graph.


@pytest.mark.cuda
@requires_cuda
@torch.no_grad()
def test_cuda_graph_replay_inside_compiled_model_has_no_sigma_driven_recompiles(monkeypatch):
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    config = _config({"dense": AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA"))})
    layer = _layer(monkeypatch, config)
    assert enable_sigma_attention_graphs(layer, config, "cuda")
    graph_calls = []
    executions = []

    class Block(nn.Module):
        def forward(self, hidden):
            qkv = torch.sin(hidden).reshape(1, 8, 2, 64)
            output = layer(qkv, qkv, qkv)
            return torch.cos(output.reshape(1, 8, 128))

    def counting_backend(gm, example_inputs):
        index = len(graph_calls)
        graph_calls.append(
            {
                getattr(node.target, "__name__", str(node.target))
                for node in gm.graph.nodes
                if node.op.startswith("call_")
            }
        )
        executions.append(0)

        def run(*args):
            executions[index] += 1
            return gm.forward(*args)

        return run

    model = torch.compile(Block(), backend=counting_backend)
    publisher = DenoiseProgressMixin()
    windows = _windows((0.0, 0.4, "dense"))
    first_frames = None
    try:
        with set_forward_context(omni_diffusion_config=config), bind_attention_sigma_schedule(windows):
            for index, sigma in enumerate([0.2, 1.0, 0.4, 0.8, 0.0, 0.399] * 6):
                publisher.record_denoise_step(index, normalized_sigma=sigma, total_steps=36)
                hidden = torch.randn(1, 8, 128, dtype=torch.float16, device="cuda")
                before = list(executions)
                output = model(hidden)
                qkv = torch.sin(hidden).reshape(1, 8, 2, 64)
                expected = torch.cos(layer.effective_attention()[0].forward(qkv, qkv, qkv).reshape(1, 8, 128))
                torch.testing.assert_close(output, expected)
                assert len(layer._sigma_graph_cache.entries) == 2
                frames = counters["frames"]["total"]
                if first_frames is None:
                    first_frames = frames
                    assert graph_calls, "the enclosing model must execute compiled graphs"
                else:
                    assert frames == first_frames
                delta = [count - (before[i] if i < len(before) else 0) for i, count in enumerate(executions)]
                assert sum(count for calls, count in zip(graph_calls, delta) if "sin" in calls) == 1
                assert sum(count for calls, count in zip(graph_calls, delta) if "cos" in calls) == 1
        assert sum(entry.replays for entry in layer._sigma_graph_cache.entries.values()) == 36
        assert all("scaled_dot_product_attention" not in calls for calls in graph_calls)
    finally:
        torch._dynamo.reset()


@pytest.mark.cuda
@requires_cuda
@torch.no_grad()
def test_cuda_capture_errors_propagate_instead_of_silent_eager_fallback(monkeypatch):
    from vllm_omni.diffusion.attention import sigma_graphs

    def fail(*args):
        raise ValueError("backend model failure")

    monkeypatch.setattr(sigma_graphs, "_CapturedAttention", fail)
    impl = _sdpa()
    cache = SigmaAttentionGraphCache()
    with pytest.raises(ValueError, match="backend model failure"):
        cache.run(impl, [impl], *_inputs(), None)
    assert cache.entries == {}


@pytest.mark.cuda
@requires_cuda
@torch.no_grad()
def test_cuda_automatic_flash_fp32_fallback_can_capture_sdpa_not_flash(monkeypatch):
    config = _config({"dense": AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA"))})

    def automatic_flash(*, attention_config=None, **kwargs):
        if attention_config is not None and attention_config.default is not None:
            return _resolve(attention_config=attention_config, **kwargs)
        return FlashAttentionBackend, None

    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", automatic_flash)
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: NoParallelAttention())
    with set_current_diffusion_config(config):
        layer = Attention(num_heads=2, head_size=64, softmax_scale=0.125, causal=False, allow_fp32_fallback=True)
    assert enable_sigma_attention_graphs(layer, config, "cuda")
    windows = _windows((0.0, 0.4, "dense"))
    with set_forward_context(omni_diffusion_config=config), bind_attention_sigma_schedule(windows):
        DenoiseProgressMixin().record_denoise_step(0, normalized_sigma=0.8, total_steps=1)
        inputs = _inputs()
        out = layer(*inputs)
        torch.testing.assert_close(out, layer.sdpa_fallback.forward(*inputs))
    cache = layer._sigma_graph_cache
    assert id(layer.attention) not in cache.entries  # Flash never sees the FP32 inputs.
    assert set(cache.entries) == {id(layer.sdpa_fallback), id(layer._schedule_candidates["dense"].impl)}
    assert cache.entries[id(layer.sdpa_fallback)].replays == 1
