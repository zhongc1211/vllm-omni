# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit attention CUDA graphs; sigma selection stays outside Dynamo.

The cache captures eager kernels, not compiled leaves. The runner disables Inductor
CUDA graphs in the surrounding compiled blocks when it enables this cache.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch

from vllm_omni.diffusion.attention.capabilities import MaskMode


@dataclass(frozen=True)
class SigmaGraphWindow:
    backend: str
    low: float
    high: float
    graph: Any


class SigmaCudaGraphTable:
    """Window lookup utility. Production replay is keyed by prepared impl identity.

    Backend names alone are not cache keys: two profiles of the same backend can
    have different kwargs/calibration. Attention selects the prepared object first.
    """

    def __init__(self) -> None:
        self._windows: list[SigmaGraphWindow] = []

    def register(self, backend: str, low: float, high: float, graph: Any) -> None:
        if not backend:
            raise ValueError("CUDA graph backend name is required")
        if not 0.0 <= float(low) < float(high) <= 1.0:
            raise ValueError(f"CUDA graph window [{low}, {high}] must lie in [0, 1]")
        for existing in self._windows:
            if existing.backend == backend and not (high <= existing.low or low >= existing.high):
                raise ValueError(f"CUDA graph windows overlap for backend {backend!r}")
        self._windows.append(SigmaGraphWindow(backend, float(low), float(high), graph))

    def select(self, backend: str, sigma: float) -> Any | None:
        for window in self._windows:
            if window.backend == backend and (window.low <= float(sigma) < window.high or window.high == sigma == 1.0):
                return window.graph
        return None

    def capture(self, backend: str, low: float, high: float, replay: Callable[[], Any]) -> Any:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graph capture requires CUDA")
        # This low-level utility cannot own a callable's inputs; production uses
        # _CapturedAttention below, which owns all input and output lifetimes.
        device = torch.accelerator.current_device_index()
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                replay()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replay()
        current.wait_stream(stream)
        self.register(backend, low, high, graph)
        return graph


_PACKED_KEYS = frozenset({"cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"})


_INELIGIBLE = object()


def _graph_metadata(impl, metadata):
    """Canonicalize only metadata actually consumed by audited CUDA forwards.

    Called after Attention's compatibility checks and SP preprocessing. SDPA
    consumes only attn_mask. Flash CUDA consumes full_attn_spans/query_ranges,
    packed cu_seqlens/maxima, and mask mode; video_layout/packed_padding and all
    other extra fields are ignored by that CUDA implementation. Do not carry
    ignored metadata into capture or signature guards.
    """
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

    if type(impl) not in (SDPAImpl, FlashAttentionImpl):
        return _INELIGIBLE
    if metadata is not None and type(metadata) is not AttentionMetadata:
        return _INELIGIBLE  # Subclasses may carry backend-owned state.
    if type(impl) is SDPAImpl:
        mask = None if metadata is None else metadata.attn_mask
        return None if mask is None else AttentionMetadata(attn_mask=mask)
    if metadata is None:
        return None
    if metadata.full_attn_spans is not None:
        return _INELIGIBLE  # Piecewise attention is not an audited capture route.
    extra = metadata.extra
    mask_mode = extra.get("attention_mask_mode")
    if (
        metadata.attn_mask is not None
        and mask_mode is not None
        and mask_mode not in (MaskMode.NONE, MaskMode.PADDING, MaskMode.ARBITRARY)
    ):
        return _INELIGIBLE  # Let eager Flash report invalid mask modes.
    present = _PACKED_KEYS.intersection(extra)
    if present:
        if present != _PACKED_KEYS:
            return _INELIGIBLE  # Eager Flash reports incomplete producer metadata.
        if not all(
            isinstance(extra[name], torch.Tensor) and extra[name].is_cuda for name in ("cu_seqlens_q", "cu_seqlens_k")
        ) or not all(type(extra[name]) is int for name in ("max_seqlen_q", "max_seqlen_k")):
            return _INELIGIBLE
        # Packed dispatch precedes mask dispatch in forward_cuda. These tensor
        # buffers are refreshed even when only the packed boundaries change.
        return AttentionMetadata(extra={name: extra[name] for name in _PACKED_KEYS})
    if metadata.attn_mask is not None and mask_mode != MaskMode.NONE:
        return _INELIGIBLE  # Mask unpadding reads device values on the host.
    return None  # Dense route, with no mask or a producer-declared no-op mask.


def capture_eligibility(impl, metadata) -> bool:
    """Only audited, context-independent production implementations may capture.

    SDPA supports runtime masks. CUDA Flash supports dense/no-mask and producer
    packed cu_seqlens (not mask unpadding or piecewise plans). TRT stays eager:
    generic packed TRT reads device scalars, its timestep skip gate must never
    be frozen, and even ungated TRT uses a class-global scratch workspace whose
    eager and captured uses cannot be isolated by a per-layer cache event.
    RAINFUSION/FASTVIDEO and all other backends/overrides stay eager because their
    private context or capture safety has not been audited. SP stays outside.
    """
    return _graph_metadata(impl, metadata) is not _INELIGIBLE


def _signature(value):
    """Tensor geometry plus *values* of Python metadata, never device contents.

    Equal metadata shapes are insufficient: max_seqlen and other Python values
    control backend dispatch. Tensor contents are instead copied on every replay.
    Unknown Python objects are ineligible, not keyed by repr or object identity.
    """
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), value.dtype, value.device, value.requires_grad)
    if value is None or type(value) in (str, int, float, bool):
        return (type(value), value)
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((f.name, _signature(getattr(value, f.name))) for f in fields(value)))
    if isinstance(value, dict):
        return (dict, tuple((key, _signature(item)) for key, item in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_signature(item) for item in value))
    raise TypeError(f"Unsupported attention graph metadata type: {type(value).__name__}")


def _owned_copy(value, device, copies):
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise ValueError("Attention graph tensors must be on the Q/K/V device")
        # clone preserves dense strides, and materializes broadcast/overlapping
        # views into writable storage. The live signature is still keyed exactly.
        static = value.clone(memory_format=torch.preserve_format)
        copies.append(static)
        return static
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: _owned_copy(getattr(value, f.name), device, copies) for f in fields(value)})
    if isinstance(value, dict):
        return {key: _owned_copy(item, device, copies) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return type(value)(_owned_copy(item, device, copies) for item in value)
    return value


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            yield from _tensors(getattr(value, f.name))
    elif isinstance(value, dict):
        for _, item in sorted(value.items()):
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _execution_signature(inputs):
    autocast = torch.is_autocast_enabled("cuda")
    return (
        torch.is_inference_mode_enabled(),
        autocast,
        torch.get_autocast_dtype("cuda") if autocast else None,
        _signature(inputs),
    )


class _CapturedAttention:
    def __init__(self, impl, inputs):
        self.impl = impl  # Retain the exact prepared configuration and its lifetime.
        self.signature = _execution_signature(inputs)
        self.device = inputs[0].device
        self.buffers = []
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            self.inputs = _owned_copy(inputs, self.device, self.buffers)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    impl.forward(*self.inputs)
            # Lazy kernel/workspace initialization must finish before capture.
            stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = impl.forward(*self.inputs)
            current.wait_stream(stream)
            self.done = torch.cuda.Event()
            self.done.record(current)
        self.replays = 0

    def replay(self, inputs):
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            # Serialize reuse if the caller changes streams between requests.
            current.wait_event(self.done)
            for static, live in zip(self.buffers, _tensors(inputs)):
                static.copy_(live)
                static.record_stream(current)
            self.output.record_stream(current)
            self.graph.replay()
            # The graph owns output storage. Returning it directly invalidates a
            # previous request's result on the next replay, including across steps.
            output = self.output.clone()
            self.done.record(current)
        self.replays += 1
        return output

    def close(self) -> None:
        """Complete queued replay before releasing the graph-private allocation pool."""
        done = getattr(self, "done", None)
        if done is not None:
            done.synchronize()
            self.done = None

    def __del__(self):
        # record_stream protects external buffers; the graph also owns private
        # intermediates. Wait on teardown so neither can be reused mid-replay.
        self.close()


class SigmaAttentionGraphCache:
    """At most one fixed Q/K/V + metadata signature per prepared implementation.

    First eligible sigma denoise dispatch pre-captures every compatible prepared
    object, including the baseline for gaps. New signatures use eager attention,
    not a new graph/compile entry. No backend computation reads sigma. Capture
    errors propagate: kernel/model errors must not become silent eager fallback.
    """

    def __init__(self) -> None:
        self.entries: dict[int, _CapturedAttention] = {}
        self._candidate_collection = None
        self._registered: dict[int, Any] = {}
        self._pending: dict[int, Any] = {}

    def _register_candidates(self, candidates) -> None:
        if candidates is self._candidate_collection:
            return
        self._candidate_collection = candidates
        for impl in candidates:
            identity = id(impl)
            if identity in self._registered:
                continue
            self._registered[identity] = impl
            # Type eligibility is permanent. Metadata and dtype eligibility are
            # checked for pending objects until their first valid capture.
            if capture_eligibility(impl, None):
                self._pending[identity] = impl

    @torch.compiler.disable
    def run(self, selected, candidates, query, key, value, metadata):
        inputs = (query, key, value, metadata)
        if (
            not query.is_cuda
            or torch.version.hip is not None
            or torch.is_grad_enabled()
            or torch.cuda.is_current_stream_capturing()
            or any(t.device != query.device or t.requires_grad for t in (query, key, value))
        ):
            return selected.forward(*inputs)
        from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

        self._register_candidates(candidates)
        for identity, impl in tuple(self._pending.items()):
            graph_metadata = _graph_metadata(impl, metadata)
            if graph_metadata is _INELIGIBLE:
                continue
            if type(impl) is not SDPAImpl and (
                query.dtype not in (torch.float16, torch.bfloat16)
                or (_PACKED_KEYS.intersection(getattr(graph_metadata, "extra", {})) and query.shape[0] != 1)
            ):
                continue
            graph_inputs = (query, key, value, graph_metadata)
            if any(t.device != query.device or t.requires_grad for t in _tensors(graph_inputs)):
                continue
            self.entries[identity] = _CapturedAttention(impl, graph_inputs)
            del self._pending[identity]
        entry = self.entries.get(id(selected))
        graph_metadata = _graph_metadata(selected, metadata)
        if entry is None or graph_metadata is _INELIGIBLE:
            return selected.forward(*inputs)
        graph_inputs = (query, key, value, graph_metadata)
        if any(t.device != query.device or t.requires_grad for t in _tensors(graph_inputs)):
            return selected.forward(*inputs)
        if entry.signature != _execution_signature(graph_inputs):
            return selected.forward(*inputs)
        return entry.replay(graph_inputs)


def enable_sigma_attention_graphs(model, config, device) -> bool:
    """Enable only generic runner compile routes with stationary local attention.

    Pipeline-owned setup_compile routes are not enabled here. Offload/HSDP,
    scheduler-paged and ring attention retain their existing eager boundaries.
    """
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.offloader.config import offload_enabled

    if (
        device is None
        or torch.device(device).type != "cuda"
        or config.enforce_eager
        or getattr(config, "diffusion_attention_schedule", None) is None
        or not torch.cuda.is_available()
        or torch.version.hip is not None
        or config.parallel_config.use_hsdp
        or offload_enabled(config)
    ):
        return False
    enabled = False
    for module in model.modules():
        if (
            isinstance(module, Attention)
            and module._schedule_configured
            and not module._has_custom_attention
            and not module.use_ring
            and module.paged_kv_cache_role is None
            and any(
                capture_eligibility(impl, None)
                for impl in (module.attention, *(record.impl for record in module._schedule_candidates.values()))
            )
        ):
            module._sigma_graph_candidates = tuple(
                {
                    id(impl): impl
                    for impl in (module.attention, *(record.impl for record in module._schedule_candidates.values()))
                }.values()
            )
            module._sigma_graph_cache = SigmaAttentionGraphCache()
            enabled = True
    return enabled
