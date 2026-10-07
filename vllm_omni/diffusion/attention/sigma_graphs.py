# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-backend CUDA graph windows selected by normalized sigma.

Selection does not require CUDA. Capture does, and callers skip it when CUDA is absent.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SigmaGraphWindow:
    backend: str
    low: float
    high: float
    graph: Any


class SigmaCudaGraphTable:
    """Register one graph per backend and sigma window, then select it at runtime."""

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
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graph capture requires CUDA")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay()
        self.register(backend, low, high, graph)
        return graph
