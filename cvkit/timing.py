"""Latency / throughput benchmarking with correct GPU synchronisation.

The subtlety this module hides: CUDA kernels are *asynchronous*. Naively timing
a model call measures only the time to enqueue work, not to finish it. We warm
up first (to trigger lazy CUDA init, cuDNN autotuning, and weight paging) and
call ``torch.cuda.synchronize()`` around each timed run so the numbers are real.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class LatencyStats:
    runs: int
    mean_ms: float
    p50_ms: float
    p90_ms: float

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else float("inf")

    def as_row(self) -> dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "fps": round(self.fps, 1),
        }


def benchmark(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = 3,
    runs: int = 20,
) -> LatencyStats:
    """Time ``fn`` over ``runs`` iterations after ``warmup`` untimed iterations.

    ``fn`` should be a zero-arg closure performing one unit of work (e.g. one
    image's forward pass). Returns wall-clock latency stats; divide work per call
    consistently so ``fps`` is comparable across models.
    """
    is_cuda = device.type == "cuda"

    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    p50 = samples[int(0.50 * (n - 1))]
    p90 = samples[int(0.90 * (n - 1))]
    return LatencyStats(runs=runs, mean_ms=mean, p50_ms=p50, p90_ms=p90)
