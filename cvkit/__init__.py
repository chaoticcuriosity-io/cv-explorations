"""cvkit — the shared toolkit that keeps each CV module thin.

Modules under ``modules/`` import from here instead of re-implementing device
selection, image/video I/O, latency benchmarking, annotation, metrics, and
report emission. The guiding rule: write each cross-cutting concern *once*.
"""

from __future__ import annotations

__version__ = "0.1.0"

from cvkit.devices import describe_device, get_device, set_seed
from cvkit.timing import LatencyStats, benchmark

__all__ = [
    "__version__",
    "get_device",
    "describe_device",
    "set_seed",
    "benchmark",
    "LatencyStats",
]
