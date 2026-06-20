"""Device, dtype, and reproducibility helpers.

One place to decide *where* tensors run and in *what* precision, so every module
behaves consistently on the local RTX 3090 Ti, on CPU, or on the DGX Spark.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the best available device, honouring a preference.

    Falls back cuda -> mps -> cpu so the same code runs on the 3090 Ti, an
    Apple laptop, or a CI runner with no GPU.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Pick a sensible autocast dtype: bf16 if the GPU supports it, else fp16, else fp32.

    Ampere (3090 Ti) and Blackwell (DGX Spark) both support bf16, which is more
    numerically forgiving than fp16 for inference.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


@dataclass(frozen=True)
class DeviceInfo:
    kind: str
    name: str
    total_vram_gb: float | None
    capability: str | None
    torch_version: str
    cuda_version: str | None

    def __str__(self) -> str:  # human-friendly one-liner for reports/logs
        if self.kind == "cuda":
            return (
                f"{self.name} | {self.total_vram_gb:.1f} GB | sm_{self.capability} | "
                f"torch {self.torch_version} / CUDA {self.cuda_version}"
            )
        return f"{self.name} | torch {self.torch_version}"


def describe_device(device: torch.device | None = None) -> DeviceInfo:
    """Collect the hardware/software facts a benchmark report should record."""
    device = device or get_device()
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        return DeviceInfo(
            kind="cuda",
            name=props.name,
            total_vram_gb=props.total_memory / 1024**3,
            capability=f"{props.major}{props.minor}",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
    return DeviceInfo(
        kind=device.type,
        name=device.type.upper(),
        total_vram_gb=None,
        capability=None,
        torch_version=torch.__version__,
        cuda_version=None,
    )


def set_seed(seed: int = 0, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch. Set ``deterministic`` for exact repeats (slower)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
