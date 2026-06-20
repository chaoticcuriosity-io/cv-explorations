"""Monocular depth adapters (Depth Anything V2).

Monocular depth estimates distance-per-pixel from a *single* RGB image — no stereo, no
LiDAR. Depth Anything V2 predicts *relative* (inverse) depth: larger values = nearer.
Each adapter returns an HxW float array aligned to the input image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DepthMeta:
    key: str
    display: str
    license: str


class DepthAnythingV2:
    def __init__(self, model_id: str, meta: DepthMeta, device: torch.device):
        from transformers import pipeline

        self.pipe = pipeline(
            "depth-estimation", model=model_id, device=0 if device.type == "cuda" else -1
        )
        self.meta = meta

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        out = self.pipe(Image.fromarray(image_rgb))
        return out["predicted_depth"].float().cpu().numpy()


_REGISTRY = {
    "dav2-small": lambda d: DepthAnythingV2(
        "depth-anything/Depth-Anything-V2-Small-hf",
        DepthMeta("dav2-small", "Depth Anything V2 (Small)", "Apache-2.0"), d),
    "dav2-base": lambda d: DepthAnythingV2(
        "depth-anything/Depth-Anything-V2-Base-hf",
        DepthMeta("dav2-base", "Depth Anything V2 (Base)", "Apache-2.0"), d),
}


def available() -> list[str]:
    return list(_REGISTRY)


def build_depth(key: str, device: torch.device) -> DepthAnythingV2:
    return _REGISTRY[key](device)
