"""Image-embedding adapters for similarity search (DINOv2 vs SigLIP).

An *embedding* maps an image to a vector such that visually/semantically similar images land
nearby. This powers retrieval, dedup, clustering, and few-shot recognition — all without a
classifier head. Each adapter returns an L2-normalised vector, so cosine similarity is a dot
product. The contrast: **DINOv2** is self-supervised (no labels, strong on visual structure);
**SigLIP** is image-text contrastive (semantically aligned to language).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EmbedMeta:
    key: str
    display: str
    kind: str
    license: str


class Dinov2Embedder:
    def __init__(self, model_id: str, meta: EmbedMeta, device: torch.device):
        from transformers import AutoImageProcessor, AutoModel

        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.meta = meta

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        inp = self.proc(images=Image.fromarray(image_rgb), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
        v = out.pooler_output[0] if out.pooler_output is not None else out.last_hidden_state[0, 0]
        v = v / v.norm()
        return v.float().cpu().numpy()


class SiglipEmbedder:
    def __init__(self, model_id: str, meta: EmbedMeta, device: torch.device):
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.device = device
        self.meta = meta

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        inp = self.proc(images=Image.fromarray(image_rgb), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inp)
        # transformers v5 returns a pooled-output object here, not a bare tensor.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            v = out.pooler_output[0]
        elif hasattr(out, "last_hidden_state"):
            v = out.last_hidden_state[0].mean(0)
        else:
            v = out[0]
        v = v / v.norm()
        return v.float().cpu().numpy()


_REGISTRY = {
    "dinov2-base": lambda d: Dinov2Embedder(
        "facebook/dinov2-base", EmbedMeta("dinov2-base", "DINOv2 Base", "self-supervised", "Apache-2.0"), d),
    "siglip2-base": lambda d: SiglipEmbedder(
        "google/siglip2-base-patch16-224", EmbedMeta("siglip2-base", "SigLIP 2 Base", "image-text", "Apache-2.0"), d),
}


def available() -> list[str]:
    return list(_REGISTRY)


def build_embedder(key: str, device: torch.device):
    return _REGISTRY[key](device)
