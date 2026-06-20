"""Image-classification adapters: supervised (timm) vs zero-shot (SigLIP/CLIP).

All adapters expose ``scores(image_rgb) -> np.ndarray`` of length 10 over the
Imagenette classes, so accuracy/latency code in ``run.py`` is written once. The
difference in *how* those scores arise — a trained head vs text-image similarity —
is the lesson of this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

# Imagenette = 10 ImageNet classes. These are their ImageNet-1k logit indices,
# in torchvision's Imagenette label order (0-9).
IMAGENET_IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
CLASS_NAMES = [
    "tench", "English springer", "cassette player", "chainsaw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]
PROMPTS = [f"a photo of a {n}." for n in CLASS_NAMES]


@dataclass(frozen=True)
class ClassifierMeta:
    key: str
    display: str
    paradigm: str   # "Supervised" | "Zero-shot"
    license: str


class Classifier(ABC):
    meta: ClassifierMeta

    @abstractmethod
    def scores(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a length-10 score vector over the Imagenette classes."""

    def predict(self, image_rgb: np.ndarray) -> int:
        return int(self.scores(image_rgb).argmax())


class TimmClassifier(Classifier):
    """Supervised ImageNet-1k model; we read off the 10 relevant logits."""

    def __init__(self, model_name: str, meta: ClassifierMeta, device: torch.device):
        import timm

        self.model = timm.create_model(model_name, pretrained=True).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.tf = timm.data.create_transform(**cfg)
        self.device = device
        self.meta = meta

    def scores(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        x = self.tf(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0]
        return logits[IMAGENET_IDX].float().cpu().numpy()


class SiglipZeroShot(Classifier):
    """SigLIP 2 zero-shot: rank image against text prompts. No head, no training on this set."""

    def __init__(self, model_id: str, meta: ClassifierMeta, device: torch.device):
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.device = device
        self.meta = meta

    def scores(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        inp = self.proc(
            text=PROMPTS, images=Image.fromarray(image_rgb),
            return_tensors="pt", padding="max_length",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
        return out.logits_per_image[0].float().cpu().numpy()


class OpenClipZeroShot(Classifier):
    """OpenCLIP zero-shot: cosine similarity to pre-encoded text features."""

    def __init__(self, model_name: str, pretrained: str, meta: ClassifierMeta, device: torch.device):
        import open_clip

        self.model, _, self.pre = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        tok = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            t = self.model.encode_text(tok(PROMPTS).to(device))
            self.text_feat = t / t.norm(dim=-1, keepdim=True)
        self.device = device
        self.meta = meta

    def scores(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        x = self.pre(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ie = self.model.encode_image(x)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            sim = (ie @ self.text_feat.T)[0]
        return sim.float().cpu().numpy()


_REGISTRY = {
    "convnext-tiny": lambda d: TimmClassifier(
        "convnext_tiny.fb_in1k",
        ClassifierMeta("convnext-tiny", "ConvNeXt-Tiny", "Supervised", "Apache-2.0"), d),
    "siglip2-base": lambda d: SiglipZeroShot(
        "google/siglip2-base-patch16-224",
        ClassifierMeta("siglip2-base", "SigLIP 2 Base", "Zero-shot", "Apache-2.0"), d),
    "openclip-vitb32": lambda d: OpenClipZeroShot(
        "ViT-B-32", "laion2b_s34b_b79k",
        ClassifierMeta("openclip-vitb32", "OpenCLIP ViT-B/32", "Zero-shot", "MIT"), d),
}


def available() -> list[str]:
    return list(_REGISTRY)


def build_classifier(key: str, device: torch.device) -> Classifier:
    if key not in _REGISTRY:
        raise KeyError(f"unknown classifier '{key}'. available: {available()}")
    return _REGISTRY[key](device)
