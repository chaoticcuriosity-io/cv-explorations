"""Vision-Language Model adapter (Qwen2.5-VL via the image-text-to-text pipeline).

A VLM fuses an image encoder with a language model, so a *single* model answers open-ended
questions about an image — caption, reason, count, read text — chosen by prompt, not by a
task-specific head. This is the most general capability in the curriculum and the closest to
"understanding". Output is free-form text, so evaluation here is qualitative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class VlmMeta:
    key: str
    display: str
    license: str


class QwenVL:
    def __init__(self, model_id: str, meta: VlmMeta, device: torch.device):
        from transformers import pipeline

        self.pipe = pipeline(
            "image-text-to-text", model=model_id,
            torch_dtype=torch.float16, device=0 if device.type == "cuda" else -1,
        )
        self.meta = meta

    def ask(self, image, question: str, max_new_tokens: int = 128) -> str:
        from PIL import Image

        img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question},
        ]}]
        out = self.pipe(text=messages, max_new_tokens=max_new_tokens)
        return out[0]["generated_text"][-1]["content"].strip()


def build_vlm(device: torch.device) -> QwenVL:
    return QwenVL(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        VlmMeta("qwen2.5-vl-3b", "Qwen2.5-VL 3B", "Apache-2.0"),
        device,
    )
