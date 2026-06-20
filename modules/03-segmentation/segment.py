"""Segmentation adapters: closed-set instance masks (YOLO-seg) vs promptable SAM 2.1.

YOLO-seg predicts a mask per detected COCO object (fixed label set). SAM 2.1 segments
*whatever you prompt* — here we feed it boxes (e.g. from a detector) and it returns a
refined mask, with no notion of class. Both return ``supervision.Detections`` carrying
``.mask``, so annotation and timing are shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch


@dataclass(frozen=True)
class SegMeta:
    key: str
    display: str
    family: str
    license: str


def _dev(device: torch.device):
    return device.index or 0 if device.type == "cuda" else "cpu"


class YoloSeg:
    """Ultralytics instance segmentation (YOLO26-seg / YOLO11-seg). AGPL-3.0."""

    def __init__(self, weights: str, meta: SegMeta, device: torch.device):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = _dev(device)
        self.meta = meta

    def predict(self, image_rgb: np.ndarray, conf: float = 0.3) -> sv.Detections:
        r = self.model.predict(image_rgb, conf=conf, device=self.device, verbose=False)[0]
        return sv.Detections.from_ultralytics(r)


class Sam21:
    """SAM 2.1 — promptable segmentation. Apache-2.0. Here: box prompts -> masks."""

    def __init__(self, weights: str, meta: SegMeta, device: torch.device):
        from ultralytics import SAM

        self.model = SAM(weights)
        self.device = _dev(device)
        self.meta = meta

    def predict_boxes(self, image_rgb: np.ndarray, boxes: np.ndarray) -> sv.Detections:
        if boxes is None or len(boxes) == 0:
            return sv.Detections.empty()
        r = self.model.predict(
            image_rgb, bboxes=boxes.tolist(), device=self.device, verbose=False
        )[0]
        return sv.Detections.from_ultralytics(r)


_SEG = {
    "yolo26-seg": lambda d: YoloSeg(
        "yolo26n-seg.pt", SegMeta("yolo26-seg", "YOLO26-n-seg", "instance (YOLO)", "AGPL-3.0"), d),
    "yolo11-seg": lambda d: YoloSeg(
        "yolo11n-seg.pt", SegMeta("yolo11-seg", "YOLO11-n-seg", "instance (YOLO)", "AGPL-3.0"), d),
}


def available() -> list[str]:
    return list(_SEG)


def build_segmenter(key: str, device: torch.device) -> YoloSeg:
    return _SEG[key](device)


def build_sam(device: torch.device) -> Sam21:
    return Sam21("sam2.1_b.pt", SegMeta("sam2.1", "SAM 2.1 (base)", "promptable", "Apache-2.0"), device)
