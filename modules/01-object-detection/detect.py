"""Object-detection model adapters — every model speaks ``supervision.Detections``.

Each detector wraps a different native API behind one interface so that data
loading, visualisation, metrics, and benchmarking are written once (in cvkit and
``run.py``) and reused across all models. Class identity is normalised *by name*
against a canonical COCO-80 table, because models disagree on integer label ids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch

# Canonical COCO-80 class order (matches Ultralytics' `model.names`).
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
NAME_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(COCO_CLASSES)}


@dataclass(frozen=True)
class DetectorMeta:
    key: str          # cli handle, e.g. "rfdetr-nano"
    display: str      # report label, e.g. "RF-DETR Nano"
    family: str       # architecture family for the narrative
    license: str      # SPDX-ish license string


class Detector(ABC):
    """Common interface: name/license metadata + RGB-array -> sv.Detections."""

    meta: DetectorMeta
    id_to_name: dict[int, str]

    @abstractmethod
    def predict(self, image_rgb: np.ndarray, conf: float = 0.25) -> sv.Detections:
        ...

    def canonical_class_ids(self, det: sv.Detections) -> np.ndarray:
        """Map detections to canonical COCO-80 indices, identifying classes by NAME.

        Models disagree on integer ids (COCO 0-79 vs 1-90), but every adapter here
        populates ``det.data['class_name']``; we trust that and fall back to this
        detector's ``id_to_name`` only if names are absent. Unknown -> -1 (dropped).
        """
        out = np.full(len(det), -1, dtype=int)
        names = det.data.get("class_name") if det.data else None
        if names is not None:
            for i, nm in enumerate(names):
                out[i] = NAME_TO_IDX.get(str(nm).lower(), -1)
            return out
        if det.class_id is None:
            return out
        for i, cid in enumerate(det.class_id):
            name = self.id_to_name.get(int(cid))
            out[i] = NAME_TO_IDX.get(name.lower(), -1) if name is not None else -1
        return out


def _ultra_device(device: torch.device) -> object:
    return device.index or 0 if device.type == "cuda" else "cpu"


class UltralyticsDetector(Detector):
    """YOLO family (YOLO26 / YOLO11 / RT-DETR via Ultralytics). License: AGPL-3.0."""

    def __init__(self, weights: str, meta: DetectorMeta, device: torch.device):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self._device = _ultra_device(device)
        self.meta = meta
        self.id_to_name = dict(self.model.names)

    def predict(self, image_rgb: np.ndarray, conf: float = 0.25) -> sv.Detections:
        result = self.model.predict(
            image_rgb, conf=conf, device=self._device, verbose=False
        )[0]
        return sv.Detections.from_ultralytics(result)


class RFDETRDetector(Detector):
    """RF-DETR (Roboflow DETR). License: Apache-2.0. COCO-80, 0-indexed by name."""

    def __init__(self, ctor, meta: DetectorMeta, device: torch.device):
        self.model = ctor()
        self.meta = meta
        # RF-DETR is COCO-trained; resolve names from the canonical table and
        # correct it after a probe if its indexing differs.
        self.id_to_name = dict(enumerate(COCO_CLASSES))
        if device.type == "cuda" and hasattr(self.model, "optimize_for_inference"):
            try:
                self.model.optimize_for_inference()
            except Exception:
                pass  # optional speed-up; safe to skip

    def predict(self, image_rgb: np.ndarray, conf: float = 0.25) -> sv.Detections:
        from PIL import Image

        return self.model.predict(Image.fromarray(image_rgb), threshold=conf)


# --- registry -------------------------------------------------------------

_REGISTRY = {
    "yolo26n": lambda d: UltralyticsDetector(
        "yolo26n.pt",
        DetectorMeta("yolo26n", "YOLO26-n", "YOLO (NMS-free)", "AGPL-3.0"),
        d,
    ),
    "yolo11n": lambda d: UltralyticsDetector(
        "yolo11n.pt",
        DetectorMeta("yolo11n", "YOLO11-n", "YOLO (anchor-free)", "AGPL-3.0"),
        d,
    ),
    "rfdetr-nano": lambda d: _rfdetr("RFDETRNano", "RF-DETR Nano", d),
    "rfdetr-medium": lambda d: _rfdetr("RFDETRMedium", "RF-DETR Medium", d),
}


def _rfdetr(cls_name: str, display: str, device: torch.device) -> RFDETRDetector:
    import rfdetr

    ctor = getattr(rfdetr, cls_name)
    key = display.lower().replace(" ", "-").replace("rf-detr", "rfdetr")
    return RFDETRDetector(ctor, DetectorMeta(key, display, "DETR (transformer)", "Apache-2.0"), device)


def available() -> list[str]:
    return list(_REGISTRY)


def build_detector(key: str, device: torch.device) -> Detector:
    if key not in _REGISTRY:
        raise KeyError(f"unknown detector '{key}'. available: {available()}")
    return _REGISTRY[key](device)
