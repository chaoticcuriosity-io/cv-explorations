"""Human pose adapters (YOLO-pose) — 17 COCO keypoints per person.

Pose estimation locates body joints (a skeleton) per person. YOLO-pose is *single-stage,
bottom-up*: one network finds people and their keypoints together. The classic alternative
is *top-down* (detect each person, then a dedicated pose net per crop, e.g. ViTPose/RTMPose),
which is usually more accurate but scales with the number of people. Adapters return
``supervision.KeyPoints``.
"""

from __future__ import annotations

from dataclasses import dataclass

import supervision as sv
import torch


@dataclass(frozen=True)
class PoseMeta:
    key: str
    display: str
    license: str


class YoloPose:
    def __init__(self, weights: str, meta: PoseMeta, device: torch.device):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device.index or 0 if device.type == "cuda" else "cpu"
        self.meta = meta

    def predict(self, image_rgb, conf: float = 0.4) -> sv.KeyPoints:
        r = self.model.predict(image_rgb, conf=conf, device=self.device, verbose=False)[0]
        return sv.KeyPoints.from_ultralytics(r)


_REGISTRY = {
    "yolo26-pose": lambda d: YoloPose("yolo26n-pose.pt", PoseMeta("yolo26-pose", "YOLO26-n-pose", "AGPL-3.0"), d),
    "yolo11-pose": lambda d: YoloPose("yolo11n-pose.pt", PoseMeta("yolo11-pose", "YOLO11-n-pose", "AGPL-3.0"), d),
}


def available() -> list[str]:
    return list(_REGISTRY)


def build_pose(key: str, device: torch.device) -> YoloPose:
    return _REGISTRY[key](device)
