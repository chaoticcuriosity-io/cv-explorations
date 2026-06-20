"""Thin, well-typed wrappers over standard metric libraries.

We deliberately wrap ``torchmetrics`` / ``pycocotools`` rather than re-implement
metrics: the point of a *benchmark* repo is that the numbers are trustworthy and
comparable to published results, so we use the canonical implementations.
"""

from __future__ import annotations

from typing import Sequence

import torch


def topk_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ks: Sequence[int] = (1, 5)
) -> dict[int, float]:
    """Top-k accuracy for classification.

    ``logits``: (N, C) scores; ``targets``: (N,) int labels. Returns {k: acc%}.
    """
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (N, maxk)
    correct = pred.eq(targets.view(-1, 1))
    out: dict[int, float] = {}
    for k in ks:
        kk = min(k, maxk)
        out[k] = 100.0 * correct[:, :kk].any(dim=1).float().mean().item()
    return out


def detection_map(preds: list[dict], targets: list[dict]) -> dict[str, float]:
    """COCO-style mean Average Precision via ``torchmetrics``.

    Each ``preds[i]`` is ``{"boxes": (M,4) xyxy, "scores": (M,), "labels": (M,)}``
    and each ``targets[i]`` is ``{"boxes": (G,4) xyxy, "labels": (G,)}`` (tensors).
    Returns the headline COCO metrics (mAP@[.5:.95], mAP@.5, mAP@.75).
    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    res = metric.compute()
    return {
        "mAP": float(res["map"]),
        "mAP_50": float(res["map_50"]),
        "mAP_75": float(res["map_75"]),
        "mAP_small": float(res["map_small"]),
        "mAP_large": float(res["map_large"]),
    }
