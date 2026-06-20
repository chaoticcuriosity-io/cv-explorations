"""Annotation and figure helpers, built on roboflow ``supervision``.

Because every detector adapter returns ``sv.Detections``, these annotators work
unchanged whether the boxes came from YOLO26, RF-DETR, or a transformers model.
``supervision`` is imported lazily so cvkit stays importable on a minimal env.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # avoid a hard import at module load
    import supervision as sv


def annotate_detections(
    image_rgb: np.ndarray,
    detections: "sv.Detections",
    *,
    labels: Sequence[str] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw boxes (+ optional labels) onto a copy of ``image_rgb``."""
    import supervision as sv

    box = sv.BoxAnnotator(thickness=thickness)
    out = box.annotate(scene=image_rgb.copy(), detections=detections)
    if labels is not None:
        label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        out = label.annotate(scene=out, detections=detections, labels=list(labels))
    return out


def save_image_grid(
    images: Sequence[np.ndarray],
    path: str | Path,
    *,
    cols: int = 3,
    titles: Sequence[str] | None = None,
    figsize_per: tuple[float, float] = (4.0, 4.0),
) -> Path:
    """Save a grid of RGB images as a single figure (for report result panels)."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe for CI / scripts
    import matplotlib.pyplot as plt

    n = len(images)
    if n == 0:
        raise ValueError("save_image_grid: no images to plot")
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows)
    )
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
