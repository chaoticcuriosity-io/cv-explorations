"""Image and video I/O, normalised to RGB ``numpy`` arrays.

Everything in cvkit speaks HxWx3 uint8 **RGB** arrays. OpenCV defaults to BGR,
which is a classic source of "why are my colours wrong" bugs, so conversions are
funnelled through here and never sprinkled across modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an HxWx3 uint8 RGB array."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def save_image(path: str | Path, image_rgb: np.ndarray) -> Path:
    """Save an HxWx3 uint8 RGB array to ``path`` (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)
    return path


def list_images(directory: str | Path) -> list[Path]:
    """Return image files in a directory, sorted, filtered by known suffixes."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )


def iter_video_frames(
    path: str | Path, *, stride: int = 1, max_frames: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, rgb_frame)`` from a video, every ``stride`` frames.

    Used by the tracking module; the BGR->RGB conversion happens once, here.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        idx, yielded = 0, 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
