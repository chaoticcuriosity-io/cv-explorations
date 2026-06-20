"""Classical CV with OpenCV: edges, contours, feature matching, optical flow.

The pre-deep-learning toolkit — hand-designed algorithms that are fast, transparent, and
need no training data or GPU. They still matter: calibration, SLAM front-ends, stabilisation,
and cheap preprocessing all lean on these. This module is the contrast that makes the rest of
the curriculum legible.

uv run python modules/09-classical/run.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cvkit import describe_device, get_device
from cvkit.io import load_image
from cvkit.report import markdown_table, write_metrics_md
from cvkit.timing import benchmark

HERE = Path(__file__).parent
RESULTS = HERE / "results"
COCO = r"C:/Users/dgbal/fiftyone/coco-2017/validation/data/*.jpg"
VIDEO = "data/people-walking.mp4"


def two_frames(width=960):
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    for target in (0, 6):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, f = cap.read()
        if ok:
            h = int(f.shape[0] * width / f.shape[1])
            frames.append(cv2.cvtColor(cv2.resize(f, (width, h)), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def panel(items, titles, path, cols=3):
    n = len(items)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    for ax, im, t in zip(np.atleast_1d(axes).ravel(), items, titles):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def flow_to_rgb(flow):
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def main() -> int:
    device = get_device()
    print(f"device: {describe_device(device)}  (classical CV runs on CPU)")
    cpu = torch.device("cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1) Edges + contours on a still image.
    img = load_image(sorted(glob.glob(COCO))[0])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
    panel([img, edges, contour_img],
          ["Original", "Canny edges", f"Contours ({len(cnts)} found)"],
          RESULTS / "edges_contours.png", cols=3)

    # 2) ORB feature matching between two frames.
    f0, f1 = two_frames()
    g0, g1 = (cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in (f0, f1))
    orb = cv2.ORB_create(nfeatures=500)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d0, d1), key=lambda m: m.distance)[:40]
    match_img = cv2.drawMatches(f0, k0, f1, k1, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    panel([match_img], [f"ORB feature matches ({len(k0)} keypoints; best 40 shown)"],
          RESULTS / "features.png", cols=1)

    # 3) Dense optical flow between the two frames.
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    panel([f0, flow_to_rgb(flow)],
          ["Frame t", "Dense optical flow (hue = direction, brightness = speed)"],
          RESULTS / "optical_flow.png", cols=2)

    # Timings — note how fast these are vs the deep models elsewhere.
    rows = [
        ["Canny edges", round(benchmark(lambda: cv2.Canny(gray, 80, 200), device=cpu, warmup=3, runs=50).mean_ms, 2)],
        ["ORB detect+match", round(benchmark(lambda: bf.match(orb.detectAndCompute(g0, None)[1], d1), device=cpu, warmup=2, runs=20).mean_ms, 2)],
        ["Farneback optical flow", round(benchmark(lambda: cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0), device=cpu, warmup=2, runs=10).mean_ms, 2)],
    ]
    notes = (
        "*All on CPU, no GPU, no training, no model download. Compare these millisecond timings "
        "to the deep models in modules 01–08 — classical CV is essentially free, which is why it "
        "still underpins calibration, SLAM front-ends, and stabilisation.*"
    )
    write_metrics_md(RESULTS / "metrics.md", title="Classical CV: operation timings (CPU)",
                     headers=["Operation", "Latency (ms)"], rows=rows, notes=notes)
    (RESULTS / "metrics.json").write_text(json.dumps({op: ms for op, ms in rows}, indent=2), encoding="utf-8")
    print("\n" + markdown_table(["Operation", "Latency (ms)"], rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
