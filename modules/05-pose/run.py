"""Pose engine: YOLO26-pose vs YOLO11-pose — skeleton overlays + latency.

uv run python modules/05-pose/run.py --images 4
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import supervision as sv
import torch

import pose
from cvkit import describe_device, get_device
from cvkit.io import load_image
from cvkit.report import markdown_table, write_metrics_md
from cvkit.timing import benchmark
from cvkit.viz import save_image_grid

HERE = Path(__file__).parent
RESULTS = HERE / "results"
COCO = r"C:/Users/dgbal/fiftyone/coco-2017/validation/data/*.jpg"

_EDGE = sv.EdgeAnnotator(thickness=2)
_VERT = sv.VertexAnnotator(radius=4)


def annotate(image_rgb, kp):
    out = _EDGE.annotate(image_rgb.copy(), kp)
    return _VERT.annotate(out, kp)


def find_people_images(scanner, n, scan_limit=80):
    paths = sorted(glob.glob(COCO))[:scan_limit]
    picks = []
    for p in paths:
        img = load_image(p)
        kp = scanner.predict(img)
        if kp.xy is not None and len(kp.xy) >= 1:
            picks.append(p)
        if len(picks) >= n:
            break
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="yolo26-pose,yolo11-pose")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {describe_device(device)}")

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    models = [pose.build_pose(k, device) for k in keys]
    paths = find_people_images(models[0], args.images)
    images = [load_image(p) for p in paths]
    print(f"using {len(images)} people images")
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in models:
        annotated, counts = [], []
        for img in images:
            kp = m.predict(img)
            counts.append(0 if kp.xy is None else len(kp.xy))
            annotated.append(annotate(img, kp))
        save_image_grid(annotated, RESULTS / f"samples_{m.meta.key}.png",
                        cols=min(len(annotated), 2), titles=[m.meta.display] * len(annotated))
        lat = benchmark(lambda: m.predict(images[0]), device=device, warmup=3, runs=20)
        rows.append({"display": m.meta.display, "license": m.meta.license,
                     "latency_ms": lat.as_row()["mean_ms"], "fps": lat.as_row()["fps"],
                     "avg_people": round(float(np.mean(counts)), 1)})
        print(f"  {m.meta.key}: {lat.as_row()['mean_ms']}ms, {np.mean(counts):.1f} people/img")

    headers = ["Model", "License", "Latency (ms)", "FPS", "People/img"]
    table = [[r["display"], r["license"], r["latency_ms"], r["fps"], r["avg_people"]] for r in rows]
    notes = (
        f"*17 COCO keypoints per person, single-stage (bottom-up). Latency = mean warmed-up "
        f"single-image forward on `{describe_device(device)}`. Qualitative + latency; COCO "
        f"keypoint OKS-AP is a deeper follow-up.*"
    )
    write_metrics_md(RESULTS / "metrics.md", title="Pose estimation: speed", headers=headers, rows=table, notes=notes)
    (RESULTS / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + markdown_table(headers, table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
