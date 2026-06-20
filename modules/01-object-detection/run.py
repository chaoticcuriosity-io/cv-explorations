"""Benchmark engine for the object-detection module.

Runs one or more detectors over a COCO-val subset, computes COCO mAP and warmed-up
latency on the local GPU, and writes a metrics table + annotated sample figures into
``results/`` (which the report embeds, so the site renders without a GPU).

Examples
--------
uv run python modules/01-object-detection/run.py --models yolo26n,rfdetr-nano --limit 25
uv run python modules/01-object-detection/run.py --models yolo26n,yolo11n,rfdetr-nano,rfdetr-medium
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import detect  # same directory
from cvkit import describe_device, get_device
from cvkit.report import markdown_table, write_metrics_md
from cvkit.timing import benchmark
from cvkit.viz import annotate_detections, save_image_grid
from cvkit.io import load_image

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# mAP needs the full PR curve -> infer at a near-zero threshold.
MAP_CONF = 0.001
# qualitative figures should look clean -> filter to a confident subset.
SHOW_CONF = 0.30


def load_coco_eval(max_samples: int, limit: int | None):
    """Return per-image dicts with image path + ground-truth boxes/labels (canonical)."""
    from cvkit.data import load_coco_val_subset

    ds = load_coco_val_subset(max_samples=max_samples)
    ds.compute_metadata()
    samples = []
    for s in ds:
        w, h = s.metadata.width, s.metadata.height
        boxes, labels = [], []
        gt = s["ground_truth"]
        if gt is not None:
            for d in gt.detections:
                if getattr(d, "iscrowd", 0) in (1, True):
                    continue  # torchmetrics has no crowd handling; match COCO's ignore
                idx = detect.NAME_TO_IDX.get(d.label.lower(), -1)
                if idx < 0:
                    continue
                x, y, bw, bh = d.bounding_box  # relative xywh
                boxes.append([x * w, y * h, (x + bw) * w, (y + bh) * h])
                labels.append(idx)
        samples.append(
            {
                "path": s.filepath,
                "boxes": np.asarray(boxes, dtype=float).reshape(-1, 4),
                "labels": np.asarray(labels, dtype=int),
            }
        )
        if limit and len(samples) >= limit:
            break
    return samples


def pred_to_canonical(det_obj, model, conf: float):
    """Filter a detection set to canonical COCO classes above ``conf``."""
    canon = model.canonical_class_ids(det_obj)
    scores = (
        det_obj.confidence
        if det_obj.confidence is not None
        else np.ones(len(det_obj), dtype=float)
    )
    keep = (canon >= 0) & (scores >= conf)
    boxes = det_obj.xyxy[keep] if len(det_obj) else np.zeros((0, 4))
    return boxes, scores[keep], canon[keep]


def evaluate_model(key: str, samples, device) -> dict:
    model = detect.build_detector(key, device)
    preds, targets = [], []
    for s in samples:
        img = load_image(s["path"])
        det_obj = model.predict(img, conf=MAP_CONF)
        boxes, scores, labels = pred_to_canonical(det_obj, model, MAP_CONF)
        preds.append(
            {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "scores": torch.as_tensor(scores, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            }
        )
        targets.append(
            {
                "boxes": torch.as_tensor(s["boxes"], dtype=torch.float32),
                "labels": torch.as_tensor(s["labels"], dtype=torch.int64),
            }
        )

    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    res = metric.compute()

    # Warmed-up latency on a representative image.
    rep = load_image(samples[0]["path"])
    lat = benchmark(lambda: model.predict(rep, conf=0.25), device=device, warmup=5, runs=30)

    return {
        "key": key,
        "display": model.meta.display,
        "family": model.meta.family,
        "license": model.meta.license,
        "mAP": round(float(res["map"]) * 100, 1),
        "mAP_50": round(float(res["map_50"]) * 100, 1),
        "mAP_75": round(float(res["map_75"]) * 100, 1),
        "mAP_small": round(float(res["map_small"]) * 100, 1),
        "mAP_large": round(float(res["map_large"]) * 100, 1),
        "latency_ms": lat.as_row()["mean_ms"],
        "fps": lat.as_row()["fps"],
        "_model": model,  # kept for the qualitative pass
    }


def qualitative_figures(rows, samples, n_images: int):
    """Save a grid of annotated sample images per model."""
    picks = samples[:n_images]
    images = [load_image(s["path"]) for s in picks]
    for r in rows:
        model = r["_model"]
        annotated = []
        for img in images:
            det_obj = model.predict(img, conf=SHOW_CONF)
            names = list(det_obj.data.get("class_name", []))
            scores = det_obj.confidence if det_obj.confidence is not None else []
            labels = [f"{n} {s:.2f}" for n, s in zip(names, scores)]
            annotated.append(annotate_detections(img, det_obj, labels=labels))
        save_image_grid(
            annotated,
            RESULTS / f"samples_{r['key']}.png",
            cols=min(len(annotated), 2),
            titles=[f"{r['display']}"] * len(annotated),
        )


def write_outputs(rows, n_images: int, device) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "Family", "License", "mAP", "mAP@50", "mAP@75", "mAP (small)", "Latency (ms)", "FPS"]
    table = [
        [r["display"], r["family"], r["license"], r["mAP"], r["mAP_50"],
         r["mAP_75"], r["mAP_small"], r["latency_ms"], r["fps"]]
        for r in sorted(rows, key=lambda r: r["mAP"], reverse=True)
    ]
    notes = (
        f"*Evaluated on {n_images} COCO val2017 images; mAP via pycocotools-equivalent "
        f"torchmetrics at IoU=.50:.95. Latency = mean over 30 warmed-up single-image "
        f"forwards on `{describe_device(device)}`.*"
    )
    write_metrics_md(RESULTS / "metrics.md", title="COCO detection benchmark",
                     headers=headers, rows=table, notes=notes)
    # machine-readable copy (drop the live model handle)
    clean = [{k: v for k, v in r.items() if k != "_model"} for r in rows]
    (RESULTS / "metrics.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print("\n" + markdown_table(headers, table))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="yolo26n,rfdetr-nano",
                    help=f"comma list; available: {','.join(detect.available())}")
    ap.add_argument("--dataset", default="coco-val-200")
    ap.add_argument("--max-samples", type=int, default=200, help="COCO subset size to load")
    ap.add_argument("--limit", type=int, default=None, help="cap images actually evaluated")
    ap.add_argument("--images", type=int, default=4, help="number of qualitative sample images")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {describe_device(device)}")

    samples = load_coco_eval(args.max_samples, args.limit)
    print(f"evaluating on {len(samples)} images")

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    rows = []
    for key in keys:
        print(f"\n=== {key} ===")
        rows.append(evaluate_model(key, samples, device))
        r = rows[-1]
        print(f"  mAP={r['mAP']}  mAP@50={r['mAP_50']}  latency={r['latency_ms']}ms  fps={r['fps']}")

    qualitative_figures(rows, samples, args.images)
    write_outputs(rows, len(samples), device)
    print(f"\nwrote {RESULTS / 'metrics.md'} and sample figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
