"""Segmentation engine: closed-set instance masks (YOLO-seg) vs promptable SAM 2.1.

Produces annotated mask figures + a latency/throughput table into ``results/``. SAM 2.1
is prompted with the YOLO detector's boxes, illustrating the common "detect, then
segment-anything" pipeline.

uv run python modules/03-segmentation/run.py --images 4
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import supervision as sv
import torch

import segment
from cvkit import describe_device, get_device
from cvkit.io import load_image
from cvkit.report import markdown_table, write_metrics_md
from cvkit.timing import benchmark
from cvkit.viz import save_image_grid

HERE = Path(__file__).parent
RESULTS = HERE / "results"
COCO_GLOB = r"C:/Users/dgbal/fiftyone/coco-2017/validation/data/*.jpg"

_MASK = sv.MaskAnnotator()
_BOX = sv.BoxAnnotator(thickness=1)
_LAB = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)


def annotate(image_rgb, det, labelled=True):
    out = _MASK.annotate(image_rgb.copy(), det)
    if labelled and det.data.get("class_name") is not None and len(det):
        names = list(det.data["class_name"])
        conf = det.confidence if det.confidence is not None else [1.0] * len(det)
        out = _BOX.annotate(out, det)
        out = _LAB.annotate(out, det, labels=[f"{n} {c:.2f}" for n, c in zip(names, conf)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="yolo26-seg,yolo11-seg")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {describe_device(device)}")
    images = [load_image(p) for p in sorted(glob.glob(COCO_GLOB))[: args.images]]
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = []
    yolo_boxes_for_sam = None
    for key in [k.strip() for k in args.models.split(",") if k.strip()]:
        seg = segment.build_segmenter(key, device)
        annotated, counts = [], []
        dets0 = None
        for img in images:
            det = seg.predict(img)
            counts.append(len(det))
            annotated.append(annotate(img, det))
            if dets0 is None:
                dets0 = det
        if key == "yolo26-seg":
            yolo_boxes_for_sam = dets0.xyxy
        lat = benchmark(lambda: seg.predict(images[0]), device=device, warmup=3, runs=20)
        save_image_grid(annotated, RESULTS / f"samples_{key}.png", cols=min(len(annotated), 2),
                        titles=[seg.meta.display] * len(annotated))
        rows.append({"display": seg.meta.display, "family": seg.meta.family, "license": seg.meta.license,
                     "latency_ms": lat.as_row()["mean_ms"], "fps": lat.as_row()["fps"],
                     "avg_masks": round(float(np.mean(counts)), 1)})
        print(f"  {key}: {lat.as_row()['mean_ms']}ms, {np.mean(counts):.1f} masks/img")

    # SAM 2.1, prompted by the YOLO detector's boxes (built once, reused).
    sam = segment.build_sam(device)
    yolo_for_sam = segment.build_segmenter("yolo26-seg", device)
    sam_annotated = []
    for img in images:
        boxes = yolo_for_sam.predict(img).xyxy
        det = sam.predict_boxes(img, boxes)
        sam_annotated.append(annotate(img, det, labelled=False))
    save_image_grid(sam_annotated, RESULTS / "samples_sam2.1.png", cols=min(len(sam_annotated), 2),
                    titles=["SAM 2.1 (box-prompted)"] * len(sam_annotated))
    rep_boxes = yolo_boxes_for_sam if yolo_boxes_for_sam is not None else np.array([[50, 50, 300, 300]])
    lat = benchmark(lambda: sam.predict_boxes(images[0], rep_boxes), device=device, warmup=3, runs=15)
    rows.append({"display": sam.meta.display, "family": sam.meta.family, "license": sam.meta.license,
                 "latency_ms": lat.as_row()["mean_ms"], "fps": lat.as_row()["fps"], "avg_masks": "n/a"})
    print(f"  sam2.1: {lat.as_row()['mean_ms']}ms")

    headers = ["Model", "Family", "License", "Latency (ms)", "FPS", "Masks/img"]
    table = [[r["display"], r["family"], r["license"], r["latency_ms"], r["fps"], r["avg_masks"]] for r in rows]
    notes = (
        f"*Latency = mean over warmed-up single-image forwards on `{describe_device(device)}`. "
        f"SAM 2.1 is prompted with the YOLO26-seg boxes (detect-then-segment). This module is "
        f"qualitative + latency; mask-mAP on COCO is a deeper follow-up.*"
    )
    write_metrics_md(RESULTS / "metrics.md", title="Segmentation: speed & masks", headers=headers, rows=table, notes=notes)
    (RESULTS / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + markdown_table(headers, table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
