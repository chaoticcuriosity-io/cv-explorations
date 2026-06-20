"""Fine-tune YOLO to detect air vents — runs on the DGX Spark (or locally).

Trains an ultralytics YOLO detector on the auto-labelled air-vent dataset, then emits
standardized artifacts into an output dir:
  - metrics.json       final mAP / precision / recall on the val split
  - before_after.png   base COCO model (knows no "air vent") vs the fine-tuned model
  - curves.png         ultralytics training curves (loss / mAP over epochs)
  - confusion.png      confusion matrix

The before/after montage is the headline: the base model detects *zero* vents (no such
class), the fine-tuned model localises them.

python finetune.py --data /data/airvents/data.yaml --model yolo11s.pt --epochs 80 --out /out
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image


def annotate(model, image_rgb, conf):
    import matplotlib

    r = model.predict(image_rgb, conf=conf, verbose=False)[0]
    det = sv.Detections.from_ultralytics(r)
    box = sv.BoxAnnotator(thickness=3)
    lab = sv.LabelAnnotator(text_scale=0.6)
    out = box.annotate(image_rgb.copy(), det)
    names = list(det.data.get("class_name", []))
    confs = det.confidence if det.confidence is not None else []
    labels = [f"{n} {c:.2f}" for n, c in zip(names, confs)]
    return lab.annotate(out, det, labels=labels), len(det)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"device: {torch.cuda.get_device_name(0) if dev == 0 else 'cpu'}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- baseline (before): the pretrained COCO model has no air_vent class ---
    base = YOLO(args.model)

    # --- fine-tune ---
    model = YOLO(args.model)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=dev, project=str(out / "runs"), name="airvent", exist_ok=True,
        patience=20, seed=0, verbose=False,
    )
    save_dir = Path(results.save_dir)

    # --- metrics on val ---
    val = model.val(data=args.data, device=dev, verbose=False)
    metrics = {
        "model": args.model,
        "epochs": args.epochs,
        "mAP50": round(float(val.box.map50), 4),
        "mAP50_95": round(float(val.box.map), 4),
        "precision": round(float(val.box.mp), 4),
        "recall": round(float(val.box.mr), 4),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("metrics:", metrics)

    # --- before/after montage on up to 6 val images ---
    data_root = Path(args.data).parent
    val_imgs = sorted(glob.glob(str(data_root / "images/val/*.jpg")))[:6]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(val_imgs)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 7))
    axes = np.atleast_2d(axes)
    for c, p in enumerate(val_imgs):
        img = np.asarray(Image.open(p).convert("RGB"))
        b_img, b_n = annotate(base, img, conf=0.25)
        f_img, f_n = annotate(model, img, conf=0.30)
        axes[0, c].imshow(b_img); axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
        axes[1, c].imshow(f_img); axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
        if c == 0:
            axes[0, c].set_ylabel(f"BEFORE\n(base {args.model})", fontsize=10)
            axes[1, c].set_ylabel("AFTER\n(fine-tuned)", fontsize=10)
    fig.suptitle("Air-vent detection — before vs after fine-tuning", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "before_after.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- copy ultralytics curves / confusion matrix ---
    for src, dst in [("results.png", "curves.png"), ("confusion_matrix.png", "confusion.png")]:
        s = save_dir / src
        if s.exists():
            shutil.copy(s, out / dst)
    # keep the trained weights alongside artifacts
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, out / "best.pt")

    print(f"\nartifacts in {out}: " + ", ".join(p.name for p in sorted(out.glob('*'))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
