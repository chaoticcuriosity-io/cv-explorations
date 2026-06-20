"""Benchmark engine for the classification module.

Evaluates supervised (timm) and zero-shot (SigLIP/CLIP) classifiers on Imagenette
val — top-1 accuracy and warmed-up latency — and writes a metrics table + a
predictions figure into ``results/``.

uv run python modules/02-classification/run.py --limit 1500
uv run python modules/02-classification/run.py            # full val (3925 imgs)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import classify
from cvkit import describe_device, get_device
from cvkit.io import load_image
from cvkit.report import markdown_table, write_metrics_md
from cvkit.timing import benchmark

HERE = Path(__file__).parent
RESULTS = HERE / "results"
DATA_ROOT = "data/imagenette"


def load_samples(limit: int | None) -> list[tuple[str, int]]:
    import torchvision

    ds = torchvision.datasets.Imagenette(DATA_ROOT, split="val", size="320px", download=True)
    samples = list(ds._samples)  # [(path, label), ...] grouped by class
    return samples[:limit] if limit else samples


def evaluate(key: str, samples, device) -> dict:
    model = classify.build_classifier(key, device)
    correct = 0
    for path, label in samples:
        if model.predict(load_image(path)) == label:
            correct += 1
    acc = 100.0 * correct / len(samples)

    rep = load_image(samples[0][0])
    lat = benchmark(lambda: model.scores(rep), device=device, warmup=5, runs=30)
    return {
        "key": key,
        "display": model.meta.display,
        "paradigm": model.meta.paradigm,
        "license": model.meta.license,
        "top1": round(acc, 1),
        "latency_ms": lat.as_row()["mean_ms"],
        "fps": lat.as_row()["fps"],
        "_model": model,
    }


def predictions_figure(rows, samples, k: int) -> None:
    n = len(samples)
    idxs = np.linspace(0, n - 1, k).astype(int)
    fig, axes = plt.subplots(1, k, figsize=(3.3 * k, 4.6))
    for ax, si in zip(np.atleast_1d(axes).ravel(), idxs):
        path, label = samples[si]
        img = load_image(path)
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"truth: {classify.CLASS_NAMES[label]}", fontsize=9)
        lines = []
        for r in rows:
            pred = r["_model"].predict(img)
            mark = "correct" if pred == label else "WRONG"
            lines.append(f"{r['display']}: {classify.CLASS_NAMES[pred]} [{mark}]")
        ax.set_xlabel("\n".join(lines), fontsize=7, loc="left")
    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "predictions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_outputs(rows, n_images: int, device) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "Paradigm", "License", "Top-1 (%)", "Latency (ms)", "FPS"]
    table = [
        [r["display"], r["paradigm"], r["license"], r["top1"], r["latency_ms"], r["fps"]]
        for r in sorted(rows, key=lambda r: r["top1"], reverse=True)
    ]
    notes = (
        f"*Top-1 accuracy on {n_images} Imagenette val images (10 classes). Latency = mean of "
        f"30 warmed-up single-image forwards on `{describe_device(device)}`. Zero-shot models "
        f"were never trained on Imagenette — they classify by image-text similarity.*"
    )
    write_metrics_md(RESULTS / "metrics.md", title="Imagenette classification benchmark",
                     headers=headers, rows=table, notes=notes)
    clean = [{k: v for k, v in r.items() if k != "_model"} for r in rows]
    (RESULTS / "metrics.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print("\n" + markdown_table(headers, table))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="convnext-tiny,siglip2-base,openclip-vitb32",
                    help=f"comma list; available: {','.join(classify.available())}")
    ap.add_argument("--limit", type=int, default=None, help="cap images evaluated (default: full val)")
    ap.add_argument("--images", type=int, default=5, help="qualitative sample images")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {describe_device(device)}")
    samples = load_samples(args.limit)
    print(f"evaluating on {len(samples)} Imagenette val images")

    rows = []
    for key in [k.strip() for k in args.models.split(",") if k.strip()]:
        print(f"\n=== {key} ===")
        rows.append(evaluate(key, samples, device))
        r = rows[-1]
        print(f"  top-1={r['top1']}%  latency={r['latency_ms']}ms  fps={r['fps']}")

    predictions_figure(rows, samples, args.images)
    write_outputs(rows, len(samples), device)
    print(f"\nwrote {RESULTS / 'metrics.md'} and predictions figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
