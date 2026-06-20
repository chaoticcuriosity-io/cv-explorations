"""Depth engine: Depth Anything V2 Small vs Base — depth maps + latency.

Saves a side-by-side RGB / depth comparison figure and a latency table into ``results/``.

uv run python modules/04-depth/run.py --images 4
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import depth
from cvkit import describe_device, get_device
from cvkit.io import load_image
from cvkit.report import markdown_table, write_metrics_md
from cvkit.timing import benchmark

HERE = Path(__file__).parent
RESULTS = HERE / "results"
COCO = r"C:/Users/dgbal/fiftyone/coco-2017/validation/data/*.jpg"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="dav2-small,dav2-base")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {describe_device(device)}")
    images = [load_image(p) for p in sorted(glob.glob(COCO))[: args.images]]
    models = [depth.build_depth(k.strip(), device) for k in args.models.split(",") if k.strip()]

    depths = {m.meta.key: [m.predict(im) for im in images] for m in models}

    ncol = 1 + len(models)
    fig, axes = plt.subplots(len(images), ncol, figsize=(4 * ncol, 3.1 * len(images)))
    axes = np.atleast_2d(axes)
    for r, im in enumerate(images):
        axes[r, 0].imshow(im); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("RGB input", fontsize=10)
        for c, m in enumerate(models):
            ax = axes[r, c + 1]
            ax.imshow(depths[m.meta.key][r], cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(m.meta.display, fontsize=9)
    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "depth_maps.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for m in models:
        lat = benchmark(lambda: m.predict(images[0]), device=device, warmup=3, runs=20)
        rows.append({"display": m.meta.display, "license": m.meta.license,
                     "latency_ms": lat.as_row()["mean_ms"], "fps": lat.as_row()["fps"]})
        print(f"  {m.meta.key}: {lat.as_row()['mean_ms']}ms  fps={lat.as_row()['fps']}")

    headers = ["Model", "License", "Latency (ms)", "FPS"]
    table = [[r["display"], r["license"], r["latency_ms"], r["fps"]] for r in rows]
    notes = (
        f"*Relative (inverse) monocular depth; in the figure brighter = nearer. Latency = mean "
        f"warmed-up single-image forward on `{describe_device(device)}`. Qualitative + latency; "
        f"metric-depth eval (AbsRel/RMSE on NYU/KITTI) is a deeper follow-up.*"
    )
    write_metrics_md(RESULTS / "metrics.md", title="Monocular depth: speed", headers=headers, rows=table, notes=notes)
    (RESULTS / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + markdown_table(headers, table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
