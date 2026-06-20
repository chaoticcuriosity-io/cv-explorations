"""Retrieval engine: embed Imagenette, measure nearest-neighbour quality, show results.

Metrics (real, not qualitative):
- **Recall@1** — is the single nearest image (excluding self) the same class?
- **Precision@5** — of the 5 nearest, what fraction share the query's class?
Plus a visual "query -> top-5 neighbours" figure and embedding latency.

uv run python modules/07-retrieval/run.py --per-class 60
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

import embed
from cvkit import describe_device, get_device
from cvkit.io import load_image
from cvkit.report import markdown_table, write_metrics_md
from cvkit.timing import benchmark

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def load_subset(per_class: int):
    import torchvision

    ds = torchvision.datasets.Imagenette("data/imagenette", split="val", size="320px", download=True)
    by: dict[int, list[str]] = {}
    for path, label in ds._samples:
        by.setdefault(label, []).append(path)
    samples = []
    for label, paths in sorted(by.items()):
        samples += [(p, label) for p in paths[:per_class]]
    return samples


def metrics(E: np.ndarray, labels: np.ndarray):
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)  # exclude self-match
    nn = S.argmax(1)
    recall1 = float((labels[nn] == labels).mean()) * 100
    top5 = np.argsort(-S, axis=1)[:, :5]
    p5 = float((labels[top5] == labels[:, None]).mean()) * 100
    return round(recall1, 1), round(p5, 1), S


def qualitative(paths, labels, S, n_queries: int):
    qi = np.linspace(0, len(paths) - 1, n_queries).astype(int)
    fig, axes = plt.subplots(n_queries, 6, figsize=(13, 2.3 * n_queries))
    axes = np.atleast_2d(axes)
    for r, q in enumerate(qi):
        neigh = np.argsort(-S[q])[:5]
        cols = [q, *neigh]
        for c, idx in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(load_image(paths[idx])); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("query", fontsize=8)
                for s in ax.spines.values():
                    s.set_color("black"); s.set_linewidth(2)
            else:
                ok = labels[idx] == labels[q]
                for s in ax.spines.values():
                    s.set_color("green" if ok else "red"); s.set_linewidth(2)
            if r == 0:
                ax.set_title("query" if c == 0 else f"#{c}", fontsize=8)
    fig.suptitle("Nearest neighbours (green = same class, red = different)", fontsize=11)
    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "retrieval.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="dinov2-base,siglip2-base")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--queries", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {describe_device(device)}")
    samples = load_subset(args.per_class)
    paths = [p for p, _ in samples]
    labels = np.array([l for _, l in samples])
    print(f"gallery: {len(samples)} images, {len(set(labels))} classes")

    rows, first_S = [], None
    for key in [k.strip() for k in args.models.split(",") if k.strip()]:
        m = embed.build_embedder(key, device)
        E = np.stack([m.embed(load_image(p)) for p in paths])
        r1, p5, S = metrics(E, labels)
        lat = benchmark(lambda: m.embed(load_image(paths[0])), device=device, warmup=3, runs=20)
        rows.append({"display": m.meta.display, "kind": m.meta.kind, "license": m.meta.license,
                     "recall1": r1, "p5": p5, "latency_ms": lat.as_row()["mean_ms"], "fps": lat.as_row()["fps"]})
        print(f"  {key}: R@1={r1}%  P@5={p5}%  {lat.as_row()['mean_ms']}ms")
        if first_S is None:
            first_S = (S, m.meta.display)

    qualitative(paths, labels, first_S[0], args.queries)

    headers = ["Model", "Type", "License", "Recall@1 (%)", "Precision@5 (%)", "Latency (ms)", "FPS"]
    table = [[r["display"], r["kind"], r["license"], r["recall1"], r["p5"], r["latency_ms"], r["fps"]]
             for r in sorted(rows, key=lambda r: r["recall1"], reverse=True)]
    notes = (
        f"*Retrieval on {len(samples)} Imagenette images ({args.per_class}/class). Recall@1 = nearest "
        f"neighbour shares the class; Precision@5 = fraction of top-5 that do. No labels used at search "
        f"time — purely embedding similarity. Latency on `{describe_device(device)}`. Figure uses "
        f"{first_S[1]}.*"
    )
    write_metrics_md(RESULTS / "metrics.md", title="Image retrieval (Imagenette)", headers=headers, rows=table, notes=notes)
    (RESULTS / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + markdown_table(headers, table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
