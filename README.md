# cv-explorations

A hands-on, **benchmark-driven tour of modern computer vision**. Each major CV
capability — detection, classification, segmentation, depth, pose, tracking,
retrieval, vision-language models, and classical CV — is a self-contained
**module** that is both *runnable* and *explained*.

Every module ships four things:

1. **Adapters + a CLI engine** (`run.py`) — reusable, model-agnostic inference and benchmarking.
2. **An interactive notebook** (`explore.ipynb`) — narrative exploration.
3. **A rigorous report** (`report.qmd`) — concept → models → metrics → latency → failure cases → licenses.
4. **Committed results** (`results/`) — so the published site renders without a GPU.

Reports render to a public site via Quarto + GitHub Pages → **https://chaoticcuriosity-io.github.io/cv-explorations**.

## Why this exists

To onboard the author — and others — to *how these capabilities actually work*,
by running current (mid-2026) open-source models on real data and reporting
honest numbers rather than repeating marketing claims. A recurring theme is the
shift from **task-specific models** to **promptable foundation models** (SAM,
Grounding DINO, CLIP, Florence-2), shown side by side per task.

## Hardware

| Resource | Specs | Role |
|---|---|---|
| Local workstation | RTX 3090 Ti (24 GB), i7-9700K, 64 GB RAM, Win 11 | Primary dev — runs/fine-tunes essentially everything here |
| DGX Spark (Tailscale) | GB10 Grace-Blackwell, ~128 GB unified, ARM64/Linux | Scale-up + Linux-only corner (mmdet/detectron2/flash-attn) + robotics bridge |

## Setup

Uses [uv](https://docs.astral.sh/uv/) for environments and **Python 3.12** with
**PyTorch + CUDA 12.8** (Ampere-friendly).

```bash
uv sync --group dev                       # core stack (PyTorch + CV libs + tooling)
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

Per-module extras are installed on demand:

```bash
uv sync --group data --group detection    # FiftyOne + YOLO/RF-DETR for module 01
```

## Repository layout

```
cvkit/        shared toolkit (devices, io, timing, viz, metrics, data, report)
modules/      one folder per capability (NN-name/: detect.py, run.py, report.qmd, explore.ipynb, results/)
assets/       small committed sample images/video (qualitative inputs)
data/ models/ gitignored: downloaded datasets / weights
scripts/      env setup + data fetch helpers
tests/        cvkit smoke tests
```

## Running a module (object detection example)

```bash
uv run python modules/01-object-detection/run.py \
    --models rfdetr,yolo26n --dataset coco-val-200 --device cuda
quarto preview        # view reports locally
```

## Licensing note

Models span permissive (Apache/MIT: RF-DETR, SAM 2.1, SigLIP 2, DINOv2,
Florence-2) and copyleft (AGPL-3.0: Ultralytics YOLO/SAM-pose). Each benchmark
table records the per-model license; for learning that's informational, but it
matters before any commercial reuse.
