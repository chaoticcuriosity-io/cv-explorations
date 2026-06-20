# 01 · Object detection

> Status: **flagship module — in progress (M1)**. This folder defines the
> repeatable template every later module copies.

**Question:** *what objects are in the image, and where (bounding boxes)?*

## What this module will benchmark

| Model | Family | License | Why included |
|---|---|---|---|
| RF-DETR | DETR (transformer) | Apache-2.0 | Permissive lead; strong accuracy/latency |
| RT-DETRv2 | DETR (real-time) | Apache-2.0 | Permissive real-time baseline |
| YOLO26 | YOLO (NMS-free) | AGPL-3.0 | Easiest/fastest reference (license flagged) |
| Grounding DINO | open-vocabulary | Apache-2.0 | Detect-anything-by-text; the "promptable" contrast |

## Planned artifacts (the template)

- `detect.py` — one adapter per model, each returning `supervision.Detections`.
- `run.py` — CLI: `--models`, `--dataset coco-val-200`, `--device cuda`; computes
  COCO mAP + warmed-up latency; writes `results/metrics.md` and annotated samples.
- `explore.ipynb` — interactive narrative.
- `report.qmd` — the rigorous write-up that renders into the site.
- `results/` — committed figures + metrics tables (site renders without a GPU).

## Run (once built)

```bash
uv sync --group data --group detection
uv run python scripts/fetch_data.py coco --max-samples 200
uv run python modules/01-object-detection/run.py --models rfdetr,yolo26n --device cuda
```
