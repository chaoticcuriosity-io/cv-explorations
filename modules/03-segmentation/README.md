# 03 · Segmentation (instance vs promptable)

> Status: **done**. Pixel-level labelling: closed-set instance masks vs Segment Anything.

| Model | Approach | License |
|---|---|---|
| YOLO26-n-seg / YOLO11-n-seg | Closed-set instance segmentation (80 COCO classes) | AGPL-3.0 |
| SAM 2.1 (base) | Promptable, class-agnostic segmentation | Apache-2.0 |

Demonstrates the **detect-then-segment** pipeline (YOLO boxes → SAM masks) and contrasts a
fast labelled instance segmenter with a slower, prompt-anything foundation model.
Qualitative masks + latency on the local RTX 3090 Ti; see the
[report](report.qmd).

```bash
uv sync --group detection
uv run python modules/03-segmentation/run.py --images 4
```
