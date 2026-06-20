# 05 · Human pose estimation

> Status: **done**. 17-keypoint skeletons per person (single-stage YOLO-pose).

| Model | Approach | License |
|---|---|---|
| YOLO26-n-pose / YOLO11-n-pose | Bottom-up single-stage | AGPL-3.0 |

Qualitative skeleton overlays + latency (both ~50 fps, crowd-constant) on the RTX 3090 Ti;
top-down (ViTPose/RTMPose) and COCO OKS-AP noted as follow-ups. See the [report](report.qmd).

```bash
uv sync --group detection
uv run python modules/05-pose/run.py --images 4
```
