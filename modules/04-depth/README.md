# 04 · Monocular depth

> Status: **done**. Distance-per-pixel from a single RGB image — the robotics-relevant one.

| Model | Output | License |
|---|---|---|
| Depth Anything V2 Small / Base | Relative (inverse) depth | Apache-2.0 |

Qualitative depth maps + latency on the local RTX 3090 Ti (Small ~26 fps, Base ~13 fps).
Metric-depth eval (AbsRel/RMSE on NYU/KITTI) is noted as a follow-up. See the
[report](report.qmd).

```bash
uv sync --group dev
uv run python modules/04-depth/run.py --images 4
```
