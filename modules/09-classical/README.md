# 09 · Classical computer vision

> Status: **done**. The pre-deep-learning toolkit — fast, transparent, no training.

OpenCV: **Canny edges + contours**, **ORB feature matching**, **Farneback optical flow**. All
on CPU in milliseconds (no GPU, no model download), as the contrast that makes modules 01–08
legible — each deep module has a classical ancestor here. See the [report](report.qmd).

```bash
uv sync --group dev
uv run python modules/09-classical/run.py
```
