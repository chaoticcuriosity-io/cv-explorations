# 06 · Multi-object tracking

> Status: **done**. Identity over time = detection + association.

YOLO26 (person class) + **ByteTrack** on a pedestrian clip, with motion traces. Outputs an
animated GIF + keyframes + throughput. Qualitative; MOT metrics (HOTA/MOTA/IDF1 via TrackEval)
noted as a follow-up. See the [report](report.qmd).

```bash
uv sync --group detection
uv run python modules/06-tracking/run.py
```

The sample video is fetched by `supervision` into `data/` (gitignored).
