"""Tracking engine: YOLO detection + ByteTrack across video frames.

Detection answers "what's here *now*"; tracking adds **identity over time** — the same
person keeps the same id across frames. We run YOLO (person class) per frame, associate
boxes with ByteTrack, and render ids + motion traces. Outputs an animated GIF + keyframes
+ a small stats table into ``results/``.

uv run python modules/06-tracking/run.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image

from cvkit import describe_device, get_device
from cvkit.report import markdown_table, write_metrics_md
from cvkit.viz import save_image_grid

HERE = Path(__file__).parent
RESULTS = HERE / "results"
VIDEO = "data/people-walking.mp4"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--gif-stride", type=int, default=5)
    ap.add_argument("--gif-width", type=int, default=480)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {describe_device(device)}")
    from ultralytics import YOLO

    model = YOLO(args.weights)
    dev = device.index or 0 if device.type == "cuda" else "cpu"
    info = sv.VideoInfo.from_video_path(VIDEO)
    tracker = sv.ByteTrack(frame_rate=int(info.fps))
    box = sv.BoxAnnotator(thickness=2)
    lab = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace = sv.TraceAnnotator(thickness=2, trace_length=30)

    key_idx = {0, args.max_frames // 2, args.max_frames - 1}
    ids: set[int] = set()
    gif_frames, keyframes = [], {}
    n = 0
    t0 = time.perf_counter()
    for i, frame_bgr in enumerate(sv.get_video_frames_generator(VIDEO)):
        if i >= args.max_frames:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = model.predict(rgb, classes=[0], conf=0.3, device=dev, verbose=False)[0]
        det = sv.Detections.from_ultralytics(r)
        det = tracker.update_with_detections(det)
        if det.tracker_id is not None:
            ids.update(int(t) for t in det.tracker_id)
        labels = [f"#{t}" for t in (det.tracker_id if det.tracker_id is not None else [])]
        ann = trace.annotate(rgb.copy(), det)
        ann = box.annotate(ann, det)
        ann = lab.annotate(ann, det, labels=labels)
        n += 1
        if i % args.gif_stride == 0:
            im = Image.fromarray(ann)
            h = int(im.height * args.gif_width / im.width)
            gif_frames.append(im.resize((args.gif_width, h)))
        if i in key_idx:
            keyframes[i] = ann
    elapsed = time.perf_counter() - t0
    proc_fps = n / elapsed if elapsed else 0.0

    RESULTS.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(RESULTS / "tracking.gif", save_all=True, append_images=gif_frames[1:],
                       duration=80, loop=0, optimize=True)
    kfs = [keyframes[k] for k in sorted(keyframes)]
    save_image_grid(kfs, RESULTS / "keyframes.png", cols=min(len(kfs), 3),
                    titles=[f"frame {k}" for k in sorted(keyframes)])

    headers = ["Metric", "Value"]
    table = [
        ["Unique tracks (ids)", len(ids)],
        ["Frames processed", n],
        ["Processing speed", f"{proc_fps:.1f} fps (detect+track, {str(describe_device(device)).split('|')[0].strip()})"],
        ["Source video", f"{info.width}x{info.height} @ {info.fps:.0f} fps"],
    ]
    notes = (
        "*YOLO26 (person class) + ByteTrack association, with motion traces. Qualitative + "
        "throughput; MOT metrics (HOTA/MOTA/IDF1 via TrackEval) on a labelled benchmark are a "
        "deeper follow-up.*"
    )
    write_metrics_md(RESULTS / "metrics.md", title="Multi-object tracking", headers=headers, rows=table, notes=notes)
    (RESULTS / "metrics.json").write_text(json.dumps({"unique_tracks": len(ids), "frames": n,
                                          "proc_fps": round(proc_fps, 1)}, indent=2), encoding="utf-8")
    print("\n" + markdown_table(headers, table))
    print(f"wrote {RESULTS/'tracking.gif'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
