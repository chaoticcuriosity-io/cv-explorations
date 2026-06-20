"""Open-vocabulary detection demo (Grounding DINO).

Shows the paradigm the closed-set benchmark can't: detecting concepts from *text*,
including phrases that are NOT among COCO's 80 classes. Saves an annotated figure to
``results/openvocab_grounding-dino.png`` and prints which detected labels are
out-of-COCO (the whole point).

uv run python modules/01-object-detection/openvocab.py
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import detect
from cvkit.data import sample_images
from cvkit.devices import describe_device, get_device
from cvkit.io import load_image
from cvkit.viz import annotate_detections, save_image_grid

HERE = Path(__file__).parent
RESULTS = HERE / "results"
COCO_DIR = r"C:/Users/dgbal/fiftyone/coco-2017/validation/data/*.jpg"

# A mix of COCO classes (person/bottle/cup) and concepts COCO never had
# (plate, napkin, logo, shoe, hand) — the out-of-COCO hits prove open-vocabulary.
DEFAULT_PROMPTS = ["person", "bottle", "cup", "plate", "napkin", "logo", "shoe", "hand"]


def pick_images(n: int) -> list[str]:
    imgs = sorted(glob.glob(COCO_DIR))
    if not imgs:
        imgs = [str(p) for p in sample_images()]
    return imgs[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3, help="number of images")
    ap.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    ap.add_argument("--box-threshold", type=float, default=0.30)
    args = ap.parse_args()

    device = get_device()
    print(f"device: {describe_device(device)}")
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    print(f"prompts: {prompts}")

    gd = detect.build_open_vocab(device)
    images = pick_images(args.n)
    annotated, titles = [], []
    for path in images:
        img = load_image(path)
        det = gd.detect(img, prompts, box_threshold=args.box_threshold)
        names = list(det.data.get("class_name", []))
        scores = det.confidence if det.confidence is not None else []
        labels = [f"{n} {s:.2f}" for n, s in zip(names, scores)]
        out_of_coco = sorted({n for n in names if n.lower() not in detect.NAME_TO_IDX})
        print(f"\n{Path(path).name}: {len(det)} detections")
        print("  labels       :", names)
        print("  NOT in COCO-80:", out_of_coco or "(none)")
        annotated.append(annotate_detections(img, det, labels=labels))
        titles.append(Path(path).name)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = save_image_grid(
        annotated, RESULTS / "openvocab_grounding-dino.png",
        cols=min(len(annotated), 3), titles=titles,
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
