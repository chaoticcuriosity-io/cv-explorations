"""Build an air-vent detection dataset by auto-labelling, no manual annotation.

Pipeline (the "distill an open-vocab model into a specialist" pattern promised in
module 01):
  1. Pull Creative-Commons air-vent images from the Openverse API (no key needed).
  2. Auto-label each with **Grounding DINO** (module 01's open-vocab detector),
     prompted with vent phrases — boxes above a confidence floor become labels.
  3. Write a YOLO-format dataset (images/ + labels/ + data.yaml, train/val split)
     that ultralytics can train on directly.

Run on the local GPU (Grounding DINO is heavy); the resulting dataset is then
shipped to the Spark for training.

uv run python modules/10-finetuning/prepare_data.py --max-images 200
uv run python modules/10-finetuning/prepare_data.py --max-images 16 --spot-check   # quick visual check
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "01-object-detection"))
import detect  # noqa: E402  (module 01's adapters; provides Grounding DINO)

from cvkit.data import DATA_DIR  # noqa: E402
from cvkit.devices import get_device  # noqa: E402
from cvkit.viz import annotate_detections, save_image_grid  # noqa: E402

OUT = DATA_DIR / "airvents"
QUERIES = [
    "air vent cover", "wall air vent", "ceiling air vent", "HVAC register vent",
    "ventilation grille wall", "air register grille", "floor air vent", "air conditioner vent",
    "louvered vent cover", "supply air diffuser", "return air grille", "wall ventilation louver",
    "air brick wall", "soffit vent", "gable vent", "dryer vent exterior", "extractor fan vent",
    "air intake grille", "metal vent cover", "ceiling air diffuser", "heating register floor",
    "wall mounted ventilation grille", "exhaust vent cover", "round duct vent",
]
# Phrases we hand Grounding DINO. Kept broad so we catch vents in varied scenes.
LABEL_PROMPTS = ["air vent", "ventilation grille", "wall vent", "ceiling vent"]
# Wikimedia enforces a descriptive User-Agent.
UA = {"User-Agent": "cv-explorations/0.1 (https://github.com/chaoticcuriosity-io/cv-explorations; educational)"}
IMG_EXT = (".jpg", ".jpeg", ".png")


def source_image_records(per_query: int) -> list[dict]:
    """Return open image records {url, license, creator} from Wikimedia Commons.

    Commons hosts CC / public-domain media via a no-auth API. We over-fetch across
    several vent queries; Grounding DINO filters out anything that isn't a vent.
    """
    seen, records = set(), []
    for q in QUERIES:
        api = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": q, "gsrnamespace": "6", "gsrlimit": per_query,
            "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "1280",
        })
        try:
            req = urllib.request.Request(api, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
        except Exception as e:  # noqa: BLE001
            print(f"  commons query {q!r} failed: {e}")
            continue
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if not url or not url.lower().endswith(IMG_EXT) or url in seen:
                continue
            seen.add(url)
            meta = info.get("extmetadata") or {}
            records.append({
                "url": url,
                "license": (meta.get("LicenseShortName") or {}).get("value"),
                "creator": (meta.get("Artist") or {}).get("value", "")[:120],
                "query": q,
            })
    return records


class ClipGate:
    """Zero-shot relevance filter (module 02's idea): keep vent-like images, reject
    smokestacks / landscapes / diagrams / building exteriors before we trust a label."""

    POS = [
        "a photo of an air vent cover",
        "an HVAC vent grille on a wall or ceiling",
        "a metal ventilation register cover indoors",
    ]
    NEG = [
        "a smokestack or industrial chimney",
        "an outdoor landscape or field",
        "a scanned document, book cover, or page of text",
        "a poster or sign with large text",
        "a technical diagram or schematic drawing",
        "a building exterior or street scene",
        "a hole in rock or a cave entrance",
        "a bathroom or kitchen interior",
        "a car, truck, or motor vehicle",
        "a car interior or dashboard",
        "a person, a face, or a statue",
    ]

    def __init__(self, device: torch.device):
        import open_clip

        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.model = self.model.to(device).eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            t = self.model.encode_text(tok(self.POS + self.NEG).to(device))
            self.text = t / t.norm(dim=-1, keepdim=True)
        self.npos, self.device = len(self.POS), device

    def relevant(self, image_rgb: np.ndarray, margin: float = 0.20) -> tuple[bool, float]:
        with torch.no_grad():
            x = self.pre(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
            ie = self.model.encode_image(x)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            # Apply CLIP's learned temperature so the softmax is meaningfully peaked,
            # not near-uniform over raw cosine similarities.
            logits = self.model.logit_scale.exp() * (ie @ self.text.T)[0]
            probs = logits.softmax(0).cpu().numpy()
        pos = float(probs[: self.npos].sum())
        return pos > (1 - pos) + margin, pos


CACHE = OUT / "_cache"


def fetch_image(url: str, retries: int = 2) -> np.ndarray | None:
    """Fetch an image, caching raw downloads so re-runs (e.g. filter tuning) are cheap."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".jpg")
    if cached.exists():
        try:
            return np.asarray(Image.open(cached).convert("RGB"))
        except Exception:  # noqa: BLE001
            cached.unlink(missing_ok=True)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
            if min(img.size) < 160:  # too small to be useful
                return None
            img.save(cached, quality=88)
            return np.asarray(img)
        except Exception:  # noqa: BLE001
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))  # back off; Commons rate-limits bursts
    return None


def to_yolo_lines(det, w: int, h: int, conf: float) -> list[str]:
    """Convert sv.Detections (xyxy abs) to YOLO label lines (class 0 = air_vent).

    Box-area sanity: drop boxes that are near-whole-image (usually a mislabelled
    building/landscape) or vanishingly small.
    """
    lines = []
    scores = det.confidence if det.confidence is not None else np.ones(len(det))
    for (x1, y1, x2, y2), s in zip(det.xyxy, scores):
        if s < conf:
            continue
        cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if not (0.0008 < bw * bh < 0.85) or bw <= 0 or bh <= 0:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--per-query", type=int, default=50)
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--clip-margin", type=float, default=0.15, help="CLIP gate strictness")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--spot-check", action="store_true",
                    help="just download+label a few and save an annotated grid for inspection")
    args = ap.parse_args()

    device = get_device()
    gd = detect.build_open_vocab(device)
    clip = ClipGate(device)
    print(f"sourcing open air-vent images from Wikimedia Commons ...")
    records = source_image_records(args.per_query)
    print(f"  {len(records)} candidate image URLs")

    if args.spot_check:
        grid, kept, rejected = [], 0, 0
        for rec in records:
            if kept >= args.max_images:
                break
            img = fetch_image(rec["url"])
            if img is None:
                continue
            ok, score = clip.relevant(img, margin=args.clip_margin)
            if not ok:
                rejected += 1
                continue
            det = gd.detect(img, LABEL_PROMPTS, box_threshold=args.box_threshold)
            lines = to_yolo_lines(det, img.shape[1], img.shape[0], args.box_threshold)
            if not lines:
                continue
            labels = [f"vent {c:.2f}" for c in det.confidence]
            grid.append(annotate_detections(img, det, labels=labels))
            kept += 1
        OUT.mkdir(parents=True, exist_ok=True)
        save_image_grid(grid, OUT / "spotcheck.png", cols=4,
                        titles=[f"clip+gdino" for _ in grid])
        print(f"  wrote {OUT/'spotcheck.png'} ({kept} kept, {rejected} rejected by CLIP gate)")
        return 0

    # Full dataset build. Clear any prior images/labels (keep the raw _cache).
    import shutil

    for sub in ("images", "labels"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    img_tr, img_va = OUT / "images/train", OUT / "images/val"
    lab_tr, lab_va = OUT / "labels/train", OUT / "labels/val"
    for d in (img_tr, img_va, lab_tr, lab_va):
        d.mkdir(parents=True, exist_ok=True)

    sources, kept, idx, rejected, fetch_fail = [], 0, 0, 0, 0
    for rec in records:
        if kept >= args.max_images:
            break
        img = fetch_image(rec["url"])
        if img is None:
            fetch_fail += 1
            continue
        ok, _ = clip.relevant(img, margin=args.clip_margin)
        if not ok:  # CLIP relevance gate: reject non-vent imagery before labelling
            rejected += 1
            continue
        h, w = img.shape[:2]
        det = gd.detect(img, LABEL_PROMPTS, box_threshold=args.box_threshold)
        lines = to_yolo_lines(det, w, h, args.box_threshold)
        if not lines:  # keep only images with at least one vent (positive examples)
            continue
        is_val = (idx % int(1 / args.val_frac)) == 0
        idx += 1
        stem = f"vent_{kept:04d}"
        (img_va if is_val else img_tr).joinpath(f"{stem}.jpg")
        Image.fromarray(img).save((img_va if is_val else img_tr) / f"{stem}.jpg", quality=90)
        (lab_va if is_val else lab_tr).joinpath(f"{stem}.txt").write_text("\n".join(lines) + "\n")
        sources.append({**rec, "file": f"{stem}.jpg", "split": "val" if is_val else "train",
                        "n_boxes": len(lines)})
        kept += 1
        if kept % 20 == 0:
            print(f"  labeled {kept} images ...")

    data_yaml = OUT / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: air_vent\n", encoding="utf-8"
    )
    (OUT / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    n_val = sum(1 for s in sources if s["split"] == "val")
    print(f"\nDONE: {kept} labeled images ({kept - n_val} train / {n_val} val), "
          f"{sum(s['n_boxes'] for s in sources)} boxes; {rejected} CLIP-rejected, "
          f"{fetch_fail} fetch failures")
    print(f"  dataset: {OUT}")
    print(f"  data.yaml: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
