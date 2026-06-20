"""Pre-download benchmark dataset subsets so module runs are offline-fast.

Currently supports the COCO-2017 validation subset used by the detection module.
Requires the ``data`` group:  ``uv sync --group data``.

Usage:  uv run python scripts/fetch_data.py coco --max-samples 200
"""

from __future__ import annotations

import argparse


def fetch_coco(max_samples: int) -> None:
    from cvkit.data import load_coco_val_subset

    ds = load_coco_val_subset(max_samples=max_samples)
    print(f"Ready: '{ds.name}' with {len(ds)} samples at {ds.first().filepath!r} ...")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    coco = sub.add_parser("coco", help="COCO-2017 validation subset (FiftyOne zoo)")
    coco.add_argument("--max-samples", type=int, default=200)
    args = ap.parse_args()

    if args.cmd == "coco":
        fetch_coco(args.max_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
