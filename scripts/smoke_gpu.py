"""Print the active device and confirm CUDA is usable.

Run with: ``uv run python scripts/smoke_gpu.py``
Exits non-zero if a CUDA GPU was expected (default) but is unavailable.
"""

from __future__ import annotations

import argparse
import sys

import torch

from cvkit import describe_device, get_device


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--require-cuda",
        action="store_true",
        default=True,
        help="fail if CUDA is unavailable (default: on)",
    )
    ap.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    args = ap.parse_args()

    dev = get_device()
    print(f"torch        : {torch.__version__}")
    print(f"cuda built   : {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device       : {describe_device(dev)}")

    if dev.type == "cuda":
        x = torch.randn(2048, 2048, device=dev)
        y = (x @ x).sum().item()  # exercise a real kernel
        print(f"matmul check : ok ({y:.3e})")
    elif args.require_cuda:
        print("ERROR: CUDA expected but not available.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
