"""Paths and dataset-fetch helpers.

Two data sources per the project plan: small *public benchmark subsets* (for real
metrics) and the owner's *own assets* (for qualitative + failure analysis). Heavy
deps (``fiftyone``) are imported lazily so the base env need not carry them.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repo root, resolved from this file (cvkit is installed editable)."""
    return Path(__file__).resolve().parents[1]


ASSETS_DIR = repo_root() / "assets"   # committed, small, owner's + iconic images
DATA_DIR = repo_root() / "data"       # gitignored, downloaded datasets
MODELS_DIR = repo_root() / "models"   # gitignored, downloaded weights


def ensure_dirs() -> None:
    for d in (ASSETS_DIR, DATA_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def sample_images() -> list[Path]:
    """Return the committed sample images under ``assets/`` (qualitative inputs)."""
    from cvkit.io import list_images

    return list_images(ASSETS_DIR)


def load_coco_val_subset(max_samples: int = 200, *, seed: int = 51):
    """Download a small COCO-2017 *validation* slice via the FiftyOne zoo.

    Returns a ``fiftyone.Dataset``. Requires the ``data`` dependency group
    (``uv sync --group data``). Used by the detection module for COCO mAP.
    """
    import fiftyone as fo
    import fiftyone.zoo as foz

    name = f"coco-2017-val-{max_samples}"
    if name in fo.list_datasets():
        return fo.load_dataset(name)

    ds = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=max_samples,
        shuffle=True,
        seed=seed,
        dataset_name=name,
    )
    ds.persistent = True
    return ds
