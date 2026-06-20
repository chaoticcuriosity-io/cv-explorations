# 02 · Image classification (supervised vs zero-shot)

> Status: **in progress (M2)**.

**Question:** *what is this image, as a whole?* — the foundational CV task, and the
cleanest place to see the **supervised → zero-shot** shift.

## What this module benchmarks

| Model | Paradigm | License | Idea |
|---|---|---|---|
| ConvNeXt-Tiny (timm) | **Supervised** | Apache-2.0 | A classifier head trained *on* ImageNet classes |
| SigLIP 2 (Base) | **Zero-shot** | Apache-2.0 | Match the image against text `"a photo of a {class}"` — never trained on this dataset |
| OpenCLIP ViT-B/32 | **Zero-shot** | MIT | Same idea, the original CLIP contrastive recipe |

Evaluated on **Imagenette** (a 10-class, Apache-licensed subset of ImageNet) — top-1
accuracy and warmed-up latency on the local GPU.

The headline lesson: a zero-shot model that has **never seen the training set** can rival
a model trained directly on those classes — the classification analogue of
open-vocabulary detection in [module 01](../01-object-detection/report.qmd).

## Run (once built)

```bash
uv sync --group dev   # timm, open_clip, transformers, torchvision are all in core
uv run python modules/02-classification/run.py
```
