# 07 · Embeddings & retrieval

> Status: **done**. Image → vector → nearest-neighbour search (a *real* metric).

| Model | Type | License |
|---|---|---|
| DINOv2 Base | Self-supervised (images only) | Apache-2.0 |
| SigLIP 2 Base | Image-text contrastive | Apache-2.0 |

Recall@1 / Precision@5 on a balanced Imagenette gallery (both >99%, no labels at search time)
+ a query→neighbours figure + embedding latency. See the [report](report.qmd).

```bash
uv sync --group dev
uv run python modules/07-retrieval/run.py --per-class 60
```
