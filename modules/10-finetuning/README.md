# 10 · Fine-tuning (air-vent detector, trained on the DGX Spark)

> Status: **in progress**. The capstone: every other module *runs* a pretrained model;
> this one *trains* one — for a class that doesn't exist in COCO.

**Goal:** a YOLO detector that finds **air vents**, a category no off-the-shelf model knows.

## The twist: no manual labelling

We never hand-label a single box. Instead we **manufacture the training set from the repo's
own earlier capabilities** (the workflow module 01 promised):

1. **Source** Creative-Commons vent images from Wikimedia Commons (no API key).
2. **Filter** them with **CLIP zero-shot** (module 02) — keep vent-like images, reject
   smokestacks / landscapes / documents / interiors.
3. **Auto-label** the survivors with **Grounding DINO** (module 01), prompted with vent phrases.
4. **Fine-tune** YOLO on the result — **on the DGX Spark** (GB10, the training/scale-up box).

So three earlier modules (open-vocab detection, zero-shot classification) become a *data
engine* for a fourth. The honest caveat: auto-sourced + auto-labelled data is noisy, which
caps reliability — the report is explicit about this and the path to production quality.

## Pipeline

```bash
# 1. Build the dataset locally (Grounding DINO + CLIP run on the 3090 Ti)
uv run python modules/10-finetuning/prepare_data.py --max-images 200

# 2. Train on the Spark (see report for the transfer + docker-exec commands)
python finetune.py --data data/airvents/data.yaml --model yolo11s.pt --epochs 80 --out out
```

See the [report](report.qmd) for results (before→after) and the full method.
