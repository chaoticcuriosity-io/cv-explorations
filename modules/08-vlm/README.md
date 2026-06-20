# 08 · Vision-language models

> Status: **done**. One model, many tasks — caption / reason / count / OCR by prompt.

| Model | License | Note |
|---|---|---|
| Qwen2.5-VL 3B | Apache-2.0 | transformers-native generative VLM |

Real Q&A transcripts + a caption panel + latency (~3.7 s/answer — the heavyweight of the
curriculum). Florence-2 was the intended unified model but its remote code doesn't load under
transformers v5; Qwen2.5-VL is the native substitute. See the [report](report.qmd).

```bash
uv sync --group dev
uv run python modules/08-vlm/run.py --images 3
```
