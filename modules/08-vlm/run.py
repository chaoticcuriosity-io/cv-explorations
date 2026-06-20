"""VLM engine: one model, many tasks. Caption / reason / count / read text via prompts.

Writes a real Q&A transcript (results/transcript.md), an image+caption panel, and a latency
note into ``results/``.

uv run python modules/08-vlm/run.py --images 3
"""

from __future__ import annotations

import argparse
import glob
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import vlm
from cvkit import describe_device, get_device
from cvkit.io import load_image
from cvkit.report import markdown_table, write_metrics_md
from cvkit.timing import benchmark

HERE = Path(__file__).parent
RESULTS = HERE / "results"
COCO = r"C:/Users/dgbal/fiftyone/coco-2017/validation/data/*.jpg"

TASKS = [
    ("Caption", "Describe this image in one sentence."),
    ("Reasoning", "What is happening in this scene, and what might happen next?"),
    ("Counting", "How many people are in this image? Answer with just a number."),
    ("Reading (OCR)", "Is there any readable text in this image? If yes, quote it; if no, say 'none'."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {describe_device(device)}")
    paths = sorted(glob.glob(COCO))[: args.images]
    images = [load_image(p) for p in paths]
    model = vlm.build_vlm(device)
    RESULTS.mkdir(parents=True, exist_ok=True)

    transcript = ["## Qwen2.5-VL 3B — one model, many tasks", ""]
    captions = []
    for path, img in zip(paths, images):
        transcript.append(f"### `{Path(path).name}`")
        for i, (name, q) in enumerate(TASKS):
            ans = model.ask(img, q, max_new_tokens=128)
            transcript.append(f"- **{name}** — *{q}*\n  > {ans}")
            if i == 0:
                captions.append(ans)
        transcript.append("")
        print(f"  {Path(path).name}: {captions[-1][:70]}")
    (RESULTS / "transcript.md").write_text("\n".join(transcript) + "\n", encoding="utf-8")

    # Panel: image + caption.
    k = len(images)
    fig, axes = plt.subplots(1, k, figsize=(4.3 * k, 4.6))
    for ax, img, cap in zip(np.atleast_1d(axes).ravel(), images, captions):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("\n".join(textwrap.wrap(cap, 40)), fontsize=8)
    fig.suptitle("Qwen2.5-VL captions", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "captions.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    lat = benchmark(lambda: model.ask(images[0], "Describe this image.", max_new_tokens=32),
                    device=device, warmup=1, runs=5)
    headers = ["Model", "License", "Latency (ms/answer, 32 tok)", "Note"]
    table = [[model.meta.display, model.meta.license, lat.as_row()["mean_ms"], "free-form text; qualitative"]]
    notes = (
        f"*One model answers caption/reason/count/OCR purely from the prompt — no task-specific "
        f"heads. Latency is per short answer on `{describe_device(device)}` and scales with the "
        f"number of generated tokens. See the transcript for real outputs.*"
    )
    write_metrics_md(RESULTS / "metrics.md", title="Vision-language model", headers=headers, rows=table, notes=notes)
    (RESULTS / "metrics.json").write_text(json.dumps({"latency_ms_32tok": lat.as_row()["mean_ms"]}, indent=2), encoding="utf-8")
    print("\n" + markdown_table(headers, table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
