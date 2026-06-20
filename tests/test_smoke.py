"""Smoke tests for cvkit — no GPU or network required.

These verify the shared toolkit's pure logic (I/O round-trip, timing, metrics,
report formatting) so module work builds on a trusted base. GPU-specific paths
are checked separately by the env smoke test in scripts/setup_env.ps1.
"""

from __future__ import annotations

import numpy as np
import torch

from cvkit import describe_device, get_device
from cvkit.io import load_image, save_image
from cvkit.metrics import topk_accuracy
from cvkit.report import markdown_table, write_metrics_md
from cvkit.timing import benchmark


def test_get_device_returns_torch_device():
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}


def test_describe_device_records_torch_version():
    info = describe_device(torch.device("cpu"))
    assert info.torch_version == torch.__version__
    assert "torch" in str(info)


def test_benchmark_positive_latency():
    stats = benchmark(lambda: sum(range(1000)), device=torch.device("cpu"), warmup=1, runs=5)
    assert stats.runs == 5
    assert stats.mean_ms >= 0.0
    assert stats.fps > 0.0


def test_image_roundtrip(tmp_path):
    img = (np.random.default_rng(0).integers(0, 256, (16, 24, 3))).astype("uint8")
    p = save_image(tmp_path / "x.png", img)
    back = load_image(p)
    assert back.shape == (16, 24, 3)
    assert np.array_equal(back, img)  # PNG is lossless


def test_topk_accuracy_perfect_and_partial():
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    targets = torch.tensor([1, 0])
    acc = topk_accuracy(logits, targets, ks=(1,))
    assert acc[1] == 100.0


def test_markdown_table_and_metrics_file(tmp_path):
    table = markdown_table(["model", "mAP"], [["rfdetr", 0.547], ["yolo26n", 0.41]])
    assert table.count("\n") == 3  # header + separator + 2 rows
    out = write_metrics_md(
        tmp_path / "metrics.md",
        title="Detection",
        headers=["model", "mAP"],
        rows=[["rfdetr", 0.547]],
    )
    text = out.read_text(encoding="utf-8")
    assert "## Detection" in text and "rfdetr" in text
