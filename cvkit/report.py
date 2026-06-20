"""Emit benchmark tables and figures that Quarto reports embed verbatim.

Reports stay reproducible because the numbers are written to disk by the engine
(``run.py``) and the ``.qmd`` just includes them -- the heavy GPU work never runs
at site-render time (that is what Quarto's ``freeze`` relies on).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join(
        "| " + " | ".join("" if c is None else str(c) for c in row) + " |"
        for row in rows
    )
    return "\n".join([head, sep, body])


def write_metrics_md(
    path: str | Path,
    *,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    notes: str | None = None,
) -> Path:
    """Write a titled metrics table (+ optional notes) to a Markdown fragment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"## {title}", "", markdown_table(headers, rows)]
    if notes:
        parts += ["", notes]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path
