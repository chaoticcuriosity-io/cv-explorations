# Bootstrap the cv-explorations environment and verify the GPU.
# Usage:  pwsh scripts/setup_env.ps1
$ErrorActionPreference = "Stop"

Write-Host "==> uv sync (core + dev)" -ForegroundColor Cyan
uv sync --group dev

Write-Host "==> GPU smoke test" -ForegroundColor Cyan
uv run python scripts/smoke_gpu.py

Write-Host "==> cvkit unit tests" -ForegroundColor Cyan
uv run pytest -q

Write-Host "Done. Next: uv sync --group data --group detection  (for module 01)" -ForegroundColor Green
