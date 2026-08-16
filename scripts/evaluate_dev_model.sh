#!/usr/bin/env bash
set -euo pipefail
exec uv run python -m ts_coder.cli evaluate --config configs/evaluation/default.yaml
