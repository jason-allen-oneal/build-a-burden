#!/usr/bin/env bash
set -euo pipefail
exec uv run python -m ts_coder.cli corpus build --config configs/data/dev.yaml --replace
