#!/usr/bin/env bash
set -euo pipefail
exec uv run python -m ts_coder.cli train --config configs/training/dev.yaml
