#!/usr/bin/env bash
set -euo pipefail
exec uv run python -m ts_coder.cli tokenizer train --config configs/tokenizer/dev.yaml
