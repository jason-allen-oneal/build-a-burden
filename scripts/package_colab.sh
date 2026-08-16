#!/usr/bin/env bash
set -euo pipefail

# Package only source, configuration, approved provenance, the rebuilt corpus,
# and the tokenizer. Never upload raw source snapshots, prior run artifacts,
# virtual environments, caches, or local credentials to a notebook runtime.
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir="$project_root/artifacts/colab"
package_suffix="${COLAB_PACKAGE_SUFFIX:-v12}"
tokenizer_artifact="${COLAB_TOKENIZER_ARTIFACT:-artifacts/tokenizers/approved-32k-v2}"
output="$output_dir/ts-coder-a100-inputs-${package_suffix}.tar.gz"
mkdir -p "$output_dir"

for required in pyproject.toml uv.lock src tests configs fixtures/repositories fixtures/evaluation \
  manifests/approved.jsonl artifacts/corpus/approved "$tokenizer_artifact"; do
  test -e "$project_root/$required" || {
    printf 'missing required package input: %s\n' "$required" >&2
    exit 1
  }
done

if find "$project_root/src" "$project_root/tests" "$project_root/configs" \
  "$project_root/manifests/approved.jsonl" "$project_root/artifacts/corpus/approved" \
  "$project_root/$tokenizer_artifact" "$project_root/tools" \
  \( -path '*/.env*' -o -name '*.pem' -o -name '*.key' \) -print -quit | grep -q .; then
  printf 'refusing package: credential-like files exist under project\n' >&2
  exit 1
fi

tar -czf "$output" -C "$project_root" \
  --exclude='*/node_modules' --exclude='*/node_modules/*' \
  --exclude='*/__pycache__' --exclude='*/__pycache__/*' \
  --exclude='*.pyc' --exclude='*.pyo' --exclude='.pytest_cache' \
  --exclude='.mypy_cache' --exclude='.ruff_cache' \
  pyproject.toml uv.lock LICENSE SECURITY.md README.md \
  src tests configs fixtures manifests/approved.jsonl \
  artifacts/corpus/approved "$tokenizer_artifact" tools

printf '%s\n' "$output"
sha256sum "$output"
