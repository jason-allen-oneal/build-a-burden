# ts-coder

`ts-coder` is a TypeScript-only decoder language-model pipeline. Its tokenizer is
trained on the approved project corpus, its Transformer is implemented directly in
PyTorch, and every model starts from randomly initialized weights. Base pretraining
uses no pretrained weights, tokenizer, embeddings, adapters, distillation, or hosted
model API.

## Status and platform

Milestone 1 is complete as a local correctness slice over controlled fixtures.
Milestone 2 now has a bounded seven-repository approved-source pilot covering Zod,
Vite, NestJS, typescript-eslint, TypeORM, Redux Toolkit, and Microsoft TypeScript
at immutable commits. A 304M-parameter A100 run and held-out completion/FIM gates
are operational, but the resulting model is not usable: unseen syntax and compile
rates remain zero. This remains an intake, tokenizer, and training-correctness
exercise, not a production corpus. Development targets Linux/macOS, Python
3.11–3.13, Node.js, `uv`, and npm; CPU is required and a single CUDA GPU is
optional.

## Setup and commands

```bash
make setup
make lint
make typecheck
make test
make security
make node-build
make reproduce-dev
```

Stages (via Makefile wrappers):
```bash
make fixture-corpus
make tokenizer-dev
make train-dev
make evaluate-dev
```

Equivalent direct CLI commands:

```bash
# Development fixture corpus
uv run python -m ts_coder.cli corpus build --config configs/data/dev.yaml --replace

# Development tokenizer
uv run python -m ts_coder.cli tokenizer train --config configs/tokenizer/dev.yaml

# Development training
uv run python -m ts_coder.cli train --config configs/training/dev.yaml

# Development evaluation
uv run python -m ts_coder.cli evaluate --config configs/evaluation/default.yaml

# Approved-source corpus (approval-gated, Milestone 2)
uv run python -m ts_coder.cli corpus approvals validate \
  --manifest manifests/approved-sources.jsonl
uv run python -m ts_coder.cli corpus acquire \
  --config configs/data/approved.yaml --dry-run
uv run python -m ts_coder.cli corpus acquire \
  --config configs/data/approved.yaml
uv run python -m ts_coder.cli corpus build \
  --config configs/data/approved.yaml --dry-run
uv run python -m ts_coder.cli corpus build \
  --config configs/data/approved.yaml
uv run python -m ts_coder.cli tokenizer train \
  --config configs/tokenizer/approved-pilot.yaml

# Approved-source training configurations
# Smoke test (small):       configs/training/approved-a100-300m-smoke.yaml
# Packed streaming (1M):    configs/training/approved-a100-300m-packed-1m.yaml
# Filtered streaming (300K): configs/training/approved-a100-300m-filtered-300k.yaml
# Probe:                    configs/training/approved-a100-300m-probe.yaml
# Full 300M:                configs/training/approved-a100-300m.yaml
# Full 500M:                configs/training/approved-a100-500m.yaml

# Model audit (dry-run configuration analysis without allocating weights)
uv run python -m ts_coder.cli model-audit \
  --config configs/model/smoke-25m.yaml \
  --output artifacts/model-audits/smoke-25m.json
uv run python -m ts_coder.cli model-audit \
  --config configs/model/approved-500m.yaml \
  --output artifacts/model-audits/approved-500m.json

# Generation (writes TypeScript to stdout, metadata to stderr)
uv run python -m ts_coder.cli generate \
  --checkpoint artifacts/runs/example/checkpoints/latest \
  --prompt-file examples/prompt.ts --max-new-tokens 256 \
  --temperature 0.2 --top-k 50 --seed 42

# Objective-aware FIM evaluation on held-out fixture set
uv run python -m ts_coder.cli fim-evaluate \
  --checkpoint artifacts/runs/example/checkpoints/latest \
  --tokenizer artifacts/tokenizers/example/tokenizer.json \
  --tasks fixtures/evaluation/fim-tasks.json \
  --output artifacts/evaluations/example-fim.json

# Completion evaluation on held-out fixture set
uv run python -m ts_coder.cli completion-evaluate \
  --checkpoint artifacts/runs/example/checkpoints/latest \
  --tokenizer artifacts/tokenizers/example/tokenizer.json \
  --tasks fixtures/evaluation/completion-tasks.json \
  --output artifacts/evaluations/example-completion.json
```

## CLI command reference

| Command | Description |
|---------|-------------|
| `corpus build` | Build corpus from sources (dev or approved) |
| `corpus acquire` | Acquire approved Git sources (requires approval manifest) |
| `corpus approvals validate` | Validate approval manifest statuses |
| `corpus approvals add` | Add approval record to manifest |
| `tokenizer train` | Train BPE tokenizer from corpus |
| `train` | Train model from config |
| `evaluate` | Evaluate model (generation, perplexity, FIM, syntax, compilation) |
| `generate` | Generate TypeScript from checkpoint |
| `completion-evaluate` | Held-out completion task evaluation |
| `fim-evaluate` | Held-out FIM (fill-in-middle) task evaluation |
| `model-audit` | Dry-run model configuration audit |

Key options:
- `corpus build --source-id` — filter to specific source(s)
- `corpus build --dry-run` — validate and summarize without writing
- `corpus acquire --dry-run` — validate approvals and paths without network
- `evaluate --checkpoint` / `--source` / `--config` / `--output`
- `generate --tokenizer` — optional tokenizer path (falls back to byte encoding)
- `model-audit --config` / `--output`

## Layout

- `configs/`: data, tokenizer, model, training, and evaluation settings.
- `docs/`: policies, architecture, threats, and experiment records.
- `fixtures/repositories/`: controlled, project-authored fixtures.
- `fixtures/evaluation/`: held-out completion and FIM task fixtures.
- `src/ts_coder/`: implementation; `tools/typescript/`: Node evaluation helpers.
- `tests/`: unit, integration, and security coverage.
- `artifacts/` and `manifests/`: generated ignored output.

## Roadmap and warnings

Next: expand the reviewed corpus to at least a frozen 50–100M-token tranche and
freeze a real validation/test suite before a larger 25M/100M experiment. A 1B-plus
model requires data, scaling, compute, and release review. Treat downloads, archives,
dependencies, and checkpoints as hostile. Never install or execute harvested
projects on the host. The license policy in `docs/DATA_POLICY.md` is conservative
project policy, not legal advice. Generated code may be incorrect or vulnerable. Do
not publish weights before corpus and release review, and do not present plumbing
metrics as capability.