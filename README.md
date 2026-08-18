# ts-coder

`ts-coder` is a TypeScript-only decoder language-model pipeline. Its tokenizer is
trained on an approved project corpus, its Transformer is implemented directly in
PyTorch, and every model starts from randomly initialized weights. Base pretraining
uses no pretrained weights, tokenizer, embeddings, adapters, distillation, or hosted
model API.

## Status and platform

Milestone 1 is complete as a local correctness slice over controlled fixtures.
Milestone 2 has a bounded seven-repository approved-source pilot covering Zod,
Vite, NestJS, typescript-eslint, TypeORM, Redux Toolkit, and Microsoft TypeScript
at immutable commits. A 304M-parameter A100 run and held-out completion/FIM gates
are operational, but the resulting model is not usable: unseen compilation remains
zero. This remains an intake, tokenizer, and training-correctness exercise, not a
production corpus.

Development targets Linux/macOS, Python 3.11-3.13, Node.js, `uv`, and npm. CPU is
required and a single CUDA GPU is optional.

## Setup and checks

```bash
make setup
make lint
make typecheck
make test
make security
make node-build
make reproduce-dev
```

The development stages are also available individually:

```bash
make fixture-corpus
make tokenizer-dev
make train-dev
make evaluate-dev
```

## Development pipeline

```bash
uv run python -m ts_coder.cli corpus build \
  --config configs/data/dev.yaml --replace

uv run python -m ts_coder.cli tokenizer train \
  --config configs/tokenizer/dev.yaml

uv run python -m ts_coder.cli train \
  --config configs/training/dev.yaml

uv run python -m ts_coder.cli evaluate \
  --config configs/evaluation/default.yaml
```

The tokenizer command trains only on records assigned to the training split and
records the corpus artifact, manifest, configuration, document count, byte count,
and SHA-256 lineage in its metadata.

## Approved-source pipeline

Approved intake is fail-closed. Acquisition uses exact URI/revision decisions,
reads bounded Git blobs without a worktree, and never executes harvested repository
code.

```bash
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
  --config configs/tokenizer/approved-32k-v2.yaml
```

The canonical approved tokenizer is `artifacts/tokenizers/approved-32k-v2`. The
active full-run 300M and 500M configurations use that path, packed streaming, and a
versioned v2 shard manifest. Historical smoke, probe, and experiment configurations
remain unchanged when their recorded hashes depend on older artifacts.

Approved training configurations include:

```text
configs/training/approved-a100-300m-smoke.yaml
configs/training/approved-a100-300m-probe.yaml
configs/training/approved-a100-300m-filtered-300k.yaml
configs/training/approved-a100-300m-packed-1m.yaml
configs/training/approved-a100-300m.yaml
configs/training/approved-a100-500m.yaml
```

`approval_required: true` now has operational meaning. The trainer verifies that
included source URI/revision pairs remain approved. Runs above 1,000,000 requested
tokens additionally require an exact structured authorization in
`manifests/training-authorizations.jsonl`. Source approval alone does not authorize
an expensive run or weight publication. See `docs/TRAINING_AUTHORIZATION.md`.

## Model audit

Model audits use arithmetic and do not allocate the model:

```bash
uv run python -m ts_coder.cli model-audit \
  --config configs/model/smoke-25m.yaml \
  --output artifacts/model-audits/smoke-25m.json

uv run python -m ts_coder.cli model-audit \
  --config configs/model/approved-500m.yaml \
  --output artifacts/model-audits/approved-500m.json
```

## Generation and evaluation

Checkpoints are loaded as a bundle. The tokenizer SHA-256 and vocabulary size must
match checkpoint metadata. There is no modulo remapping or byte-tokenizer fallback.
The normal default tokenizer is the `tokenizer.json` copied into the run directory.
Successful runs also write preflight and post-training lineage records described in
`docs/RUN_BUNDLE.md`.

```bash
uv run python -m ts_coder.cli generate \
  --checkpoint artifacts/runs/example/checkpoints/latest \
  --prompt-file examples/prompt.ts \
  --max-new-tokens 256 \
  --temperature 0.2 \
  --top-k 50 \
  --seed 42
```

An explicit tokenizer can be supplied when evaluating a copied checkpoint:

```bash
uv run python -m ts_coder.cli completion-evaluate \
  --checkpoint artifacts/runs/example/checkpoints/latest \
  --tokenizer artifacts/tokenizers/approved-32k-v2/tokenizer.json \
  --tasks fixtures/evaluation/completion-tasks.json \
  --output artifacts/evaluations/example-completion.json

uv run python -m ts_coder.cli fim-evaluate \
  --checkpoint artifacts/runs/example/checkpoints/latest \
  --tokenizer artifacts/tokenizers/approved-32k-v2/tokenizer.json \
  --tasks fixtures/evaluation/fim-tasks.json \
  --output artifacts/evaluations/example-fim.json
```

The general `evaluate` command honors the configured manifest, split, task fixtures,
metric selection, TypeScript tool root, compile timeout, generation settings, and
evaluation token budget. Cross-entropy is weighted by supervised token count.

## CLI command reference

| Command | Description |
|---------|-------------|
| `corpus build` | Build a corpus from configured sources |
| `corpus acquire` | Acquire exact approved Git sources |
| `corpus approvals validate` | Validate the source approval ledger |
| `corpus approvals add` | Append a source approval decision |
| `tokenizer train` | Train and record a BPE tokenizer lineage |
| `train` | Train from a configuration after preflight gates |
| `evaluate` | Run configured held-out evaluation |
| `generate` | Generate TypeScript from a verified run bundle |
| `completion-evaluate` | Run held-out causal completion fixtures |
| `fim-evaluate` | Run held-out fill-in-the-middle fixtures |
| `model-audit` | Audit model size without allocating weights |

## Layout

- `configs/`: data, tokenizer, model, training, and evaluation settings.
- `docs/`: policies, architecture, threats, authorization, and experiment records.
- `fixtures/repositories/`: controlled, project-authored fixtures.
- `fixtures/evaluation/`: held-out completion and FIM task fixtures.
- `src/ts_coder/`: Python implementation.
- `tools/typescript/`: pinned Node evaluation helpers.
- `tests/`: unit, integration, and security coverage.
- `artifacts/`: generated and ignored outputs.
- `manifests/approved-sources.jsonl`: tracked source approval ledger.
- `manifests/training-authorizations.jsonl`: optional tracked expensive-run ledger.

## Roadmap and warnings

The next useful work is data and evaluation work, not another parameter increase.
Expand and freeze a substantially larger distinct approved corpus, keep repository
families isolated, and establish a real validation/test suite before another large
run. The current approximately 7.38M-token filtered corpus validates the pipeline
but does not justify a 300M or 500M full run.

Treat downloads, archives, dependencies, shards, tokenizers, and checkpoints as
hostile. Never execute harvested projects on the host. Generated code may be
incorrect or vulnerable. Do not publish weights before corpus and release review,
and do not present plumbing metrics as capability.
