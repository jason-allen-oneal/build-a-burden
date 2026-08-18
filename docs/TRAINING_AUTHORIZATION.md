# Training authorization

Source intake approval and training authorization are separate controls.
`manifests/approved-sources.jsonl` authorizes exact source URI/revision pairs for
bounded corpus intake. It does not automatically authorize a large GPU run,
redistribution, or publication of weights.

When a training configuration sets `approval_required: true`, preflight verifies
that every included URI/revision in the derived corpus manifest remains approved.
A run requesting more than 1,000,000 input tokens also requires an exact record in:

```text
manifests/training-authorizations.jsonl
```

The ledger is append-only JSONL. Each record binds approval to all of the following:

- run name
- complete training configuration SHA-256
- model configuration SHA-256
- derived corpus manifest SHA-256
- derived corpus artifact SHA-256
- tokenizer SHA-256
- maximum requested input-token budget
- approval status, approver, and timestamp

A copied approval does not survive a configuration, model, corpus, tokenizer, or
budget change. Append a new decision instead of editing an existing record.

## Creating a record

Run this only after reviewing the resolved run contract and its data/release terms:

```bash
uv run python scripts/authorize_training.py \
  --run-name approved-a100-300m \
  --training-config configs/training/approved-a100-300m.yaml \
  --model-config configs/model/approved-300m.yaml \
  --manifest manifests/approved.jsonl \
  --corpus artifacts/corpus/approved/documents.jsonl \
  --tokenizer artifacts/tokenizers/approved-32k-v2/tokenizer.json \
  --max-tokens 300000000 \
  --approved-by REVIEWER \
  --approved-at YYYY-MM-DDTHH:MM:SSZ \
  --notes "State the compute, data, retention, and release authorization."
```

The authorization ledger may contain later `rejected` or `revoked` decisions for
the same exact identity. The latest exact decision controls.

## Required review

Before approving a costly run, confirm at minimum:

1. The corpus is large and diverse enough for the selected parameter count and
   token budget.
2. Train, validation, and test families are isolated and contamination checks are
   current.
3. Tokenizer, corpus, shard, model, and checkpoint lineage is reproducible.
4. Persistent checkpoint storage, interruption recovery, and deletion policy are
   defined.
5. Held-out syntax, compilation, FIM, memorization, and security gates are defined.
6. Weight publication and corpus redistribution are explicitly allowed or denied.

The current filtered corpus is a pipeline-validation corpus. It does not, by itself,
justify the full 300M or 500M configurations.
