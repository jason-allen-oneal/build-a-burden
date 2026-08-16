# Evaluation

The unified harness writes machine-readable results for validation cross-entropy,
perplexity, FIM exact match/token accuracy, syntax parse rate, compilation rate and
diagnostics, generation length/repetition, throughput, peak memory, and deterministic
generation. Metrics that require an unavailable timing source or task oracle are
written as JSON `null`, never fabricated. TypeScript parser/compiler helpers consume
and emit JSON. Controlled fixtures cover TS/TSX syntax, cross-file imports,
diagnostics/fixes, and FIM spans.

Use `completion-evaluate` for ordinary causal prompts and `fim-evaluate` for the
objective-aware serialization:

```bash
PYTHONPATH=src python -m ts_coder.cli fim-evaluate \
  --checkpoint artifacts/runs/example/checkpoints/latest \
  --tokenizer artifacts/tokenizers/example/tokenizer.json \
  --tasks fixtures/evaluation/fim-tasks.json \
  --output artifacts/evaluations/example-fim.json
```

FIM evaluation reconstructs `prefix + generated_middle + suffix`, measures exact
and token-level middle accuracy, and then parses/compiles the reconstructed source.
Control markers are stop boundaries, not source repair. The fixture report is local
evidence and is not an independent benchmark. The pinned TypeScript helper preserves
TSX syntax and supplies only a minimal intrinsic-element declaration; it does not
install or execute a React runtime.

Memorization checks report exact generated substrings, longest token match, exact
file reproduction, fixture contamination, and exact/near-duplicate cross-split
overlap. A match is evidence to review, not by itself proof of unacceptable
memorization. Future interfaces cover unit-test pass@k, repository context, imports,
patch application, API preservation, security tasks, AST equivalence, and IDE
acceptance. No result is reported as passing unless produced by the harness.
