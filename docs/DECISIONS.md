# Decisions

## 2026-08-14 — Narrow TypeScript boundary

Accept TypeScript/TSX and useful human-authored declarations; exclude JavaScript and
other implementation languages to keep the first model's objective measurable.

## 2026-08-14 — Byte-level BPE and custom PyTorch decoder

Train a byte-level BPE on the approved corpus and implement the decoder directly in
PyTorch. This preserves arbitrary UTF-8 round trips and the from-scratch boundary.

## 2026-08-14 — Conservative provenance and licensing

Use append-only manifests, repository-family splits, and an explicit permissive
license allowlist. Ambiguous sources require review rather than optimistic inclusion.

## 2026-08-14 — Controlled fixtures before scale

Milestone 1 proves correctness locally. Corpus downloads, paid compute, and smoke or
larger training require explicit approval after stop-condition checks pass.

## 2026-08-14 — Bounded Git-object acquisition

Acquire approved GitHub commits into ignored immutable snapshots by reading verified
Git objects from an isolated bare repository. Do not create a worktree or run hooks,
filters, submodules, package managers, or repository code. Bind each snapshot to an
atomic content receipt and enforce its approval scope during ingestion.

## 2026-08-14 — Pilot tokenizer before production tokenizer

Use an 8,192-token approved-pilot vocabulary for the initial three-repository corpus.
Keep the 32K production tokenizer frozen until the corpus and held-out evaluation set
are materially larger and reviewed.

## 2026-08-14 — Second controlled source tranche

Add typescript-eslint for parser and static-analysis patterns, TypeORM for typed
application/data-access code, and Redux Toolkit for modern TypeScript and TSX. Keep
OpenFable outside the corpus: it is a Python RAG system depending on external LLMs
and embeddings, not TypeScript base-model training data.
