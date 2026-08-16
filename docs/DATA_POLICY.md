# Data Policy

## Scope and provenance

Before a real source enters intake, it must pass the explicit approval ledger
procedure in [APPROVED_SOURCES.md](APPROVED_SOURCES.md). The ledger records the
exact revision, credential-free source URI, license decision, reviewer, scope,
and later removals. The approved-source dry run performs no downloads or code
execution.

The initial corpus is 85–90% TypeScript code/tests and 10–15% English documentation,
diagnostics, tasks, and limited repository metadata. Primary code is `.ts`, `.tsx`,
and useful human-authored `.d.ts`; JavaScript and other implementation languages are
excluded. Every accepted and rejected file receives an append-only JSONL manifest
record with source identifier, exact revision, retrieval time, license result, paths,
hashes, quality and safety results, dedup cluster, split, pipeline version, decision,
and exclusion reasons. Changed decisions create versioned replacement records.

Explicitly configured MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD,
CC0-1.0, and Unlicense sources may be accepted. Missing, custom, proprietary,
source-available, GPL/LGPL/AGPL, MPL, EPL, unclear, or conflicting licenses are
rejected or held for review. This is conservative project policy, not legal advice.

## Filtering and removal

Deterministic stages are path/language checks, generated-code detection, secret
scanning, quality analysis, exact/normalized/near deduplication, then grouped
splitting. Reject dependencies, build output, caches, bundles, source maps, lockfiles,
large snapshots, invalid encodings, probable credentials, and unsafe paths. Never log
secret values. Exact and near-duplicate families remain in one split; repository
families are split 90/5/5 rather than individual files.

Repository removal marks its records removed, excludes all family members, rebuilds
affected shards, regenerates hashes, and records the replacement manifest. Original
source is never silently mutated. Benchmark sources are quarantined. No weights are
published until corpus and release policy review.

Approval decisions are append-only in `manifests/approved-sources.jsonl`. Derived
corpus manifests use `append_manifest: false`; a repeat build refuses to overwrite an
existing manifest unless `--replace` is explicit. Material changes should use a new
versioned output path and retain prior manifest hashes.
