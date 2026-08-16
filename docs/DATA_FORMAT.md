# Data Format

The Milestone 1 manifest is append-only JSONL using the schema in the project
specification. Stable IDs derive from repository identity, revision, and relative
path; content and normalized content use SHA-256. Larger corpora may additionally
materialize Parquet manifests and token shards while JSONL remains the audit trail.

Repository examples preserve boundaries and paths:

```text
<repo><file path="src/types.ts">…</file><file path="src/main.ts">…</file></repo>
```

Files are ordered by import dependency or relevance and unrelated repositories are
never packed into one logical example. Original source bytes remain unchanged by
dedup normalization.

## Scale-out shard contract

The small vertical slice may read `documents.jsonl` directly, but a serious run
must consume an immutable shard index rather than materialize the corpus in one
process. `ts_coder.data.streaming` defines the v1 contract:

- each shard is relative to a fixed root and contains one validated JSON object
  per line, with `record.record_id` and `text`;
- the index records `shard_id`, relative path, record count, optional token
  count, and SHA-256, plus the tokenizer and source-manifest hashes;
- a `DataCursor` is `(epoch, shard_index, record_offset, token_offset, rank,
  world_size)` and is part of resume identity;
- workers partition by a stable SHA-256 hash of `record_id`, never by a
  process-local iterator order; each worker must see a disjoint union;
- readers enforce regular-file paths, root containment, and a per-line byte
  limit while streaming, so malformed or oversized records fail closed.

The current training CLI still uses the materialized fixture path. Before a
25M/100M run, replace that adapter with a token-shard loader that emits the same
cursor and accounting fields; do not infer resume position from an integer
batch counter.

`ts_coder.data.token_stream.TokenizedStreamingDataset` is the first bounded
memory adapter for this contract. It reads one shard record, applies the stable
causal/FIM choice, tokenizes that record, and emits fixed-length windows before
discarding the source text. `TokenizedStreamingBatcher` groups those windows
into ordinary model mappings while carrying `data_position`,
`shard_manifest_hash`, and objective counts as non-model metadata. The trainer
commits that cursor only after a successful optimizer step and automatically
persists it in a format-v2 checkpoint. A resumed job must call `seek()` with
the checkpoint's `data_position` after verifying tokenizer and shard hashes.
