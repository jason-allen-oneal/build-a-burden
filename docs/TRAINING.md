# Training

Approved documents are filtered, grouped and split, serialized as isolated-file or
repository examples, tokenized, packed, and deterministically mixed between causal
and FIM objectives. FIM uses `<fim_prefix> PREFIX <fim_suffix> SUFFIX <fim_middle>
MIDDLE`, stays within one file, and derives spans from sample ID plus seed.

Training uses AdamW, warmup plus cosine decay, token-based progress, gradient
accumulation/clipping, BF16 when supported and FP32 fallback. Validation, generation,
and atomic checkpoints occur at configured token intervals. Checkpoints contain all
model/optimizer/scheduler/scaler state, counters, data cursor, RNG states, resolved
configuration and artifact hashes. Resume restores the exact cursor and RNG state.

Every run records resolved config, environment, Git/dirty state, tokenizer and
manifest metadata, JSONL metrics, samples, checkpoints, and a report. Before scaling,
the model must overfit a batch and tiny dataset, resume without cursor drift, and pass
leakage and KV-cache equivalence tests. The 25M/100M configs require explicit approval;
the 1B plan is recalculated from observed scaling and approved resources.

## Execution tiers

The local CPU trainer is the reference correctness path, not the production
throughput path. Local runs are limited to the 5M development model and a bounded
25M smoke run. A useful 100M–300M pilot requires a GPU host; the approximately
1.13B architecture is planning-only until a multi-GPU memory and communication plan
exists. No configuration may infer permission to launch paid or distributed compute
from its parameter count alone.

The current filtered approved corpus contains 7,376,774 tokenizer tokens
(train 6,069,058 / validation 733,966 / test 573,750) using the v2 32K
tokenizer. It contains 15,070 included records (13,670 train / 988 validation /
412 test) from five train repository families, one validation family, and one
test family. The source manifest remains the append-only audit record with
99,427 records; compiler/fourslash harness records are rejected from the
training corpus rather than silently treated as ordinary TypeScript. It is a
pipeline fixture at real-source quality, not a training corpus at pilot scale.
Before a GPU
pilot, the data program must add approved TypeScript repositories, preserve
repository-family and near-duplicate isolation, and publish a token-count and
license-distribution report. A 300M diagnostic run is blocked below a corpus-sized
budget of hundreds of millions of distinct tokens; a serious 1.13B run is blocked
until the corpus and token budget are reviewed together. Repeating the same ~8.9M
tokens longer would measure memorization and optimizer behavior, not useful model
scaling.

The next data gate is a frozen 50–100M-token tranche, not a longer replay of this
7.38M-token set. The current candidate order is FuelLabs/fuels-ts and
puppeteer/puppeteer, then narrow-scope reviews of Angular and Storybook; VS Code
remains a later repository-context tranche. Every candidate still requires an exact
revision, nested-license review, source-scope record, secret scan, harness/generated
filter report, and split-overlap check. `alainbrown/openfable` is intentionally not
included: its observed tree is primarily Python, so it is outside the TypeScript
base-model boundary.

The intended compute sequence is:

1. **Local reference:** keep the 5M–25M CPU path for correctness, resume, and
   evaluation regressions.
2. **GPU pilot:** run the planning-only 304M architecture on an approved single
   GPU (24 GB is the sane starting point; an 80 GB A100 gives ample headroom)
   after the corpus gate passes. Use a bounded token budget and stop on held-out
   syntax/compile or repetition failure.
3. **Serious candidate:** the 1.13B architecture is now memory-feasible on one
   80 GB A100 with BF16 and activation checkpointing, but its required token budget
   makes multi-GPU throughput or multiple persistent sessions the practical plan.
4. **Scale decision:** compare loss, FIM, syntax, compilation, memorization, and
   throughput against the 100M baseline. Only then decide whether the 1.13B
   multi-GPU candidate merits compute approval.

Before a longer single-GPU run, use
`configs/training/approved-a100-300m-probe.yaml`. It is an 8,192-token,
single-microbatch, single-accumulation probe for the 304M architecture with
CUDA, BF16, streaming input, and all progress callbacks at the same 8,192-token
boundary. The probe is a wiring and throughput check—not a capability result;
its short budget cannot establish useful TypeScript completion behavior.
`validation_max_tokens` bounds each periodic and final validation pass by actual
input tokens (the A100 probe uses 8,192); it prevents short-file-heavy validation
splits from turning a smoke run into an unbounded evaluation job. Set it explicitly
for longer pilots and record the value with the resolved configuration.
For streaming runs, `streaming_tokens_per_step_estimate` is a conservative estimate
of actual (unmasked) tokens consumed per optimizer step. It is used only to derive
the scheduler horizon and an upper bound on loop steps; attention-mask counts remain
the source of truth for `max_tokens`, metrics, checkpoints, and reports. The current
A100 probe uses 150 tokens/step; the smoke and full configs scale that estimate for
their accumulation factors.

Before a GPU pilot, the run contract must include: resolved model audit, tokenizer
and manifest hashes, global token budget, per-rank batch/sequence geometry, gradient
accumulation, precision and loss-scaling policy, checkpoint sharding format, data
cursor semantics, validation cadence, and an abort threshold for loss, syntax,
compilation, repetition, or memory regression.

## Three-tier data and training architecture

The implementation boundary is deliberately staged:

1. **Tier 0 — correctness (current):** one process, CPU/GPU, materialized
   fixture or approved JSONL, deterministic objective selection, exact
   next-token accounting, and tensor-only atomic checkpoints. This tier exists
   to prove model and resume semantics.
2. **Tier 1 — single-node scale:** one launcher owns a process group; DDP or
   FSDP workers consume immutable token shards through the streaming shard
   contract. Rank/world-size, shard index, record offset, token offset, and
   sampler seed are checkpoint identity. Loss and token counters are reduced
   across ranks before logging or scheduling.
3. **Tier 2 — multi-node production:** an elastic-capable launcher adds node
   membership and restart policy, while the dataset manifest and checkpoint
   remain content-addressed. A failed worker may restart only against the same
   manifest/tokenizer/config hashes; changing world size requires an explicit
   resharding operation and a new run lineage, not silent cursor reuse.

`training.distributed` is a validated planning contract (`single`, `ddp`, or
`fsdp`; backend, rank, world size, timeout, and stable record-hash partition).
The Milestone 1 CLI rejects non-single strategies until process-group setup and
rank-aware checkpointing are implemented. This is intentional: a config that
looks distributed while running single-process would produce false throughput,
loss, and reproducibility claims.

### Token accounting contract

Metrics distinguish input tokens consumed (`batch_tokens`) from supervised
next-token targets (`loss_tokens`). The latter excludes the first conditioning
position, padding, and masked labels, and is the denominator for weighted loss.
Metrics also record `padded_tokens` (the tensor positions sent through the
model), `padding_tokens`, and the derived `token_utilization`. This matters for
the current streaming adapter: it emits one fixed-length window per source
record, so a short TypeScript file can otherwise turn a 4,096-position forward
pass into only a few hundred useful tokens. Throughput and token budgets use
actual tokens; padded positions remain visible as a hardware-efficiency cost.
The future distributed trainer must all-reduce actual, supervised, and padded
counts before reporting.

The default record-oriented adapter deliberately does not concatenate unrelated
records to fill a window. The opt-in `packed_streaming` adapter now packs
records only within one repository, emits `<eos>` boundaries, masks padding
and the first conditioning position, and reports exact shifted objective-token
counts. Its resume cursor identifies the next source position after each
emitted block; it does not yet serialize `<repo>`/`<file>` path markers or
retain a per-record cursor table inside a block. Therefore it is suitable for a
bounded throughput probe, not yet the repository-context objective. The
materialized `data.packing.pack_documents` helper remains repository-scoped
and is not the streaming training path.

### Evaluation boundary

Validation data is a frozen, hash-pinned split and is never drawn from the
training iterator. Online validation may report loss only; syntax, compilation,
FIM, memorization, and security evaluations run as separate jobs against the
same checkpoint hash. No generated sample is a substitute for a held-out
metric, and no external repository code executes in the training process.

## 304M packed-run diagnosis

The completed single-GPU packed diagnostic (1,000,187 tokens) reached 60%
parse rate but 0% compile rate. This is not evidence that the TypeScript helper
is broken: the compiler evaluation intentionally uses a strict, isolated
single-file program, so unresolved imports, decorators, missing declarations,
and incomplete completion fragments can parse while failing compilation.

The more actionable data finding is corpus composition. The approved snapshot
is dominated by TypeScript test records, including compiler/fourslash fixtures
whose `// @Filename:` and `// @BaselineFile:` lines are virtual harness
directives. A small model can learn these high-frequency markers as completion
targets, producing outputs such as `// @Filename: true`. The next packed
training view sets `exclude_compiler_harness: true` to omit only records with
those exact directive forms. It deliberately retains ordinary TypeScript
decorators such as `@Column()`; those are executable source syntax, not
metadata. The source manifest remains unchanged and continues to retain the
excluded records for provenance.

Do not interpret the 0% compile result as a reason to remove decorators or
relax compiler checks. First rerun a bounded probe with the harness view
excluded, then separate evaluation into complete-file reconstruction and
completion fragments with appropriate imports/declarations. A future corpus
revision should also report source-type and repository-family token shares so
test-heavy snapshots cannot silently become the base objective.
