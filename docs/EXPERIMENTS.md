# Experiments

## 2026-08-15 — `approved-a100-300m-filtered-300k`

- Purpose: bounded A100 comparison after removing compiler/fourslash test-harness
  records from the included corpus. This is a data-quality diagnosis, not a
  capability claim or release candidate.
- Hardware: NVIDIA A100-SXM4-80GB (79.25 GiB visible), 167.05 GiB host RAM,
  235.68 GiB disk; PyTorch 2.11.0+cu128; BF16; sequence length 4,096;
  microbatch 1; gradient accumulation 1; seed 42. Completion timestamp was
  `2026-08-15T05:56:44.474461Z`; wall time was not captured in the run report.
- Model: random initialization, tied 32K v2 tokenizer, 24 layers / 1,024 hidden /
  16 attention heads / 4 KV heads / 2,816 SwiGLU width; exact parameter count
  `304,137,216`.
- Data identity: filtered manifest SHA-256
  `da0f1fb79c466dad8376b23cd738bf4bc4a8d81b5d6b6f346a4a976f343a11d7`;
  documents SHA-256
  `67ad5d1241137e221723165bdd97d664781d798b94fd4e6bc8bae09148021c3b`;
  tokenizer SHA-256
  `e9c9067decab08ef46600291d6b7c79485c842f759e475e0b0d7a897474cde52`;
  config SHA-256
  `58a4d4c64b13dd4758743f388f10c700b9d9d1d3023ccd3781e8385ba6b0860a`.
  The corpus has 15,070 included records and 7,376,774 tokenizer tokens;
  84,357 source records remain rejected in the audit manifest.
- Training result: 301,830 actual tokens / 380 steps; final training loss
  `7.232125`; validation loss `7.116434`. Realized objective totals were
  204,571 causal tokens and 96,879 FIM tokens (document-level selection is not
  token-exact 50/50). Latest checkpoint SHA-256
  `99ab5249c2b395468aaf0b8c718e80d743c73e12ddc5dcc4abbd45b4211e2517`;
  Drive copy: `/content/drive/MyDrive/ts-coder-runs/approved-a100-300m-filtered-300k`.
- Held-out causal completion (five local TypeScript/TSX fixtures): raw greedy
  syntax `0.0`, compilation `0.0`, deterministic `1.0`, repetition `0.5249832776`.
  With repetition penalty 1.1 and no-repeat 3-grams, syntax and compilation
  remained `0.0`, deterministic `1.0`, and repetition fell to `0.0137529138`.
- Held-out FIM: raw exact `0.0`, token accuracy `0.0195238095`, syntax `0.2`,
  compilation `0.0`, deterministic `1.0`, repetition `0.6914345660`. With the
  same decode controls, exact `0.0`, token accuracy `0.0095238095`, syntax
  `0.0`, compilation `0.0`, deterministic `1.0`, repetition `0.0052631579`.
- Evidence hashes on Drive: raw causal
  `616f2fc6e1a36bbb2ba2b40605c8162f538fcb7f73f9df4e39311a11d5fb5db5`;
  raw FIM `580c5437c0fc20a78d57e95ed8583e500ce8b5d574e4b071b6d3a7676abbf998`;
  controlled causal
  `887d478fa5ab8d0a20729154e98722fe05b3313eccff0da8fd059e71bf31fea7`;
  controlled FIM
  `877e8c3cace9f7ac89b912f4b754817240388cbe052fe36cdfe2f75d498fa3f3`.
- Decision: harness filtering improved the data view and decode controls stop
  collapse, but the model still cannot produce valid unseen TypeScript. Do not
  scale this 300k configuration as-is; the next serious run needs substantially
  more approved tokens, source-type balancing, and a curriculum/evaluation split
  that rewards complete compilable units.

## 2026-08-15 — `approved-a100-300m-packed-1m`

- Purpose: first meaningful CUDA training run using the approved seven-source
  corpus, packed repository-local token blocks, the 32K tokenizer, and the
  corrected checkpoint/resume path. This is a capability probe, not a release
  candidate.
- Hardware: NVIDIA A100-SXM4-80GB (79.25 GiB visible), 167.05 GiB host RAM,
  235.68 GiB disk; PyTorch 2.11.0+cu128; BF16; sequence length 4,096;
  microbatch 1; gradient accumulation 1; seed 42. Wall time was 1,690.694 s
  (28m10.694s).
- Model: random initialization, tied 32K tokenizer, 24 layers / 1,024 hidden /
  16 attention heads / 4 KV heads / 2,816 SwiGLU width; exact parameter count
  `304,137,216`. Packed streaming preserved repository boundaries and filled
  context blocks without crossing repositories.
- Data identity: approved manifest SHA-256 (pre-harness-filter historical corpus)
  `2767820fa881643ff85cd7fbf955f9cef6b2b48dd899bb856b3c7094261e02ce`;
  tokenizer SHA-256
  `061572a2de0f006873a6ff953f972c68ffc7b98272587a8d66bc2dc2055b1c1a`;
  training config SHA-256
  `8e765545fc6436b009e9fbbce4da5238f75ad130dba2604e701e2f5762dfa78f`.
  The approved corpus contained 22,172 included records and approximately
  8.85M tokenizer tokens (train 7,542,179 / validation 735,847 / test
  575,543).
- Training result: 1,000,187 actual tokens / 1,290 steps; final training loss
  `4.464441`; validation loss `6.106343`. Realized objective totals were
  545,464 causal tokens and 454,723 FIM tokens (the configured fractions are
  deterministic document selection, not a guaranteed token-exact 50/50 mix).
  Checkpoint `latest` SHA-256:
  `eed0ad5fa8f7fced10883fba402b0712c52d61888d5454f3c6eba9f24c56a1`.
  The persistent Drive copy is
  `/content/drive/MyDrive/ts-coder-runs/approved-a100-300m-packed-1m` and is
  approximately 18 GiB because it retains interval checkpoints.
- Held-out causal completion (five local TypeScript/TSX fixtures, greedy,
  deterministic): syntax parse `0.6`, compilation `0.0`, deterministic `1.0`,
  mean repetition `0.5707070707`. The model produced some parseable fragments
  but no compiling completion.
- Held-out FIM (five local TypeScript/TSX fixtures): exact middle `0.0`, token
  accuracy `0.0`, syntax parse `0.2`, compilation `0.0`, deterministic `1.0`,
  mean repetition `0.6351163074`. The pinned TypeScript helper was rebuilt with
  `npm ci --ignore-scripts`; failures are model failures, not helper setup.
- Evidence hashes: completion evaluation
  `f897b9f8149aa1c6fb700a8de430893b4bafa3d84f8c995d0a4cd30e3b07e250`;
  FIM evaluation
  `6696c93365233e8d3003123bbd2e8b1b0941f98bfee3b5a78a5a66f41c81ae78`.
- Decision: packed streaming and the 304M BF16 training path are operational,
  and syntax moved above zero, but compilation and FIM quality remain zero.
  Do not scale this exact objective/curriculum unchanged. The next work should
  address generation stability and objective/data curriculum, then repeat a
  measured comparison before increasing parameter count or token budget.

## 2026-08-15 — `approved-a100-300m-probe`

- Purpose: verify the real CUDA training, checkpoint, persistent-artifact, and
  evaluation path on the supplied Colab runtime before spending a larger
  budget. This is a wiring/throughput probe, not a capability run.
- Hardware: NVIDIA A100-SXM4-80GB (79.25 GiB visible), 167.05 GiB host RAM,
  235.68 GiB disk; PyTorch 2.11.0+cu128; CUDA/BF16; sequence length 4,096;
  microbatch 1; gradient accumulation 1; seed 42.
- Model: random initialization, tied 32K tokenizer, 24 layers / 1,024 hidden /
  16 attention heads / 4 KV heads / 2,816 SwiGLU width; exact parameter count
  `304,137,216`.
- Probe budget and result: requested 8,192 input tokens; 48 optimizer steps;
  8,225 actual input tokens; runtime 113.831 seconds; final training loss
  `5.281111`; bounded validation loss `7.917333`. Realized objective counts
  were 3,806 causal tokens and 4,419 FIM tokens. No OOM or checkpoint-write
  failure occurred. The checkpoint was copied to the persistent Drive run
  directory at `/content/drive/MyDrive/ts-coder-runs/approved-a100-300m-probe-v2`.
- Held-out causal completion (five local TypeScript/TSX fixtures, greedy,
  deterministic): syntax parse `0.0`, compilation `0.0`, deterministic `1.0`,
  mean repetition `0.9626086957`. The model emitted repeated ` = ` fragments.
- Held-out FIM (five local TypeScript/TSX fixtures): exact middle `0.0`, token
  accuracy `0.0`, syntax parse `0.0`, compilation `0.0`, deterministic `1.0`,
  mean repetition `0.9247334643`. The pinned TypeScript helper was rebuilt in
  Colab from the lockfile with `npm ci --ignore-scripts`; the zero syntax and
  compile rates are therefore genuine model failures, not helper-missing
  errors.
- Evidence hashes: completion evaluation
  `b3ed8eef966dd4eb1a5db7dd04aea05a03314bde72caa2246abd0e40af7c368a`;
  FIM evaluation `51536e2205e8e37717a188644c58549ae59a10156a1b2cd41f2f54cdf1f7bd1f`.
- Decision: CUDA wiring, BF16 memory envelope, streaming data cursor,
  checkpoint persistence, and machine-readable evaluation pass. The model is
  not usable. Do not scale this objective/data presentation unchanged. The
  next run needs packed token blocks, a substantially larger/cleaner corpus,
  generation-stability controls, and a curriculum that separates objective
  loss from held-out completion quality.

## 2026-08-14 — `approved-general-1m` stopped at 500k gate

- Intended budget: 1,000,000 tokens; CPU FP32; 5,245,184 parameters; six approved
  repository families; deterministic 50/50 causal/FIM target
- Last fully evaluated checkpoint: 500,118 tokens; SHA-256
  `d748d67ed7bce4f847339ceccfcb16503ef9e10ce7ec3bcb91a0a38c8b7b1bac`
- Graceful interrupted checkpoint: 578,042 tokens; SHA-256
  `faec4a923e1be4ccd58c5149ac3df6ea814a216c7ace35b13ac174e6393cdf52`
- Validation loss at 100k/200k/300k/400k/500k: 7.0552 / 6.5524 / 6.2426 /
  6.2978 / 6.0180
- Held-out syntax and compilation rates: 0.0 at all five gates; deterministic rate 1.0
- Mean repetition at the five gates: 70.1% / 66.9% / 12.2% / 59.6% / 72.0%
- Realized objective mixture over the prepared stream: 15,035 causal and 15,046 FIM
  examples; 1,755,393 causal and 1,762,211 FIM tokens (50.10% FIM)
- Decision: stop early under the declared generation-collapse condition. Lower
  validation loss did not produce valid unseen completions, and repetition regressed
  sharply after 300k. Do not resume this configuration unchanged.

## 2026-08-14 — `completion-overfit-aligned-50k`

- Purpose: prove that training and generation can learn an ordinary TypeScript
  completion after correcting the corpus-artifact and next-token alignment defects
- Dataset: controlled fixture training split; causal-only curriculum
- Model: random initialization, 5,245,184 parameters, CPU FP32
- Training: 50,030 tokens / 924 steps; final loss 0.01613; validation loss 9.23804
- Checkpoint SHA-256: `cc897c905c0c5dd014c0c9bc6d8aadb26127a3bd751db76a2a83d974be2812e7`
- Exact-prefix completion:

  ```typescript
  export function greeting(name: string): string {
    return `Hello, ${name}!`;
  }
  ```

- TypeScript parser: success, zero diagnostics; compiler: success, zero diagnostics
- Nearby unseen prompts remain incoherent. This demonstrates memorization/correct
  objective wiring, not general coding capability.
- Corrections made: training now uses its configured corpus artifact, respects a
  token budget over a repeated stream, covers later document windows, rejects empty
  generation prompts, and applies the causal shift exactly once.

## 2026-08-14 — `approved-dev` expansion smoke run

**Invalidated for capability measurement.** The run exposed two defects: the CLI
silently read the fixture document artifact despite recording the approved manifest,
and the data pipeline pre-shifted labels that the model shifted a second time. Retain
the record as failure evidence; do not compare its losses or inference to corrected
runs.

- Sources: the original Zod/Vite/NestJS tranche plus typescript-eslint, TypeORM,
  and Redux Toolkit at exact revisions in `manifests/approved-sources.jsonl`
- Corpus: 18,059 audited records / 6,780 included / 11,279 rejected
- Manifest SHA-256: `4e3b4925adebb4d66b9591f1dac7e4f31b85f9cb097443e89af26bc32c25a21f`
- Documents SHA-256: `f71a72f1929e57106072d6e68af154de12891a494143abaf5603b33b28e78a9a`
- Splits: 5,379 train / 988 validation / 413 test; four/one/one complete
  repository families respectively; zero repository, dedup-cluster, or normalized
  content overlap
- Safety: 472 detector-hit records rejected without logging matched values
- Tokenizer: byte-level BPE, 8,192 vocabulary; SHA-256
  `500b89b47752764a7610412dd274dce086f6d84d597f717f0714ca814f086404`;
  round-trip 1.0, 4.28526 bytes/token, 1.07002 tokens/lexical token, TSX and
  template-literal coverage, no special-token collision
- Model: random initialization, 5,245,184 parameters, 8 optimization steps,
  CPU FP32, 50/50 causal/FIM objective
- Checkpoint SHA-256: `bd7b3e80f39a4a4fb5fef44ea7adb72d56fea61672398cc06b835a5508d0c39a`
- Training: final loss 8.19160; validation loss 8.27116
- Evaluation: cross-entropy 8.81036; perplexity 6,703.36; syntax and compilation
  rates 0.0; FIM exact/token accuracy 0.0/0.0; deterministic generation true;
  27.36 tokens/s; peak memory 428,781,568 bytes
- Completion result: greedy causal and FIM prompts both collapsed to repeated
  ` =` tokens. This is a failed capability result, not a useful coding model.
- Defect found and fixed: empty prompts now fail with a clear `ValueError` instead
  of reaching an ambiguous zero-length KV-cache reshape.
- Decision: intake, training, checkpoint, and inference plumbing work on the expanded
  corpus. Do not interpret eight optimization steps as evidence of capability or
  begin paid/scaled training.

## 2026-08-14 — `approved-pilot-corpus-v1`

- Sources: Zod, Vite, and NestJS at the exact revisions recorded in
  `manifests/approved-sources.jsonl`
- Acquisition: isolated bare Git fetch, verified commit, bounded Git-object
  materialization, atomic snapshot receipts; no repository code executed
- Corpus: 5,692 audited records / 2,340 included / 3,352 rejected
- Manifest SHA-256: `d9a44720138bf02d335c3e4fab0aeb2693a3485b794800f4a91d15a9868b9a08`
- Documents SHA-256: `0859a5622538856c33f96f93d61fd38b93d0d2b9766b1ffea561bda4bb6551ff`
- Snapshot tree SHA-256: Zod `fff9ceb88e522123228a4220c9454aa741dd143d21f84c6cdffe2ce11302c4a9`;
  Vite `c016282b66d1954a6bfd54a4d5c310774293f5b90f983079f81b6665d9eecb03`;
  NestJS `30deb0854b01de2909d5fbab41478d15c9b9cd7aba25cf1033270506e1e177a5`
- Included split: 1,605 train (NestJS) / 413 validation (Zod) / 322 test (Vite);
  one complete repository family per split
- Duplicate checks: 4,832 clusters; zero cluster or normalized-hash overlap across
  splits; zero repository overlap across splits
- Safety: 180 detector-hit records rejected; seven symlinks recorded and rejected;
  no detected secret values printed
- Tokenizer: byte-level BPE, requested/actual vocabulary 8,192; tokenizer SHA-256
  `9b5d00cbe890bf675f20a18568c6540335d7acbdade1c5ca854d9eeb56385e87`
- Tokenizer metrics: round-trip 1.0; bytes/token 3.86845; lexical-token ratio
  1.02899; vocabulary utilization 0.95349; TSX/template-literal checks present;
  no special-token collision
- Reproducibility: a second tokenizer training run produced the same SHA-256
- Decision: use this dataset to validate approved-source intake and tokenizer
  behavior only. Do not call it a production corpus, train the 32K production
  tokenizer, start paid compute, or publish model weights.

## 2026-08-14 — `approved-causal-300k` objective A/B

- Purpose: compare a causal-only objective with the matched 50/50 causal/FIM
  run before spending more compute. Both use the same random-initialized
  5,245,184-parameter model, seed 42, CPU FP32 training, sequence length 128,
  approved six-repository corpus, manifest, and 8,192-token tokenizer.
- Data identity: manifest SHA-256
  `4e3b4925adebb4d66b9591f1dac7e4f31b85f9cb097443e89af26bc32c25a21f`;
  documents SHA-256
  `f71a72f1929e57106072d6e68af154de12891a494143abaf5603b33b28e78a9a`;
  tokenizer SHA-256
  `500b89b47752764a7610412dd274dce086f6d84d597f717f0714ca814f086404`.
- Causal configuration SHA-256:
  `fd3fab1548bab92affff80aa65f16cc330b55ef61b429202c125529813091b7e`.
  The matched mixed configuration SHA-256 is
  `bb57fa9fb70f3049af619ba83b32a2a9b49a97091049e9bd818a8f6321e9772d` and
  differs only in run/objective settings.
- Causal run: 300,075 processed tokens / 2,632 steps; final training loss
  `6.03332`; validation loss `6.86545 / 6.36278 / 6.07775` at 100k/200k/300k;
  29,977 causal examples and 3,502,095 objective tokens. Final checkpoint
  (`checkpoints/latest`) SHA-256:
  `58e5d8e70d5ab2766a10557e7b4b276b725a53c2969ef1983a17f8e0cc8ad4ef`.
- Matched mixed comparator: checkpoint at 300,072 tokens; validation loss
  `6.24256`; realized prepared-stream mixture was 50.10% FIM by token count.
- Held-out inference: five local TypeScript/TSX fixtures, greedy decoding,
  deterministic repeat. Neither checkpoint parsed or compiled any fixture
  (syntax `0.0`, compilation `0.0`, deterministic `1.0`). Mean repetition was
  `0.12177` for mixed and `0.79803` for causal-only. The causal-only model
  mostly repeated `}` / `export classColumn()` patterns; this is generation
  collapse, not coding ability. Evaluation artifact SHA-256 values are mixed
  `6109762583530edd2958562caf1f4e5a6f9d3f6a11a817fd39a60d571aa81538` and
  causal `e94f0a67d1795ee29fcf51de6dded45ae4d5be0168dce2e9cba12135ea03942`;
  the evaluator explicitly makes no independent benchmark claim.
- Decision: causal-only reduced validation loss relative to the matched mixed
  checkpoint but materially worsened held-out generation and did not clear the
  syntax/compile gate. Do not scale either objective unchanged. Preserve both
  runs as failure evidence and improve the evaluation/objective curriculum
  before another larger experiment.

### Objective-aware FIM follow-up

- Added `fim-evaluate` and a separate strict fixture schema so FIM-trained
  behavior is evaluated as `<fim_prefix> + prefix + <fim_suffix> + suffix +
  <fim_middle>`, rather than being judged only with a causal prompt. Fixture
  SHA-256: `36fe6513dc119940f39999fbf4baa7be66bd90a960333cb13ba87699d17d5fda`.
- The evaluator removes generated control markers at the stopping boundary,
  but does not repair or rewrite generated TypeScript. It reports exact middle
  match, token accuracy, syntax, compilation, determinism, and repetition.
- The pinned TypeScript helper was adjusted to preserve TSX syntax with a minimal
  intrinsic-element declaration; all five expected FIM fixture middles now parse
  and compile without a React install or runtime.
- Mixed checkpoint: FIM exact `0.0`, token accuracy `0.0`, syntax `0.0`,
  compilation `0.0`, deterministic `1.0`, repetition `0.07556`.
- Causal-only checkpoint: FIM exact `0.0`, token accuracy `0.00909`, syntax
  `0.0`, compilation `0.0`, deterministic `1.0`, repetition `0.78182`.
- Evaluation artifact SHA-256: mixed
  `8e88fecdbf968469970a9945f04d63df6141097e8e76ea97f75dbf972b14a585`;
  causal `d9d1dd1bd930ffa36fc3f1c5f76f355e17ef5b9db5a5c08c5c97feea9ec732ef`.
- Decision: the failure is objective/model capability, not merely the causal
  evaluator treating FIM markers as source. Keep the new FIM metrics as a
  required gate; do not claim FIM competence or start a larger run yet.

## 2026-08-14 — `dev` local vertical slice

- Run ID: `dev`
- Dataset: controlled fixture corpus, 17 manifest records / 10 included documents
- Manifest SHA-256: `1a6c05a8c6e56060c4fa4b6c807ba1f5ea80711faf6acaf7ef696be18fc13542`
- Tokenizer: byte-level BPE, requested 4096 / actual 491 vocabulary entries
- Tokenizer SHA-256: `49916f9c7b8b8054ba1594b07b6c57d30c09480e7641d01687dbf89d5b59be77`
- Model: 4,196,608 parameters; 8 optimization steps; 557 tokens processed
- Hardware: CPU-only PyTorch 2.7.1, Python 3.12.12, approximately 62 GiB RAM
- Results: final loss 7.35733175; validation loss 7.49015188; fixed-seed generation
- Evaluation: validation cross-entropy 7.42565298; perplexity 1678.49524;
  FIM exact/token accuracy 0.0/0.0; syntax parse 0.0; compilation 0.0;
  exact training match 0.0; deterministic generation true; generation throughput
  approximately 236–366 tokens/s across repeated CPU runs
- Decision: correctness plumbing passes; do not scale until approved corpus and a
  useful held-out evaluation are available. Invalid generated output is expected
  from this deliberately tiny random-initialized development run.

For each completed run record: run ID, resolved configuration, dataset/manifest hash,
tokenizer hash, tokens processed, exact parameter count, hardware, duration, metrics,
problems, and resulting decision.
