# Direct-to-1B development plan

## Decision

The project will not retrain 100M- or 300M-class models. Existing smaller-model
runs remain immutable diagnostic evidence, and bounded smoke programs may still
exercise infrastructure without becoming capability-training runs. The next
capability candidate is the 1,130,465,280-parameter architecture in
`configs/model/serious-1b.yaml`.

This decision removes an intermediate model ladder; it does not waive data,
evaluation, reproducibility, compute, or safety gates.

## Promotion gates

The planning-only 1B model may receive an executable training configuration only
after all of the following are true:

1. **Evaluation quarantine:** task artifacts are hash-pinned, excluded from every
   training view, and checked for normalized text and token-sequence fragment
   contamination.
2. **Corpus scale:** the initial run has at least 3 billion reviewed tokenizer
   tokens. This is a floor for an exploratory 1B run, not a claim of
   compute-optimal training. The corpus program should continue toward 15–25
   billion tokens when licensing, quality, diversity, and compute permit.
3. **Corpus balance:** reports expose token shares by repository family, source
   type, code/tests/docs, TS/TSX/declarations, and duplicate cluster. No single
   family or compiler/test harness may silently dominate the objective.
4. **Stable tokenizer:** the tokenizer is trained only after the first
   billion-token-scale tranche is frozen. Its configuration, corpus manifest, and
   artifact hashes become immutable run lineage.
5. **Scale trainer:** DDP or FSDP is implemented rather than accepted as a
   planning-only configuration. Global counters, deterministic rank sharding,
   sharded checkpoints, and exact resume are verified under worker failure.
6. **Compute contract:** persistent storage and persistent multi-GPU capacity are
   approved. Colab sessions have not had one stable memory profile: the latest
   recorded session exposed about 40 GB, while historical runs recorded 80 GB.
   Every run trusts measured preflight VRAM rather than a notebook label or an
   earlier session.
7. **Run authorization:** the exact manifest, tokenizer, model, token budget,
   hardware topology, checkpoint cadence, and stop conditions have a structured
   authorization record.

## Corpus program

Corpus growth proceeds in immutable, reviewed tranches. A source candidate is not
approved merely because it appears in this plan.

### Tranche 1 — pipeline expansion

- Review and, only after explicit approval, acquire `FuelLabs/fuels-ts` and
  `puppeteer/puppeteer` at newly observed immutable revisions.
- Review narrow path scopes for Angular and Storybook.
- Rebuild reports with evaluation contamination, license, secret, generated-code,
  compiler-harness, duplicate, repository-family, and source-type statistics.
- Use the tranche to validate intake behavior and accounting; do not train a
  smaller capability model on it.

### Tranche 2 — ecosystem diversity

Add independently reviewed sources across server frameworks, browser tooling,
CLIs, build systems, React/TSX applications, data access, validation, testing,
package tooling, and ordinary application code. Avoid filling the corpus with many
near-identical framework repositories or generated declarations.

### Tranche 3 — billion-token freeze

Freeze versioned manifests and token shards once the corpus reaches the initial
3-billion-token floor with acceptable balance. Material source, filter, tokenizer,
or quarantine changes create a new lineage rather than rewriting the freeze.

## Training sequence

1. Arithmetic model audit; no allocation.
2. Single-device allocation and forward/backward smoke at reduced context.
3. Multi-GPU communication, checkpoint, failure, and exact-resume smoke using a
   bounded fixture or tiny approved slice.
4. Short 1B throughput probe with no capability claim.
5. Authorized 1B pretraining in checkpointed token-budget stages.
6. Held-out causal, FIM, syntax, compilation, repository, memorization, security,
   and repetition evaluation against each promoted checkpoint.
7. Supervised completion/FIM/patch tuning only after base-pretraining evidence is
   retained separately.
8. Quantization, inference packaging, and editor/API acceptance testing.

## Stop conditions

Stop and preserve a resumable checkpoint when any of these occur:

- non-finite loss or gradients;
- sustained loss divergence beyond the authorized threshold;
- tokenizer, corpus, source, or quarantine lineage mismatch;
- rank disagreement in counters or checkpoint identity;
- hardware or software drift outside the recorded run contract;
- evaluation contamination;
- unacceptable exact reproduction or secret-like output;
- persistent generation collapse or repetition regression;
- storage durability or checkpoint verification failure.

Parameter count is not a completion criterion. The model becomes usable only when
held-out TypeScript tasks and real editor/API acceptance tests demonstrate it.
