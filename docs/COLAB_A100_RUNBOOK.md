# Colab A100 run contract

An 80 GB A100 changes the hardware constraint, not the data-policy or
reproducibility requirements. This runbook is for an approved, single-GPU pilot;
it does not authorize a paid run or a release of weights.

## What fits

| Candidate | Exact parameters | Context | A100 posture |
|---|---:|---:|---|
| 300M pilot | 304,137,216 | 4,096 | Comfortable on the verified 80 GB A100 with BF16 and checkpointing |
| Serious candidate | 1,130,465,280 | 8,192 | 80 GB target; 40 GB requires a reduced context or further memory engineering |

The supplied notebook was verified at preflight as
`NVIDIA A100-SXM4-80GB`, 79.25 GiB visible VRAM, PyTorch 2.11.0+cu128. High-RAM
is enabled and the Colab header reports 167.05 GiB host RAM; that is system
memory, not GPU memory. The completed 304M probe used this runtime and did not
OOM, but still failed the held-out capability gates, so hardware headroom is not
evidence of a usable model.

The analytic estimate is not a promise of peak memory. Leave room for logits,
temporary kernels, the CUDA allocator, validation, and checkpoint writes. Keep
`gradient_checkpointing: true`, `use_sdpa: true`, and start with microbatch 1.
Increase gradient accumulation to reach the target global batch without raising
the per-step memory footprint.

## Gates before starting a session

1. Confirm the runtime is actually an A100 80 GB and record the driver, CUDA,
   PyTorch, and GPU memory details in the run directory.
2. Mount persistent storage before creating a run directory. Colab storage is
   ephemeral; checkpoints and resolved configs must live on Drive or another
   approved persistent volume.
3. Verify the approved tokenizer and corpus-manifest hashes. Never rebuild a
   tokenizer inside the training notebook without recording a new lineage.
4. Stop if the corpus is still only approximately 7.38M filtered tokens. That corpus validates the
   pipeline but cannot support a useful 300M or 1.13B run. Expand approved data,
   rebuild shards, and review license, secret, duplicate, and split reports first.
5. Run `model-audit` and save its JSON beside the run configuration. A planning
   config must be converted to an explicit, reviewed executable config before
   training.

## Session discipline

- Use BF16 autocast, activation checkpointing, and a fixed seed.
- Keep sequence length at 4,096 for the 300M pilot. Do not jump to 8,192 until
  memory and held-out behavior are measured.
- Checkpoint to persistent storage every bounded token interval, and test resume
  from a copied checkpoint before spending a long session.
- Treat a Colab disconnect as expected failure. Resume only when tokenizer,
  manifest, model, optimizer, scheduler, and data-cursor identities match.
- Record actual tokens/sec, peak memory, validation loss, syntax/compile/FIM
  metrics, repetition, and checkpoint hashes. Do not infer usefulness from loss
  alone.

## Recommended order

1. Expand and freeze the approved corpus.
2. Run a short 300M A100 smoke (enough to validate throughput, memory, resume,
   and evaluation, not enough to claim capability).
3. Run the 300M pilot against held-out TypeScript tasks.
4. Only if syntax, compilation, FIM, and memorization gates improve, promote the
   1.13B configuration for a separately approved run.
