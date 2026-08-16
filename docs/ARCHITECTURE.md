# Architecture

The dense decoder consists of token embeddings, pre-normalization Transformer
blocks, tied output embeddings, RMSNorm, RoPE, causal grouped-query attention, and
SwiGLU feed-forward layers. For batch `B`, sequence `T`, hidden width `D`, query
heads `H`, and KV heads `K`, hidden states are `[B,T,D]`; queries are
`[B,H,T,D/H]`, while the cached keys/values remain `[B,K,T,D/H]`. Native SDPA
GQA consumes those smaller KV tensors directly; the explicit fallback expands
them across `H/K` query groups. RoPE rotates paired query/key channels without
learned absolute positions. Padding and causal masks prevent padding attention
and future leakage.

Each block applies RMSNorm → attention → residual, then RMSNorm → SwiGLU → residual.
The LM head reuses the embedding matrix. Initialization is seeded and scaled by
module role. `Transformer` owns one lazy RoPE cache, keyed by device and dtype,
and shares the selected position rows with every block in a forward pass; the
cache is non-persistent and is not part of checkpoints. Inference caches per-layer
rotated keys and values; one-token cached logits must match full-prefix logits
within tolerance.

Parameter count is computed from instantiated trainable tensors, not estimated. A
useful audit formula is embedding `V*D`, attention projections
`D*D + 2*D*(K*D/H) + D*D`, SwiGLU `3*D*F`, norms `2*D` per block, plus final norm;
bias terms follow implementation configuration. Exact counts belong in run reports.

The analytic audit is available without allocating the model:

```bash
uv run python -m ts_coder.cli model-audit \
  --config configs/model/smoke-25m.yaml \
  --output artifacts/model-audits/smoke-25m.json
```

## Architecture tiers and hardware contract

The project deliberately separates a local correctness tier from useful training:

| Tier | Model | Purpose | Hardware decision |
|---|---:|---|---|
| Local correctness | 5M–25M | tests, overfit, checkpoint/resume, tokenizer/eval | CPU workstation |
| Useful pilot | 100M–300M | scaling evidence and held-out compile/test behavior | approved GPU host |
| Serious target | ~1.13B | production-shaped base model candidate | 80 GB A100 with checkpointing; multi-GPU for throughput |

The local development workstation has an 8-thread i7-1165G7, 62 GiB RAM, no CUDA
device, and roughly 96 GiB free disk. It can host the 25M smoke model and audit
larger configs, but it is not a credible host for the 100M/1B token budgets. A
parameter count that fits in RAM is not a training plan: AdamW moments, gradients,
activations, data, checkpoints, and wall-clock cost dominate.

The supplied Colab tab was subsequently verified as an
`NVIDIA A100-SXM4-80GB` with 79.25 GiB visible VRAM. High-RAM is enabled,
providing approximately 167 GiB host RAM; it does not increase GPU memory. The
304M CUDA/BF16 probe completed without OOM, but its held-out completion and FIM
gates were both zero, so memory capacity is not evidence of model usefulness.

The serious planning candidate is 24 layers, hidden size 2048, 16 query heads,
4 KV heads, FFN size 5504, vocabulary 32768, context 8192, tied embeddings:
`1,130,465,280` parameters by the analytic formula. It is planning-only until
GPU memory, corpus size, token budget, and checkpoint throughput are approved. An
80 GB A100 is sufficient for the model's memory envelope with BF16,
`gradient_checkpointing`, microbatch 1, and accumulation; multi-GPU is still
preferable for a serious token budget. It must not be instantiated or trained on
the local CPU host.

The first architecture worth taking to a GPU is the 300M pilot in
`configs/model/pilot-300m.yaml`: 24 layers, hidden size 1024, 16 query heads,
4 KV heads, FFN size 2816, vocabulary 32768, context 4096, and tied embeddings.
Its exact analytic count is `304,137,216` parameters. The audit estimates about
5.1 GB for BF16 weights, gradients, AdamW moments, and a deliberately incomplete
activation reserve; in practice it needs a larger safety margin for kernels,
temporary buffers, validation, and checkpointing. Treat 16 GB as an aggressive
floor and 24 GB as the sane single-GPU starting point. This file is planning-only
until the corpus and GPU run contract are approved.

The current filtered approved corpus is 7,376,774 tokenizer tokens. That is enough to
exercise the pipeline, not enough to train either pilot credibly. A 300M run needs
at least hundreds of millions of *distinct* approved tokens for a diagnostic
scaling experiment and preferably several billion for a useful code model. The
1.13B candidate should not be scheduled until the corpus plan, deduplication
statistics, and token budget support it; parameter count alone is not evidence of
capability.

## Scale-up work required before the serious target

1. Make the model schema single-source (the runtime `ModelConfig` is canonical) and
   audit every tier analytically before allocation.
2. The model now uses a shared device/dtype-aware RoPE cache.
3. The model now avoids materializing repeated GQA keys/values on PyTorch SDPA
   runtimes exposing `enable_gqa`; older runtimes use a materialized SDPA fallback,
   while `use_sdpa: false` remains the explicit attention correctness path.
4. Add residual-output depth scaling for the serious initialization after a small
   ablation; do not silently change initialization between runs.
5. Add sharded token streams, distributed data cursors, and checkpoint manifests
   before multi-GPU training. Local CPU training remains the reference implementation.

The current model exposes `use_sdpa` as an explicit architecture switch. It defaults
to the PyTorch scaled-dot-product attention path. When the runtime exposes native
`enable_gqa`, the path keeps KV heads compact; otherwise it materializes repeated
KV heads before calling SDPA. `use_sdpa: false` selects the explicit score/softmax
implementation. This switch is part of resolved configuration and must remain fixed
for a comparable experiment. Native SDPA reduces transient KV expansion and can
select fused kernels, but backend behavior and numerical tolerances remain runtime-
dependent; the explicit path remains the reference correctness fallback.

Long-context GPU configurations also set `gradient_checkpointing: true`. This
recomputes Transformer blocks during the backward pass to trade compute for a much
smaller activation footprint. It is disabled for the local correctness defaults,
and KV-cache/inference paths never checkpoint.

Current configuration checks: development `4,196,608`, 25M smoke `25,172,352`,
and 100M pilot `100,682,496` trainable parameters. These are construction checks
only; no smoke or pilot training run was started.
