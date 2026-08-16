"""Analytic model-size and memory audits for scale planning.

The audit intentionally uses arithmetic rather than instantiating a model. That
keeps 1B planning safe on the local workstation and makes configuration review
independent of available RAM.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import ModelConfig


def parameter_breakdown(config: ModelConfig) -> dict[str, int]:
    """Return exact trainable parameter components for the dense architecture."""

    vocab, hidden, layers = config.vocab_size, config.hidden_size, config.layers
    head_dim = hidden // config.attention_heads
    kv_width = config.kv_heads * head_dim
    embeddings = vocab * hidden
    attention = layers * (hidden * hidden + hidden * kv_width + hidden * kv_width + hidden * hidden)
    feed_forward = layers * (3 * hidden * config.ffn_size)
    norms = layers * (2 * hidden) + hidden
    output = 0 if config.tied_embeddings else vocab * hidden
    return {
        "token_embeddings": embeddings,
        "attention_projections": attention,
        "feed_forward": feed_forward,
        "normalization": norms,
        "output_projection": output,
        "total": embeddings + attention + feed_forward + norms + output,
    }


def estimate_memory_bytes(
    config: ModelConfig,
    *,
    parameter_bytes: int = 4,
    gradient_bytes: int = 4,
    optimizer_bytes: int = 8,
    activation_multiplier: float = 4.0,
) -> dict[str, int]:
    """Estimate a conservative single-process training memory envelope.

    ``optimizer_bytes`` covers AdamW's two FP32 moment tensors. Activations are
    deliberately a planning multiplier, not a promise; real usage depends on
    batch size, checkpointing, kernels, and precision.
    """

    if min(parameter_bytes, gradient_bytes, optimizer_bytes) <= 0:
        raise ValueError("memory byte widths must be positive")
    if activation_multiplier < 0:
        raise ValueError("activation_multiplier must be non-negative")
    parameters = parameter_breakdown(config)["total"]
    weights = parameters * parameter_bytes
    gradients = parameters * gradient_bytes
    optimizer = parameters * optimizer_bytes
    # This scales with the live hidden states, not the full parameter count.
    activation_elements = (
        config.context_length * config.hidden_size * config.layers * activation_multiplier
    )
    activations = int(activation_elements * parameter_bytes)
    return {
        "weights": weights,
        "gradients": gradients,
        "optimizer_states": optimizer,
        "activation_reserve": activations,
        "total_estimate": weights + gradients + optimizer + activations,
    }


def model_audit(config: ModelConfig) -> dict[str, Any]:
    """Return a serializable architecture and memory audit."""

    breakdown = parameter_breakdown(config)
    memory_fp32 = estimate_memory_bytes(config)
    memory_bf16 = estimate_memory_bytes(config, parameter_bytes=2)
    return {
        "schema_version": 1,
        "config": asdict(config),
        "head_dimension": config.hidden_size // config.attention_heads,
        "query_to_kv_group_ratio": config.attention_heads // config.kv_heads,
        "parameter_breakdown": breakdown,
        "parameter_count": breakdown["total"],
        "memory_estimate_fp32": memory_fp32,
        "memory_estimate_bf16_weights": memory_bf16,
    }
