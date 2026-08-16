from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # Defaults are the local correctness tier. Larger tiers are always explicit
    # YAML configurations; keeping defaults here makes this the single canonical
    # schema used by the model and strict config loader.
    vocab_size: int = 4096
    context_length: int = 512
    layers: int = 4
    hidden_size: int = 256
    attention_heads: int = 4
    kv_heads: int = 2
    ffn_size: int = 768
    name: str = "ts-coder"
    normalization: str = "rmsnorm"
    position_encoding: str = "rope"
    activation: str = "swiglu"
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tied_embeddings: bool = True
    use_sdpa: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        positive = (
            "vocab_size",
            "context_length",
            "layers",
            "hidden_size",
            "attention_heads",
            "kv_heads",
            "ffn_size",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        upper_bounds = {
            "vocab_size": 1_000_000,
            "context_length": 1_000_000,
            "layers": 256,
            "hidden_size": 32_768,
            "attention_heads": 512,
            "kv_heads": 512,
            "ffn_size": 131_072,
        }
        for name, maximum in upper_bounds.items():
            if getattr(self, name) > maximum:
                raise ValueError(f"{name} exceeds safety bound {maximum}")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")
        if (self.hidden_size // self.attention_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (self.normalization, self.position_encoding, self.activation) != (
            "rmsnorm",
            "rope",
            "swiglu",
        ):
            raise ValueError("only rmsnorm/rope/swiglu are supported")
