from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention, KVCache
from .config import ModelConfig
from .feedforward import SwiGLU
from .rmsnorm import RMSNorm
from .rope import RotaryValues


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, getattr(config, "rms_norm_eps", 1e-6))
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, getattr(config, "rms_norm_eps", 1e-6))
        self.feed_forward = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
        rotary_values: RotaryValues | None = None,
    ):
        attended, new_cache = self.attention(
            self.attention_norm(x), attention_mask, cache, use_cache, rotary_values
        )
        x = x + self.dropout(attended)
        return x + self.dropout(self.feed_forward(self.ffn_norm(x))), new_cache
