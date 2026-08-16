from __future__ import annotations

import math
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .rope import RotaryValues, apply_rope, rotary_frequencies

KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]

# ``enable_gqa`` was added after the original SDPA API. Checking the runtime
# docstring avoids passing an unknown keyword to older supported PyTorch builds.
# Backend-specific failures still use the materialized SDPA fallback below.
_SDPA_GQA_AVAILABLE = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")


def sdpa_gqa_available() -> bool:
    """Whether this PyTorch runtime exposes native SDPA grouped-query attention."""
    return _SDPA_GQA_AVAILABLE


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.attention_heads
        self.num_kv_heads = config.kv_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.groups = self.num_heads // self.num_kv_heads
        self.theta = config.rope_theta
        self.use_sdpa = config.use_sdpa
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
        rotary_values: RotaryValues | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch, query_len, _ = x.shape
        past_len = 0 if cache is None else cache[0].shape[2]
        positions = torch.arange(past_len, past_len + query_len, device=x.device)
        if rotary_values is None:
            cos, sin = rotary_frequencies(self.head_dim, positions, self.theta)
        else:
            cos, sin = rotary_values
            if cos.shape[0] != query_len or sin.shape[0] != query_len:
                raise ValueError("rotary_values must match the query sequence length")
        q = self.q_proj(x).view(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        new_cache = (k, v) if use_cache else None
        key_len = k.shape[2]
        query_positions = torch.arange(past_len, past_len + query_len, device=x.device)[:, None]
        key_positions = torch.arange(key_len, device=x.device)[None, :]
        causal_allowed = key_positions <= query_positions
        allowed: torch.Tensor | None = causal_allowed
        if attention_mask is not None:
            if attention_mask.shape != (batch, key_len):
                raise ValueError(f"attention_mask must have shape {(batch, key_len)}")
            allowed = causal_allowed[None, None, :, :] & attention_mask[:, None, None, :].bool()
        elif self.use_sdpa and cache is None:
            allowed = None
        else:
            allowed = causal_allowed[None, None, :, :]
        if self.use_sdpa and _SDPA_GQA_AVAILABLE:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=allowed is None,
                enable_gqa=True,
            )
        else:
            # Older PyTorch builds, or builds whose selected SDPA backend does
            # not implement native GQA, take this path. SDPA still avoids the
            # explicit score/softmax tensors when available; only the K/V head
            # expansion is materialized for API compatibility.
            k_attn = k.repeat_interleave(self.groups, dim=1)
            v_attn = v.repeat_interleave(self.groups, dim=1)
            if self.use_sdpa:
                output = F.scaled_dot_product_attention(
                    q,
                    k_attn,
                    v_attn,
                    attn_mask=allowed,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=allowed is None,
                )
            else:
                if allowed is None:
                    raise RuntimeError("explicit attention requires an attention mask")
                scores = torch.matmul(q, k_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
                scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
                weights = torch.dropout(weights, self.dropout, self.training)
                output = torch.matmul(weights, v_attn)
        output = output.transpose(1, 2).contiguous().view(batch, query_len, -1)
        return self.o_proj(output), new_cache
