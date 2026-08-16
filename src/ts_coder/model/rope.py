from __future__ import annotations

from typing import TypeAlias

import torch
from torch import nn

RotaryValues: TypeAlias = tuple[torch.Tensor, torch.Tensor]


def rotary_frequencies(
    head_dim: int, positions: torch.Tensor, theta: float = 10_000.0
) -> RotaryValues:
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions.float(), inv_freq)
    return angles.cos(), angles.sin()


class RotaryEmbedding(nn.Module):
    """Lazy, shared RoPE table cache.

    The inverse frequencies are kept as a non-persistent buffer so the module
    follows a model across devices without changing the checkpoint format. The
    actual cosine/sine tables are cached by device and output dtype. This keeps
    one table per Transformer instead of rebuilding the same table once per
    layer and avoids retaining a table for devices/dtypes that were never used.
    """

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be positive and even for RoPE")
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple[str, torch.dtype], RotaryValues] = {}

    def _apply(self, fn):
        # ``Module.to``/``Module.cuda`` may change the buffer device or dtype;
        # discard tables built for the old placement before applying the move.
        self._cache.clear()
        return super()._apply(fn)

    @staticmethod
    def _device_key(device: torch.device) -> str:
        # ``str(device)`` includes the CUDA index and is stable for cache keys;
        # it also distinguishes CPU, CUDA, MPS, and other accelerator devices.
        return str(device)

    def _build_cache(
        self,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RotaryValues:
        if sequence_length <= 0:
            empty = torch.empty((0, self.head_dim // 2), device=device, dtype=dtype)
            return empty, empty.clone()
        # Always calculate angles in FP32. This preserves phase accuracy when
        # the model runs in BF16/FP16, then stores the requested output dtype.
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)

    def forward(
        self,
        positions: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> RotaryValues:
        """Return RoPE values for a one-dimensional integer position tensor."""
        if positions.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        if positions.is_floating_point() or positions.is_complex():
            raise ValueError("positions must contain integer indices")
        output_dtype = dtype or torch.get_default_dtype()
        if not output_dtype.is_floating_point:
            raise ValueError("RoPE output dtype must be floating point")
        device = positions.device
        key = (self._device_key(device), output_dtype)
        cached = self._cache.get(key)
        required_length = 0 if positions.numel() == 0 else int(positions.max().item()) + 1
        if cached is None or cached[0].shape[0] < required_length:
            cached = self._build_cache(required_length, device, output_dtype)
            self._cache[key] = cached
        if positions.numel() == 0:
            return cached[0], cached[1]
        index = positions.to(device=device, dtype=torch.long)
        return cached[0].index_select(0, index), cached[1].index_select(0, index)

    @property
    def cache_entries(self) -> int:
        """Number of device/dtype tables currently retained (for diagnostics/tests)."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Release non-checkpointed cosine/sine tables."""
        self._cache.clear()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to [batch, heads, sequence, head_dim]."""
    cos = cos[None, None, :, :].to(dtype=x.dtype)
    sin = sin[None, None, :, :].to(dtype=x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
