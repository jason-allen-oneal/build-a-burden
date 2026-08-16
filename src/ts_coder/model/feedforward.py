from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
