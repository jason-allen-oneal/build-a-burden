from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_scheduler(
    optimizer: Optimizer, warmup_steps: int, total_steps: int, minimum_ratio: float = 0.1
) -> LambdaLR:
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("require 0 <= warmup_steps < total_steps")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1]")

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0))
        )

    return LambdaLR(optimizer, multiplier)
