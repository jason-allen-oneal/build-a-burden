"""Loss and perplexity metrics."""

from __future__ import annotations

import math


def perplexity(loss: float) -> float:
    return math.exp(min(float(loss), 80.0))
