"""Deterministic training-objective selection."""

from __future__ import annotations

import hashlib


def use_fim(sample_id: str, seed: int, fim_fraction: float) -> bool:
    """Select FIM with a stable per-sample uniform variate.

    This is independent of input ordering, worker count, and Python's randomized
    hash seed, so resuming or rebuilding a stream preserves objective choices.
    """
    if not 0.0 <= fim_fraction <= 1.0:
        raise ValueError("fim_fraction must be in [0, 1]")
    if fim_fraction == 0.0:
        return False
    if fim_fraction == 1.0:
        return True
    digest = hashlib.sha256(f"objective:{seed}:{sample_id}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return draw < fim_fraction
