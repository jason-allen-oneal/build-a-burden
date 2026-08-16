"""Reproducible character-boundary-safe FIM construction."""

import hashlib
import random
from dataclasses import dataclass

_SPECIAL_TOKENS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<repo>",
    "</repo>",
    "<file>",
    "</file>",
    "<fim_prefix>",
    "<fim_suffix>",
    "<fim_middle>",
    "<diagnostic>",
    "</diagnostic>",
    "<test>",
    "</test>",
    "<patch>",
    "</patch>",
)


@dataclass(frozen=True)
class FIMSample:
    serialized: str
    prefix: str
    suffix: str
    middle: str
    start: int
    end: int


def make_fim(
    text: str, sample_id: str, seed: int = 42, min_span: int = 1, max_span: int = 128
) -> FIMSample:
    if min_span < 1 or max_span < min_span:
        raise ValueError("invalid FIM span lengths")
    if len(text) < min_span:
        raise ValueError("text shorter than minimum span")
    rng = random.Random(  # nosec B311 - reproducible sampling, never a security primitive
        int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:16], 16)
    )
    for _ in range(32):
        span = rng.randint(min_span, min(max_span, len(text)))
        start = rng.randint(0, len(text) - span)
        end = start + span
        middle = text[start:end]
        if not any(token in middle for token in _SPECIAL_TOKENS):
            break
    else:
        raise ValueError("could not choose a FIM span outside special-token text")
    prefix, middle, suffix = text[:start], text[start:end], text[end:]
    return FIMSample(
        f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}",
        prefix,
        suffix,
        middle,
        start,
        end,
    )


def reconstruct(sample: FIMSample) -> str:
    return sample.prefix + sample.middle + sample.suffix
