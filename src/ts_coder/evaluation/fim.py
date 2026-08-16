"""FIM reconstruction metrics."""

from __future__ import annotations


def token_accuracy(expected: list[int], actual: list[int]) -> float:
    if not expected:
        return 1.0
    return sum(a == b for a, b in zip(expected, actual, strict=False)) / len(expected)


def exact_match(expected: list[int], actual: list[int]) -> bool:
    return expected == actual
