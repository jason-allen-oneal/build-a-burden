"""Hooks for future sandboxed unit-test pass@k evaluation."""

from __future__ import annotations


def unavailable_reason() -> str:
    return (
        "External repository tests are intentionally disabled in Milestone 1; "
        "use controlled fixtures only."
    )
