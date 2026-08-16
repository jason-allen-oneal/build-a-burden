"""Repository-context evaluation interface."""

from __future__ import annotations


def evaluate_repository_context(*_args, **_kwargs) -> dict[str, object]:
    return {
        "status": "not_implemented",
        "reason": "reserved for Milestone 4 repository-level evaluation",
    }
