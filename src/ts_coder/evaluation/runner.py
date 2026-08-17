"""Unified evaluation primitives; outputs JSON-serializable dictionaries."""

from __future__ import annotations

import hashlib
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compile import compile_typescript
from .memorization import exact_file_match, longest_common_substring
from .security import contains_secret
from .syntax import parse_typescript


@dataclass(frozen=True)
class _CandidateRank:
    expected_length: int

    def __call__(self, item: str) -> tuple[int, bytes]:
        return (
            abs(len(item) - self.expected_length),
            hashlib.sha256(item.encode()).digest(),
        )


def repetition_rate(source: str, width: int = 3) -> float:
    tokens = source.split()
    if len(tokens) < width:
        return 0.0
    windows = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    return 1.0 - len(set(windows)) / len(windows)


def peak_memory_bytes() -> int:
    """Return process peak RSS in bytes on Linux and macOS."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def evaluate_sources(
    sources: list[str],
    corpus: list[str] | None = None,
    *,
    tool_root: str | Path | None = None,
    compile_timeout_seconds: int = 10,
    compute_longest_matching_span: bool = True,
) -> dict[str, Any]:
    parsed = [
        parse_typescript(item, tool_root=tool_root, timeout=compile_timeout_seconds)
        for item in sources
    ]
    compiled = [
        compile_typescript(item, tool_root=tool_root, timeout=compile_timeout_seconds)
        for item in sources
    ]
    security_clean = [not contains_secret(item) for item in sources]
    reference = corpus or []
    longest = 0
    if compute_longest_matching_span:
        for source in sources:
            candidates = sorted(reference, key=_CandidateRank(len(source)))[:16]
            longest = max(
                longest,
                max(
                    (longest_common_substring(source, item) for item in candidates),
                    default=0,
                ),
            )
    return {
        "syntax_parse_rate": sum(x.get("success", False) for x in parsed) / max(1, len(parsed)),
        "compilation_rate": sum(x.get("success", False) for x in compiled) / max(1, len(compiled)),
        "exact_training_match_rate": (
            sum(exact_file_match(x, reference) for x in sources) / max(1, len(sources))
        ),
        "longest_matching_span": longest,
        "security_clean_rate": sum(security_clean) / max(1, len(security_clean)),
        "security_findings": [
            {"index": index, "category": "secret-pattern"}
            for index, clean in enumerate(security_clean)
            if not clean
        ],
        "diagnostic_count": sum(len(x.get("diagnostics", [])) for x in parsed + compiled),
        "repetition_rate": sum(repetition_rate(x) for x in sources) / max(1, len(sources)),
        "validation_cross_entropy": None,
        "perplexity": None,
        "fim_exact_match": None,
        "fim_token_accuracy": None,
        "tokens_per_second": None,
        "peak_memory_bytes": peak_memory_bytes(),
        "deterministic_generation": None,
    }
