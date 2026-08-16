"""Unified evaluation primitives; outputs JSON-serializable dictionaries."""

from __future__ import annotations

import resource
from typing import Any

from .compile import compile_typescript
from .memorization import exact_file_match
from .security import contains_secret
from .syntax import parse_typescript


def repetition_rate(source: str, width: int = 3) -> float:
    tokens = source.split()
    if len(tokens) < width:
        return 0.0
    windows = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    return 1.0 - len(set(windows)) / len(windows)


def evaluate_sources(sources: list[str], corpus: list[str] | None = None) -> dict[str, Any]:
    parsed = [parse_typescript(item) for item in sources]
    compiled = [compile_typescript(item) for item in sources]
    security_clean = [not contains_secret(item) for item in sources]
    return {
        "syntax_parse_rate": sum(x.get("success", False) for x in parsed) / max(1, len(parsed)),
        "compilation_rate": sum(x.get("success", False) for x in compiled) / max(1, len(compiled)),
        "exact_training_match_rate": (
            sum(exact_file_match(x, corpus or []) for x in sources) / max(1, len(sources))
        ),
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
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "deterministic_generation": None,
    }
