"""Deterministic quality measurements and conservative rejection."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QualityMetrics:
    size_bytes: int
    line_count: int
    mean_line_length: float
    max_line_length: int
    comment_ratio: float
    whitespace_ratio: float
    identifier_diversity: float
    repetition_ratio: float


def measure_quality(text: str) -> QualityMetrics:
    lines = text.splitlines() or [""]
    size = len(text.encode("utf-8"))
    chars = max(len(text), 1)
    identifiers = re.findall(r"[A-Za-z_$][\w$]*", text)
    comments = sum(len(x) for x in re.findall(r"//[^\n]*|/\*[\s\S]*?\*/", text))
    return QualityMetrics(
        size,
        len(lines),
        sum(map(len, lines)) / len(lines),
        max(map(len, lines), default=0),
        comments / chars,
        sum(c.isspace() for c in text) / chars,
        len(set(identifiers)) / max(len(identifiers), 1),
        1 - len(set(lines)) / max(len(lines), 1),
    )


def assess_quality(
    text: str, max_bytes: int = 1_000_000
) -> tuple[float, list[str], dict[str, object]]:
    m = measure_quality(text)
    reasons = []
    if "\x00" in text:
        reasons.append("null-byte")
    if any(ord(c) in range(0x202A, 0x202F) for c in text):
        reasons.append("unicode-control")
    if m.size_bytes > max_bytes:
        reasons.append("oversized")
    if m.max_line_length > 5000:
        reasons.append("minified")
    score = max(0.0, 1.0 - 0.25 * len(reasons) - 0.3 * m.repetition_ratio)
    return score, reasons, asdict(m)
