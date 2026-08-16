"""Simple contamination checks using hashes and bounded substring scans."""

from __future__ import annotations

import hashlib


def exact_file_match(generated: str, corpus: list[str]) -> bool:
    target = hashlib.sha256(generated.encode()).digest()
    return any(hashlib.sha256(item.encode()).digest() == target for item in corpus)


def longest_common_substring(a: str, b: str, max_cells: int = 2_000_000) -> int:
    if len(a) * len(b) > max_cells:
        return 0
    previous = [0] * (len(b) + 1)
    longest = 0
    for char_a in a:
        current = [0]
        for index, char_b in enumerate(b, 1):
            value = previous[index - 1] + 1 if char_a == char_b else 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest
