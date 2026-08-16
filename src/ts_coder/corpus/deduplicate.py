"""Exact, normalized, and conservative near-duplicate grouping."""

from __future__ import annotations

import hashlib
import re


def normalize_typescript(text: str) -> str:
    text = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_fingerprint(text: str, width: int = 5) -> set[str]:
    tokens = re.findall(r"[A-Za-z_$][\w$]*|\S", normalize_typescript(text))
    return {
        sha256_text("\x1f".join(tokens[i : i + width]))[:16]
        for i in range(max(1, len(tokens) - width + 1))
    }


def near_duplicate(a: str, b: str, threshold: float = 0.85) -> bool:
    left, right = token_fingerprint(a), token_fingerprint(b)
    return bool(left or right) and len(left & right) / max(len(left | right), 1) >= threshold


def assign_clusters(records: list[dict], contents: dict[str, str], threshold: float = 0.85) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError("near-duplicate threshold must be between 0 and 1")
    fingerprints = {
        record["record_id"]: (
            token_fingerprint(contents[record["record_id"]])
            if record.get("included", True)
            else set()
        )
        for record in records
    }
    leaders: list[str] = []
    leader_rank: dict[str, int] = {}
    normalized_leaders: dict[str, str] = {}
    fingerprint_index: dict[str, list[str]] = {}
    for record in sorted(records, key=lambda x: x["record_id"]):
        rid = record["record_id"]
        norm = record["normalized_sha256"]
        leader = normalized_leaders.get(norm)
        right = fingerprints[rid]
        if leader is None and record.get("included", True) and right:
            intersections: dict[str, int] = {}
            for shingle in right:
                for candidate in fingerprint_index.get(shingle, ()):
                    intersections[candidate] = intersections.get(candidate, 0) + 1
            for candidate in sorted(intersections, key=leader_rank.__getitem__):
                left = fingerprints[candidate]
                intersection = intersections[candidate]
                union = len(left) + len(right) - intersection
                if intersection / max(union, 1) >= threshold:
                    leader = candidate
                    break
        if leader is None:
            leader = rid
            leader_rank[rid] = len(leaders)
            leaders.append(rid)
            normalized_leaders.setdefault(norm, rid)
            if record.get("included", True):
                for shingle in right:
                    fingerprint_index.setdefault(shingle, []).append(rid)
        record["dedup_cluster"] = leader


def records_by_id(records: list[dict], rid: str) -> dict:
    return next(r for r in records if r["record_id"] == rid)
