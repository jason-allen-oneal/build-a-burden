"""Integrity and corpus-contamination checks for held-out evaluation fixtures."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ts_coder.corpus.deduplicate import normalize_typescript


@dataclass(frozen=True)
class QuarantineFixture:
    fixture_id: str
    kind: str
    path: Path
    sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_contained_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("quarantine fixture path must be a non-empty string")
    candidate = root / value
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"quarantine fixture must be a regular file: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"quarantine fixture escapes project root: {candidate}") from exc
    return resolved


def load_quarantine_manifest(
    manifest_path: str | Path, *, project_root: str | Path
) -> tuple[dict[str, Any], list[QuarantineFixture]]:
    """Load a strict manifest and verify every fixture's byte-level identity."""
    manifest = Path(manifest_path)
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"quarantine manifest must be a regular file: {manifest}")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "scope", "fixtures"}:
        raise ValueError("invalid quarantine manifest fields")
    if raw["schema_version"] != 1 or raw["scope"] != "held-out-evaluation":
        raise ValueError("unsupported quarantine manifest identity")
    entries = raw["fixtures"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("quarantine manifest must contain fixtures")
    root = Path(project_root).resolve()
    fixtures: list[QuarantineFixture] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"id", "kind", "path", "sha256"}:
            raise ValueError("invalid quarantine fixture fields")
        fixture_id = entry["id"]
        kind = entry["kind"]
        digest = entry["sha256"]
        if not isinstance(fixture_id, str) or not fixture_id or fixture_id in seen:
            raise ValueError("quarantine fixture ids must be unique non-empty strings")
        if kind not in {"completion", "fim"}:
            raise ValueError(f"unsupported quarantine fixture kind: {kind!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("quarantine fixture sha256 must be a 64-character digest")
        path = _regular_contained_path(root, entry["path"])
        actual = _sha256_bytes(path.read_bytes())
        if actual != digest:
            raise ValueError(f"quarantine fixture hash mismatch: {fixture_id}")
        _load_task_fragments(path, kind, fixture_id)
        fixtures.append(QuarantineFixture(fixture_id, kind, path, digest))
        seen.add(fixture_id)
    return raw, fixtures


def _typescript_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z_$][\w$]*|\S", normalize_typescript(value)))


def _load_task_fragments(path: Path, kind: str, fixture_id: str) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"invalid evaluation fixture root: {path}")
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"evaluation fixture contains no tasks: {path}")
    fragments: dict[str, str] = {}
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str):
            raise ValueError(f"invalid evaluation task: {path}")
        task_id = task["id"]
        if not task_id or task_id in seen:
            raise ValueError(f"evaluation task ids must be unique: {path}")
        values: dict[str, object]
        if kind == "completion":
            values = {"prompt": task.get("prompt")}
        else:
            parts = (task.get("prefix"), task.get("middle"), task.get("suffix"))
            values = {
                "full": "".join(parts) if all(isinstance(part, str) for part in parts) else None,
                "prefix": task.get("prefix"),
                "middle": task.get("middle"),
            }
        for fragment_kind, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evaluation task has no quarantinable content: {task_id}")
            normalized = normalize_typescript(value)
            tokens = _typescript_tokens(normalized)
            if not tokens:
                raise ValueError(f"evaluation task normalizes to empty content: {task_id}")
            # Tiny punctuation fragments would match almost every TypeScript file.
            if fragment_kind != "full" and len(tokens) < 5:
                continue
            key = f"{fixture_id}:{task_id}:{fragment_kind}"
            if key in fragments:
                raise ValueError(f"duplicate quarantine fragment identity: {key}")
            fragments[key] = normalized
        seen.add(task_id)
    return fragments


def verify_quarantine(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
    corpus: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Verify fixture hashes and reject exact normalized corpus contamination.

    ``corpus`` maps stable corpus record IDs to their original source text. Both
    equality and containment are checked after TypeScript normalization because
    completion fixtures are intentionally prefixes rather than whole files.
    """
    _, fixtures = load_quarantine_manifest(manifest_path, project_root=project_root)
    fragments: dict[str, str] = {}
    for fixture in fixtures:
        fixture_fragments = _load_task_fragments(fixture.path, fixture.kind, fixture.fixture_id)
        overlap = fragments.keys() & fixture_fragments.keys()
        if overlap:
            raise ValueError(f"duplicate quarantine fragment identity: {sorted(overlap)[0]}")
        fragments.update(fixture_fragments)
    matches: list[dict[str, str]] = []
    records = corpus.items() if isinstance(corpus, Mapping) else corpus or ()
    records_checked = 0
    fragment_tokens = {task_id: _typescript_tokens(value) for task_id, value in fragments.items()}
    for record_id, text in records:
        records_checked += 1
        normalized_corpus = normalize_typescript(text)
        if not normalized_corpus:
            continue
        corpus_tokens = _typescript_tokens(normalized_corpus)
        for task_id, normalized_task in fragments.items():
            if normalized_task == normalized_corpus:
                relationship = "equal"
            elif normalized_task in normalized_corpus:
                relationship = "task-contained-in-corpus"
            elif _contains_token_sequence(corpus_tokens, fragment_tokens[task_id]):
                relationship = "token-normalized-task-contained-in-corpus"
            else:
                continue
            matches.append(
                {"task": task_id, "record_id": str(record_id), "relationship": relationship}
            )
    if matches:
        details = ", ".join(f"{item['task']}={item['record_id']}" for item in matches[:10])
        raise ValueError(f"evaluation quarantine contamination detected: {details}")
    return {
        "schema_version": 1,
        "manifest_sha256": _sha256_bytes(Path(manifest_path).read_bytes()),
        "fixture_sha256": {fixture.fixture_id: fixture.sha256 for fixture in fixtures},
        "fixtures_verified": len(fixtures),
        "tasks_verified": sum(
            len(json.loads(fixture.path.read_text(encoding="utf-8"))["tasks"])
            for fixture in fixtures
        ),
        "fragments_verified": len(fragments),
        "corpus_records_checked": records_checked,
        "contamination_matches": 0,
    }


def _contains_token_sequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    limit = len(haystack) - len(needle) + 1
    return any(
        haystack[index] == first and haystack[index : index + len(needle)] == needle
        for index in range(limit)
    )
