"""Host-safe ingestion of controlled directory sources."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .deduplicate import assign_clusters, normalize_typescript, sha256_text
from .generated_filter import generated_status
from .language_filter import is_typescript
from .licensing import DEFAULT_ACCEPTED, detect_license
from .manifest import stable_id
from .quality_filter import assess_quality
from .secret_filter import scan_secrets
from .split import assign_splits
from .test_harness_filter import test_harness_reasons

REJECTED_PARTS = {"node_modules", "dist", "build", ".next", "coverage", ".cache", ".git", "vendor"}
REJECTED_SUFFIXES = {".map", ".lock", ".pem", ".key"}


def sanitize_source_uri(value: str) -> str:
    """Remove credentials, query strings, and fragments before manifest storage."""
    parsed = urlsplit(value)
    if not parsed.scheme:
        # A path-like source identifier may still contain a signed query or
        # fragment.  Preserve the path while dropping those components.
        return urlunsplit(("", "", parsed.path, "", ""))
    safe_netloc = ""
    if parsed.hostname:
        safe_netloc = parsed.hostname
        try:
            if parsed.port is not None:
                safe_netloc = f"{safe_netloc}:{parsed.port}"
        except ValueError:
            # Malformed ports are not provenance we can safely retain.
            safe_netloc = parsed.hostname
    return urlunsplit((parsed.scheme, safe_netloc, parsed.path, "", ""))


def _path_rejection(path: Path) -> str | None:
    if any(part in REJECTED_PARTS for part in path.parts):
        return "excluded-path"
    if path.suffix.lower() in REJECTED_SUFFIXES or path.name.endswith(".lock"):
        return "excluded-file-type"
    return None


def ingest_repository(
    repository: Path,
    *,
    source_uri: str | None = None,
    commit_sha: str = "fixture",
    retrieved_at: str | None = None,
    pipeline_version: str = "working-tree",
    seed: int = 42,
    max_file_bytes: int = 1_000_000,
    max_files: int = 10_000,
    max_total_bytes: int = 100_000_000,
    max_path_depth: int = 32,
    accepted_licenses: frozenset[str] = DEFAULT_ACCEPTED,
    approved_scope: tuple[str, ...] | None = None,
    near_duplicate_threshold: float = 0.85,
) -> tuple[list[dict], dict[str, str]]:
    if min(max_file_bytes, max_files, max_total_bytes, max_path_depth) <= 0:
        raise ValueError("ingestion limits must be positive")
    unauthorized_licenses = frozenset(accepted_licenses) - DEFAULT_ACCEPTED
    if unauthorized_licenses:
        raise ValueError(
            "accepted_licenses contains policy-disallowed values: "
            + ", ".join(sorted(unauthorized_licenses))
        )
    repository = repository.resolve()
    safe_source_uri = sanitize_source_uri(source_uri or f"fixture://{repository.name}")
    repo_id = hashlib.sha256(safe_source_uri.encode()).hexdigest()[:24]
    license_result = detect_license(repository, accepted_licenses)
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    records = []
    contents = {}
    paths = _bounded_paths(repository, max_files=max_files, max_path_depth=max_path_depth)
    regular_bytes = 0
    for path in paths:
        rel = path.relative_to(repository).as_posix()
        path_reason = _path_rejection(Path(rel))
        exclusions = [path_reason] if path_reason else []
        if not _scope_allows(rel, approved_scope):
            exclusions.append("outside-approved-scope")
        symlink = path.is_symlink()
        if symlink:
            raw = os.readlink(path).encode("utf-8", errors="replace")
            text = ""
            encoding_error = False
            exclusions.append("symlink")
        else:
            size = path.stat().st_size
            regular_bytes += size
            if regular_bytes > max_total_bytes:
                raise ValueError(f"repository exceeds max_total_bytes={max_total_bytes}")
            raw = path.open("rb").read(max_file_bytes + 1)
            oversized = size > max_file_bytes
            if oversized:
                raw = raw[:max_file_bytes]
                text = ""
                encoding_error = False
                exclusions.append("oversized")
            else:
                try:
                    text = raw.decode("utf-8")
                    encoding_error = False
                except UnicodeDecodeError:
                    text = raw.decode("utf-8", errors="replace")
                    encoding_error = True
        content_hash = hashlib.sha256(raw).hexdigest()
        rid = stable_id(repo_id, rel, content_hash)
        if len(Path(rel).parts) > max_path_depth:
            exclusions.append("path-depth-limit")
        contextual = Path(rel).name in {"package.json", "tsconfig.json"}
        language_ok, language_reason = is_typescript(Path(rel), text)
        if contextual:
            language_ok, language_reason = True, "contextual-metadata"
        if not language_ok:
            exclusions.append(language_reason)
        gen, gen_reasons = generated_status(Path(rel), text)
        if gen == "generated":
            exclusions.extend(gen_reasons)
        exclusions.extend(test_harness_reasons(Path(rel), text))
        secret_status, findings = scan_secrets(Path(rel), text)
        if secret_status != "clean":  # nosec B105 - scanner status, not a credential
            exclusions.extend(sorted({f"secret:{f['category']}" for f in findings}))
        score, quality_reasons, quality = assess_quality(text)
        exclusions.extend(quality_reasons)
        if encoding_error:
            exclusions.append("invalid-utf8")
        if license_result.status != "accepted":
            exclusions.append(f"license:{license_result.status}")
        record = {
            "record_id": rid,
            "repository_id": repo_id,
            "source_uri": safe_source_uri,
            "commit_sha": commit_sha,
            "retrieved_at": timestamp,
            "license_spdx": license_result.spdx,
            "license_status": license_result.status,
            "relative_path": rel,
            "language": (
                "typescript"
                if Path(rel).suffix.lower() in {".ts", ".tsx"}
                else ("metadata" if contextual else "other")
            ),
            "source_type": _source_type(rel),
            "size_bytes": path.stat().st_size if not symlink else len(raw),
            "line_count": len(text.splitlines()),
            "content_sha256": content_hash,
            "content_hash_scope": "prefix" if "oversized" in exclusions else "full",
            "normalized_sha256": sha256_text(normalize_typescript(text)),
            "quality_score": round(score, 6),
            "quality_metrics": quality,
            "secret_scan_status": secret_status,
            "generated_status": gen,
            "dedup_cluster": "",
            "split": "excluded",
            "pipeline_version": pipeline_version,
            "included": not exclusions,
            "exclusion_reasons": sorted(set(exclusions)),
        }
        records.append(record)
        contents[rid] = text
    assign_clusters(records, contents, near_duplicate_threshold)
    # Keep only one exact/normalized duplicate, but retain all records and cluster identity.
    seen = set()
    for record in sorted(records, key=lambda x: x["record_id"]):
        key = record["normalized_sha256"]
        if record["included"] and key in seen:
            record["included"] = False
            record["exclusion_reasons"].append("duplicate")
        elif record["included"]:
            seen.add(key)
    assign_splits(records, seed)
    return records, contents


def _scope_allows(path: str, scope: tuple[str, ...] | None) -> bool:
    """Apply machine-readable approval scope without hiding rejected files.

    Entries use ``include:<glob>`` and ``exclude:<glob>``.  The legacy
    ``typescript-source`` value remains a broad sentinel for project-authored
    fixtures; the language and generated-code filters still apply after it.
    """
    if scope is None or "typescript-source" in scope:
        return True
    includes = [item.removeprefix("include:") for item in scope if item.startswith("include:")]
    excludes = [item.removeprefix("exclude:") for item in scope if item.startswith("exclude:")]
    if not includes:
        return False
    included = any(fnmatchcase(path, pattern) for pattern in includes)
    excluded = any(fnmatchcase(path, pattern) for pattern in excludes)
    return included and not excluded


def _source_type(path: str) -> str:
    if path.endswith(".d.ts"):
        return "declaration"
    if path.endswith(".tsx"):
        return "code"
    if "/test" in path.lower() or path.lower().startswith("test"):
        return "test"
    if path.endswith(("package.json", "tsconfig.json")):
        return "metadata"
    return "code" if path.endswith(".ts") else "metadata"


def _bounded_paths(repository: Path, *, max_files: int, max_path_depth: int) -> list[Path]:
    """Enumerate without materializing an unbounded recursive glob.

    Directory traversal is deterministic and records files inside rejected
    build/vendor paths as excluded manifest entries. A source exceeding the
    file limit fails closed before the caller reads the remaining content.
    """
    pending = [repository]
    paths: list[Path] = []
    while pending:
        current = pending.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda item: item.name, reverse=True)
        except OSError as exc:
            raise ValueError(f"cannot enumerate repository path {current}") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(repository)
            if len(relative.parts) > max_path_depth:
                continue
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
                continue
            if entry.is_file(follow_symlinks=False) or entry.is_symlink():
                paths.append(path)
                if len(paths) > max_files:
                    raise ValueError(f"repository exceeds max_files={max_files}")
    return sorted(paths)
