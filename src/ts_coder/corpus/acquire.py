"""Bounded acquisition of immutable Git source snapshots.

The implementation never creates a worktree and never executes source-controlled
hooks or filters.  Files are copied from verified Git blob objects after their
paths, modes, and declared sizes have passed strict limits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import subprocess  # nosec B404
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_GITHUB_HOST = "github.com"
_REGULAR_MODES = {"100644", "100755"}
_SYMLINK_MODE = "120000"
_RECEIPT_VERSION = 1
_OWNER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,99})$")
_MAX_LIMITS = (64 * 1024 * 1024, 1_000_000, 10 * 1024 * 1024 * 1024, 256)
_MAX_TIMEOUT_SECONDS = 600
_MAX_TREE_OUTPUT = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 64 * 1024
_FETCH_BASE_OVERHEAD = 16 * 1024 * 1024
_FETCH_VARIABLE_OVERHEAD_CAP = 64 * 1024 * 1024
_RECEIPT_KEYS = {
    "schema_version",
    "source_id",
    "source_uri",
    "commit_sha",
    "tree_sha256",
    "file_count",
    "total_bytes",
    "license_sha256",
    "retrieved_at",
    "git_version",
}


class AcquisitionError(ValueError):
    """Raised when a source cannot be acquired without weakening a boundary."""


def _canonical_github_uri(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _GITHUB_HOST
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AcquisitionError("source_uri must be a canonical HTTPS GitHub repository URI")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        raise AcquisitionError("source_uri must identify exactly one GitHub owner/repository")
    repository = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    if not _OWNER.fullmatch(parts[0]) or not _REPOSITORY.fullmatch(repository):
        raise AcquisitionError("source_uri contains an invalid GitHub repository name")
    canonical = f"https://github.com/{parts[0]}/{repository}"
    if value != canonical:
        raise AcquisitionError(f"source_uri is not canonical; expected {canonical}")
    return canonical


def _validate_spec(
    source_id: str,
    source_uri: str,
    commit_sha: str,
    destination_root: Path,
    limits: tuple[int, int, int, int],
    timeout_seconds: int,
) -> tuple[str, Path, Path]:
    if not _SOURCE_ID.fullmatch(source_id):
        raise AcquisitionError("source_id must be a short stable identifier")
    uri = _canonical_github_uri(source_uri)
    if not _COMMIT.fullmatch(commit_sha):
        raise AcquisitionError("commit_sha must be exactly 40 lowercase hexadecimal characters")
    if any(value <= 0 for value in (*limits, timeout_seconds)):
        raise AcquisitionError("acquisition limits and timeout must be positive")
    if any(value > maximum for value, maximum in zip(limits, _MAX_LIMITS, strict=True)):
        raise AcquisitionError("acquisition limit exceeds the hard safety ceiling")
    if timeout_seconds > _MAX_TIMEOUT_SECONDS:
        raise AcquisitionError("acquisition timeout exceeds the hard safety ceiling")
    root = Path(destination_root)
    snapshot = root / source_id / commit_sha
    receipt = root / source_id / f"{commit_sha}.receipt.json"
    return uri, snapshot, receipt


def _git_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "",
            "GIT_SSH_COMMAND": "false",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_NO_LAZY_FETCH": "1",
        }
    )
    return env


def _run_git(
    arguments: list[str],
    *,
    timeout: int,
    cwd: Path | None = None,
    max_output: int = 4_000_000,
    file_size_limit: int | None = None,
) -> bytes:
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "protocol.ext.allow=never",
        *arguments,
    ]
    limit_preexec: Callable[[], None] | None = None
    if file_size_limit is not None:
        if file_size_limit <= 0:
            raise AcquisitionError("Git file-size quota must be positive")

        def apply_file_size_limit() -> None:
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_limit, file_size_limit))

        limit_preexec = apply_file_size_limit
    try:
        result = subprocess.run(  # nosec B603  # noqa: S603
            command,
            cwd=cwd,
            env=_git_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
            preexec_fn=limit_preexec,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AcquisitionError("bounded Git operation failed or timed out") from exc
    if len(result.stdout) > max_output or len(result.stderr) > max_output:
        raise AcquisitionError("Git operation exceeded its output limit")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise AcquisitionError(f"Git operation failed: {detail or 'no diagnostic'}")
    return result.stdout


def _fetch_quota(max_total_bytes: int) -> int:
    variable = min(max_total_bytes // 10, _FETCH_VARIABLE_OVERHEAD_CAP)
    return max_total_bytes + _FETCH_BASE_OVERHEAD + variable


def _fetch_commit(
    bare: Path,
    source_uri: str,
    commit_sha: str,
    timeout: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> None:
    _run_git(["init", "--bare", str(bare)], timeout=timeout)
    _run_git(["-C", str(bare), "remote", "add", "origin", source_uri], timeout=timeout)
    _run_git(
        [
            "-C",
            str(bare),
            "fetch",
            "--depth=1",
            "--no-tags",
            f"--filter=blob:limit={max_file_bytes + 1}",
            "origin",
            commit_sha,
        ],
        timeout=timeout,
        file_size_limit=_fetch_quota(max_total_bytes),
    )


def _verify_commit(bare: Path, commit_sha: str, timeout: int) -> None:
    kind = _run_git(["-C", str(bare), "cat-file", "-t", commit_sha], timeout=timeout).strip()
    resolved = (
        _run_git(["-C", str(bare), "rev-parse", "--verify", "FETCH_HEAD^{commit}"], timeout=timeout)
        .decode("ascii", errors="strict")
        .strip()
    )
    if kind != b"commit" or resolved != commit_sha:
        raise AcquisitionError("fetched object does not resolve to the approved commit")


def _safe_tree_path(raw: bytes, max_path_depth: int) -> PurePosixPath:
    if b"\x00" in raw:
        raise AcquisitionError("Git tree path contains NUL")
    try:
        value = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AcquisitionError("Git tree path is not valid UTF-8") from exc
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) > max_path_depth
    ):
        raise AcquisitionError("Git tree contains an unsafe or over-depth path")
    return path


def _tree_entries(
    bare: Path, commit_sha: str, timeout: int, max_files: int
) -> list[tuple[str, str, int, PurePosixPath]]:
    raw = _run_git(
        ["-C", str(bare), "ls-tree", "-r", "-z", "--full-tree", "-l", commit_sha],
        timeout=timeout,
        max_output=min(max(4_000_000, max_files * 1_024), _MAX_TREE_OUTPUT),
    )
    entries: list[tuple[str, str, int, PurePosixPath]] = []
    for item in raw.split(b"\x00"):
        if not item:
            continue
        try:
            metadata, raw_path = item.split(b"\t", 1)
            mode_b, kind_b, object_b, size_b = metadata.split(b" ", 3)
            mode, kind, object_id = mode_b.decode(), kind_b.decode(), object_b.decode()
        except (ValueError, UnicodeDecodeError) as exc:
            raise AcquisitionError("Git returned a malformed tree entry") from exc
        if (
            mode not in _REGULAR_MODES | {_SYMLINK_MODE}
            or kind != "blob"
            or not _COMMIT.fullmatch(object_id)
        ):
            raise AcquisitionError("Git tree contains a submodule or special entry")
        try:
            size = int(size_b)
        except ValueError as exc:
            raise AcquisitionError("Git returned a malformed blob size") from exc
        entries.append((mode, object_id, size, _safe_tree_path(raw_path, 10**9)))
        if len(entries) > max_files:
            raise AcquisitionError(f"source exceeds max_files={max_files}")
    return entries


def _internal_symlink_target(path: PurePosixPath, raw_target: bytes) -> PurePosixPath:
    if b"\x00" in raw_target:
        raise AcquisitionError("symlink target contains NUL")
    try:
        target = raw_target.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AcquisitionError("symlink target is not valid UTF-8") from exc
    target_path = PurePosixPath(target)
    if not target or target.startswith("/") or "\\" in target or target_path.is_absolute():
        raise AcquisitionError("symlink target must be a relative internal path")
    resolved_parts = list(path.parent.parts)
    for part in target_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not resolved_parts:
                raise AcquisitionError("symlink target escapes the snapshot")
            resolved_parts.pop()
        else:
            resolved_parts.append(part)
    if not resolved_parts:
        raise AcquisitionError("symlink target does not identify a snapshot entry")
    return PurePosixPath(*resolved_parts)


def _verify_internal_symlink(snapshot: Path, path: Path, raw_target: bytes) -> None:
    relative = PurePosixPath(path.relative_to(snapshot).as_posix())
    lexical_target = _internal_symlink_target(relative, raw_target)
    expected = snapshot.joinpath(*lexical_target.parts)
    try:
        resolved_snapshot = snapshot.resolve(strict=True)
        resolved_target = expected.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AcquisitionError("symlink target is dangling or cyclic") from exc
    if resolved_target != resolved_snapshot and resolved_snapshot not in resolved_target.parents:
        raise AcquisitionError("symlink target escapes the snapshot")


def _read_symlink_target(path: Path) -> bytes:
    try:
        return os.readlink(path).encode("utf-8", errors="strict")
    except (OSError, UnicodeEncodeError) as exc:
        raise AcquisitionError("symlink target is unreadable or not valid UTF-8") from exc


def _snapshot_paths(snapshot: Path, max_files: int) -> list[Path]:
    pending = [snapshot]
    paths: list[Path] = []
    visited_nodes = 0
    max_nodes = min(max(max_files * 4 + 1_024, 4_096), 2_000_000)
    while pending:
        current = pending.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda item: item.name, reverse=True)
        except OSError as exc:
            raise AcquisitionError("snapshot cannot be enumerated safely") from exc
        for entry in entries:
            visited_nodes += 1
            if visited_nodes > max_nodes:
                raise AcquisitionError("snapshot exceeds its bounded verification node count")
            path = Path(entry.path)
            if entry.is_symlink():
                paths.append(path)
            elif entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                paths.append(path)
            else:
                raise AcquisitionError("snapshot contains a special filesystem entry")
            if len(paths) > max_files:
                raise AcquisitionError("snapshot exceeds its receipt file count")
    return sorted(paths, key=lambda item: item.as_posix())


def _snapshot_fingerprint(
    snapshot: Path,
    *,
    max_files: int = _MAX_LIMITS[1],
    max_total_bytes: int = _MAX_LIMITS[2],
) -> tuple[str, int, int, dict[str, str]]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    licenses: dict[str, str] = {}
    for path in _snapshot_paths(snapshot, max_files):
        relative = path.relative_to(snapshot).as_posix()
        if path.is_symlink():
            target = _read_symlink_target(path)
            _verify_internal_symlink(snapshot, path, target)
            digest.update(b"symlink\x00" + relative.encode("utf-8") + b"\x00")
            digest.update(target + b"\n")
            count += 1
            total += len(target)
            if total > max_total_bytes:
                raise AcquisitionError("snapshot exceeds its receipt byte count")
            continue
        try:
            size = path.stat(follow_symlinks=False).st_size
        except OSError as exc:
            raise AcquisitionError("snapshot file cannot be inspected") from exc
        if size > _MAX_LIMITS[0] or total + size > max_total_bytes:
            raise AcquisitionError("snapshot exceeds its bounded verification byte count")
        with path.open("rb") as handle:
            content = handle.read(_MAX_LIMITS[0] + 1)
        if len(content) != size:
            raise AcquisitionError("snapshot file changed during verification")
        content_hash = hashlib.sha256(content).hexdigest()
        digest.update(b"regular\x00" + relative.encode("utf-8") + b"\x00")
        digest.update(str(len(content)).encode() + b"\x00")
        digest.update(content_hash.encode("ascii") + b"\n")
        count += 1
        total += len(content)
        if Path(relative).name.lower().startswith(("license", "copying")):
            licenses[relative] = content_hash
    return digest.hexdigest(), count, total, dict(sorted(licenses.items()))


def _materialize(
    bare: Path,
    snapshot: Path,
    commit_sha: str,
    *,
    max_file_bytes: int,
    max_files: int,
    max_total_bytes: int,
    max_path_depth: int,
    timeout: int,
) -> None:
    entries = _tree_entries(bare, commit_sha, timeout, max_files)
    total = 0
    symlinks: list[tuple[PurePosixPath, bytes]] = []
    for mode, object_id, size, path in entries:
        path = _safe_tree_path(path.as_posix().encode(), max_path_depth)
        if size < 0 or size > max_file_bytes:
            raise AcquisitionError(f"source blob exceeds max_file_bytes={max_file_bytes}")
        total += size
        if total > max_total_bytes:
            raise AcquisitionError(f"source exceeds max_total_bytes={max_total_bytes}")
        content = _run_git(
            ["-C", str(bare), "cat-file", "blob", object_id],
            timeout=timeout,
            max_output=max_file_bytes + 1,
        )
        if len(content) != size:
            raise AcquisitionError("Git blob size differs from its tree declaration")
        if mode == _SYMLINK_MODE:
            _internal_symlink_target(path, content)
            symlinks.append((path, content))
            continue
        destination = snapshot.joinpath(*path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(content)
    for path, target in symlinks:
        destination = snapshot.joinpath(*path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.symlink_to(target.decode("utf-8", errors="strict"))
        except OSError as exc:
            raise AcquisitionError("could not materialize an internal symlink") from exc
    for path, target in symlinks:
        _verify_internal_symlink(snapshot, snapshot.joinpath(*path.parts), target)


def verify_snapshot_receipt(
    snapshot_path: str | Path,
    receipt_path: str | Path,
    expected_uri: str,
    expected_sha: str,
) -> dict[str, Any]:
    """Verify the immutable identity and content digest of an acquired snapshot."""
    uri = _canonical_github_uri(expected_uri)
    if not _COMMIT.fullmatch(expected_sha):
        raise AcquisitionError("expected_sha must be exactly 40 lowercase hexadecimal characters")
    snapshot = Path(snapshot_path)
    receipt_file = Path(receipt_path)
    invalid_paths = (
        not snapshot.is_dir()
        or snapshot.is_symlink()
        or not receipt_file.is_file()
        or receipt_file.is_symlink()
    )
    if invalid_paths:
        raise AcquisitionError("snapshot and receipt must be regular existing paths")
    try:
        if receipt_file.stat().st_size > _MAX_RECEIPT_BYTES:
            raise AcquisitionError("snapshot receipt exceeds its size limit")
        with receipt_file.open("r", encoding="utf-8") as handle:
            receipt = json.loads(handle.read(_MAX_RECEIPT_BYTES + 1))
    except AcquisitionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError("snapshot receipt is unreadable") from exc
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise AcquisitionError("snapshot receipt has missing or unknown fields")
    source_id = receipt["source_id"]
    file_count = receipt["file_count"]
    total_bytes = receipt["total_bytes"]
    license_hashes = receipt["license_sha256"]
    timestamp = receipt["retrieved_at"]
    valid_scalars = (
        receipt["schema_version"] == _RECEIPT_VERSION
        and isinstance(source_id, str)
        and _SOURCE_ID.fullmatch(source_id) is not None
        and isinstance(receipt["tree_sha256"], str)
        and _SHA256.fullmatch(receipt["tree_sha256"]) is not None
        and isinstance(file_count, int)
        and not isinstance(file_count, bool)
        and 0 <= file_count <= _MAX_LIMITS[1]
        and isinstance(total_bytes, int)
        and not isinstance(total_bytes, bool)
        and 0 <= total_bytes <= _MAX_LIMITS[2]
        and isinstance(receipt["git_version"], str)
        and bool(receipt["git_version"].strip())
    )
    if not valid_scalars:
        raise AcquisitionError("snapshot receipt has invalid schema values")
    if not isinstance(license_hashes, dict) or any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        for path, digest in license_hashes.items()
    ):
        raise AcquisitionError("snapshot receipt has invalid license hashes")
    if not isinstance(timestamp, str) or _UTC_TIMESTAMP.fullmatch(timestamp) is None:
        raise AcquisitionError("snapshot receipt has an invalid UTC timestamp")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise AcquisitionError("snapshot receipt has an invalid UTC timestamp") from exc
    if parsed_timestamp.utcoffset() != timezone.utc.utcoffset(parsed_timestamp):
        raise AcquisitionError("snapshot receipt timestamp is not UTC")
    identity_mismatch = (
        receipt["source_uri"] != uri
        or receipt["commit_sha"] != expected_sha
        or snapshot.name != expected_sha
        or snapshot.parent.name != source_id
        or receipt_file.name != f"{expected_sha}.receipt.json"
        or receipt_file.parent != snapshot.parent
    )
    if identity_mismatch:
        raise AcquisitionError("snapshot receipt identity does not match the approved source")
    tree_hash, count, total, licenses = _snapshot_fingerprint(
        snapshot,
        max_files=file_count,
        max_total_bytes=total_bytes,
    )
    expected = {
        "tree_sha256": tree_hash,
        "file_count": count,
        "total_bytes": total,
        "license_sha256": licenses,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise AcquisitionError("snapshot content does not match its receipt")
    return receipt


def acquire_git_source(
    *,
    source_id: str,
    source_uri: str,
    commit_sha: str,
    destination_root: str | Path,
    max_file_bytes: int,
    max_files: int,
    max_total_bytes: int,
    max_path_depth: int,
    timeout_seconds: int = 120,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Acquire one approved GitHub commit into a bounded immutable snapshot."""
    uri, snapshot, receipt_path = _validate_spec(
        source_id,
        source_uri,
        commit_sha,
        Path(destination_root),
        (max_file_bytes, max_files, max_total_bytes, max_path_depth),
        timeout_seconds,
    )
    result: dict[str, Any] = {
        "source_id": source_id,
        "source_uri": uri,
        "commit_sha": commit_sha,
        "snapshot_path": str(snapshot),
        "receipt_path": str(receipt_path),
        "dry_run": dry_run,
    }
    if dry_run:
        if snapshot.exists() or receipt_path.exists():
            verify_snapshot_receipt(snapshot, receipt_path, uri, commit_sha)
        return result
    if snapshot.exists() or receipt_path.exists():
        if not snapshot.exists() or not receipt_path.exists():
            raise AcquisitionError("partial pre-existing snapshot state fails closed")
        existing_receipt = verify_snapshot_receipt(snapshot, receipt_path, uri, commit_sha)
        return {
            **existing_receipt,
            "snapshot_path": str(snapshot),
            "receipt_path": str(receipt_path),
        }

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{source_id}-", dir=snapshot.parent))
    bare = temp_root / "source.git"
    staged_snapshot = temp_root / "snapshot"
    try:
        _fetch_commit(
            bare,
            uri,
            commit_sha,
            timeout_seconds,
            max_file_bytes,
            max_total_bytes,
        )
        _verify_commit(bare, commit_sha, timeout_seconds)
        staged_snapshot.mkdir()
        _materialize(
            bare,
            staged_snapshot,
            commit_sha,
            max_file_bytes=max_file_bytes,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
            max_path_depth=max_path_depth,
            timeout=timeout_seconds,
        )
        tree_hash, file_count, total_bytes, license_hashes = _snapshot_fingerprint(staged_snapshot)
        git_version = _run_git(["--version"], timeout=timeout_seconds).decode().strip()
        receipt: dict[str, Any] = {
            "schema_version": _RECEIPT_VERSION,
            "source_id": source_id,
            "source_uri": uri,
            "commit_sha": commit_sha,
            "tree_sha256": tree_hash,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "license_sha256": license_hashes,
            "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "git_version": git_version,
        }
        staged_snapshot.rename(snapshot)
        receipt_temp = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        try:
            with receipt_temp.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
            receipt_temp.replace(receipt_path)
        except Exception:
            shutil.rmtree(snapshot, ignore_errors=True)
            receipt_temp.unlink(missing_ok=True)
            raise
        return {**receipt, "snapshot_path": str(snapshot), "receipt_path": str(receipt_path)}
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
