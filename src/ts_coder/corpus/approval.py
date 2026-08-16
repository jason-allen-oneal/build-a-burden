"""Explicit, append-only approval decisions for real-source intake.

The approval ledger is intentionally separate from the per-file corpus
manifest.  A source may be approved only for an exact URI, revision, and
license decision; a later removal is represented by another ledger record and
never by rewriting history.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from .ingest import sanitize_source_uri
from .licensing import DEFAULT_ACCEPTED

ApprovalStatus = Literal["approved", "review", "rejected", "removed"]
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_FIXTURE_REVISION = "fixture"
_REQUIRED_KEYS = {
    "approval_id",
    "source_id",
    "source_uri",
    "commit_sha",
    "license_spdx",
    "status",
    "approved_by",
    "approved_at",
    "scope",
    "notes",
    "supersedes",
}
_STATUSES = {"approved", "review", "rejected", "removed"}


class ApprovalManifestError(ValueError):
    """Raised when an approval ledger is malformed or unsafe to use."""


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    source_id: str
    source_uri: str
    commit_sha: str
    license_spdx: str
    status: ApprovalStatus
    approved_by: str
    approved_at: str
    scope: tuple[str, ...]
    notes: str = ""
    supersedes: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> ApprovalRecord:
        if set(value) != _REQUIRED_KEYS:
            raise ApprovalManifestError("approval record has missing or unknown fields")
        scope = value["scope"]
        if not isinstance(scope, list) or not all(isinstance(item, str) for item in scope):
            raise ApprovalManifestError("approval scope must be a list of strings")
        status = value["status"]
        if status not in _STATUSES:
            raise ApprovalManifestError(f"unsupported approval status: {status}")
        record = cls(
            approval_id=str(value["approval_id"]),
            source_id=str(value["source_id"]),
            source_uri=str(value["source_uri"]),
            commit_sha=str(value["commit_sha"]),
            license_spdx=str(value["license_spdx"]),
            status=status,
            approved_by=str(value["approved_by"]),
            approved_at=str(value["approved_at"]),
            scope=tuple(scope),
            notes=str(value["notes"]),
            supersedes=None if value["supersedes"] is None else str(value["supersedes"]),
        )
        validate_approval_record(record)
        return record


def approval_id(
    source_id: str,
    source_uri: str,
    commit_sha: str,
    license_spdx: str,
    status: str = "approved",
    approved_at: str = "",
) -> str:
    return hashlib.sha256(
        f"{source_id}\0{source_uri}\0{commit_sha}\0{license_spdx}\0{status}\0{approved_at}".encode()
    ).hexdigest()


def validate_approval_record(record: ApprovalRecord) -> None:
    if not _SOURCE_ID.fullmatch(record.source_id):
        raise ApprovalManifestError("source_id must be a short stable identifier")
    if sanitize_source_uri(record.source_uri) != record.source_uri:
        raise ApprovalManifestError(
            "source_uri must not contain credentials, queries, or fragments"
        )
    parsed = urlsplit(record.source_uri)
    if not parsed.scheme or (parsed.username or parsed.password):
        raise ApprovalManifestError("source_uri must be an explicit URI without credentials")
    if record.commit_sha != _FIXTURE_REVISION and not _COMMIT_SHA.fullmatch(record.commit_sha):
        raise ApprovalManifestError("commit_sha must be a 40-character lowercase revision")
    if record.license_spdx not in DEFAULT_ACCEPTED:
        raise ApprovalManifestError("approval license is outside the conservative allowlist")
    if record.status == "approved":
        if not record.approved_by.strip() or not record.approved_at.strip() or not record.scope:
            raise ApprovalManifestError("approved sources require approver, timestamp, and scope")
    for item in record.scope:
        if item == "typescript-source":
            continue
        if not item.startswith(("include:", "exclude:")):
            raise ApprovalManifestError(
                "approval scope entries must use include:<glob> or exclude:<glob>"
            )
        pattern = item.split(":", 1)[1]
        parsed_pattern = PurePosixPath(pattern)
        if (
            not pattern
            or "\0" in pattern
            or "\\" in pattern
            or parsed_pattern.is_absolute()
            or ".." in parsed_pattern.parts
        ):
            raise ApprovalManifestError("approval scope contains an unsafe path pattern")
    if record.approval_id != approval_id(
        record.source_id,
        record.source_uri,
        record.commit_sha,
        record.license_spdx,
        record.status,
        record.approved_at,
    ):
        raise ApprovalManifestError("approval_id does not match source identity")
    if record.supersedes == record.approval_id:
        raise ApprovalManifestError("approval record cannot supersede itself")


def load_approval_manifest(path: str | Path) -> list[ApprovalRecord]:
    manifest = Path(path)
    if not manifest.is_file() or manifest.is_symlink():
        raise ApprovalManifestError(f"approval manifest is not a regular file: {manifest}")
    records: list[ApprovalRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ApprovalManifestError(f"invalid JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ApprovalManifestError(f"approval line {line_number} is not an object")
        record = ApprovalRecord.from_dict(value)
        if record.approval_id in seen:
            raise ApprovalManifestError(f"duplicate approval_id at line {line_number}")
        seen.add(record.approval_id)
        records.append(record)
    return records


def append_approval(path: str | Path, record: ApprovalRecord) -> None:
    """Append one validated decision without rewriting existing history."""
    validate_approval_record(record)
    destination = Path(path)
    existing = load_approval_manifest(destination) if destination.exists() else []
    if any(item.approval_id == record.approval_id for item in existing):
        raise ApprovalManifestError("approval_id already exists; append a distinct decision")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")


def require_approved_source(
    records: list[ApprovalRecord], source_id: str, source_uri: str, commit_sha: str
) -> ApprovalRecord:
    """Return the latest exact approval or fail closed."""
    safe_uri = sanitize_source_uri(source_uri)
    matches = [
        record
        for record in records
        if record.source_id == source_id
        and record.source_uri == safe_uri
        and record.commit_sha == commit_sha
    ]
    if not matches:
        raise ApprovalManifestError(
            f"source is not approved for exact URI/revision: {source_id}@{commit_sha}"
        )
    latest = matches[-1]
    if latest.status != "approved":
        raise ApprovalManifestError(
            f"source approval status is {latest.status}: {source_id}@{commit_sha}"
        )
    return latest


def make_approval(
    *,
    source_id: str,
    source_uri: str,
    commit_sha: str,
    license_spdx: str,
    approved_by: str,
    approved_at: str,
    scope: list[str],
    status: ApprovalStatus = "approved",
    notes: str = "",
    supersedes: str | None = None,
) -> ApprovalRecord:
    safe_uri = sanitize_source_uri(source_uri)
    if safe_uri != source_uri:
        raise ApprovalManifestError(
            "source_uri must not contain credentials, queries, or fragments"
        )
    record = ApprovalRecord(
        approval_id=approval_id(source_id, safe_uri, commit_sha, license_spdx, status, approved_at),
        source_id=source_id,
        source_uri=safe_uri,
        commit_sha=commit_sha,
        license_spdx=license_spdx,
        status=status,
        approved_by=approved_by,
        approved_at=approved_at,
        scope=tuple(scope),
        notes=notes,
        supersedes=supersedes,
    )
    validate_approval_record(record)
    return record
