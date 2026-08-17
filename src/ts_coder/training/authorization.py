"""Structured authorization gates for expensive training runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..reproducibility import sha256_file

AuthorizationStatus = Literal["approved", "rejected", "revoked"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_KEYS = {
    "schema_version",
    "authorization_id",
    "run_name",
    "training_config_sha256",
    "model_config_sha256",
    "manifest_sha256",
    "corpus_sha256",
    "tokenizer_sha256",
    "max_tokens",
    "status",
    "approved_by",
    "approved_at",
    "notes",
}


class TrainingAuthorizationError(ValueError):
    """Raised when a costly training run is not explicitly authorized."""


@dataclass(frozen=True)
class TrainingAuthorization:
    authorization_id: str
    run_name: str
    training_config_sha256: str
    model_config_sha256: str
    manifest_sha256: str
    corpus_sha256: str
    tokenizer_sha256: str
    max_tokens: int
    status: AuthorizationStatus
    approved_by: str
    approved_at: str
    notes: str = ""
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingAuthorization:
        if set(value) != _REQUIRED_KEYS:
            raise TrainingAuthorizationError("training authorization has missing or unknown fields")
        string_fields = {
            "authorization_id",
            "run_name",
            "training_config_sha256",
            "model_config_sha256",
            "manifest_sha256",
            "corpus_sha256",
            "tokenizer_sha256",
            "status",
            "approved_by",
            "approved_at",
            "notes",
        }
        if any(not isinstance(value[field], str) for field in string_fields):
            raise TrainingAuthorizationError("training authorization string field is invalid")
        if not isinstance(value["schema_version"], int) or isinstance(
            value["schema_version"], bool
        ):
            raise TrainingAuthorizationError("training authorization schema_version is invalid")
        if not isinstance(value["max_tokens"], int) or isinstance(value["max_tokens"], bool):
            raise TrainingAuthorizationError("training authorization max_tokens is invalid")
        status = value["status"]
        if status not in {"approved", "rejected", "revoked"}:
            raise TrainingAuthorizationError("authorization status is invalid")
        record = cls(
            authorization_id=value["authorization_id"],
            run_name=value["run_name"],
            training_config_sha256=value["training_config_sha256"],
            model_config_sha256=value["model_config_sha256"],
            manifest_sha256=value["manifest_sha256"],
            corpus_sha256=value["corpus_sha256"],
            tokenizer_sha256=value["tokenizer_sha256"],
            max_tokens=value["max_tokens"],
            status=status,
            approved_by=value["approved_by"],
            approved_at=value["approved_at"],
            notes=value["notes"],
            schema_version=value["schema_version"],
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.schema_version != 1:
            raise TrainingAuthorizationError("unsupported training authorization schema")
        if not self.run_name or Path(self.run_name).name != self.run_name:
            raise TrainingAuthorizationError("authorization run_name is invalid")
        if _SHA256.fullmatch(self.authorization_id) is None:
            raise TrainingAuthorizationError("authorization_id is invalid")
        for name in (
            "training_config_sha256",
            "model_config_sha256",
            "manifest_sha256",
            "corpus_sha256",
            "tokenizer_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise TrainingAuthorizationError(f"authorization {name} is invalid")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise TrainingAuthorizationError("authorization max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise TrainingAuthorizationError("authorization max_tokens must be positive")
        if self.status not in {"approved", "rejected", "revoked"}:
            raise TrainingAuthorizationError("authorization status is invalid")
        if not self.approved_by.strip():
            raise TrainingAuthorizationError("authorization requires an approver")
        if not self.approved_at.endswith("Z"):
            raise TrainingAuthorizationError("authorization approved_at must be UTC")
        try:
            timestamp = datetime.fromisoformat(self.approved_at[:-1] + "+00:00")
        except ValueError as exc:
            raise TrainingAuthorizationError("authorization approved_at is invalid") from exc
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise TrainingAuthorizationError("authorization approved_at must be UTC")
        expected = training_authorization_id(
            self.run_name,
            self.training_config_sha256,
            self.model_config_sha256,
            self.manifest_sha256,
            self.corpus_sha256,
            self.tokenizer_sha256,
            self.max_tokens,
            self.status,
            self.approved_at,
        )
        if self.authorization_id != expected:
            raise TrainingAuthorizationError("authorization_id does not match run identity")


def training_authorization_id(
    run_name: str,
    training_config_sha256: str,
    model_config_sha256: str,
    manifest_sha256: str,
    corpus_sha256: str,
    tokenizer_sha256: str,
    max_tokens: int,
    status: str,
    approved_at: str,
) -> str:
    payload = "\0".join(
        (
            run_name,
            training_config_sha256,
            model_config_sha256,
            manifest_sha256,
            corpus_sha256,
            tokenizer_sha256,
            str(max_tokens),
            status,
            approved_at,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_training_authorizations(path: str | Path) -> list[TrainingAuthorization]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TrainingAuthorizationError(
            f"training authorization ledger is not a regular file: {source}"
        )
    records: list[TrainingAuthorization] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingAuthorizationError(
                f"invalid training authorization JSON at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise TrainingAuthorizationError(
                f"training authorization line {line_number} is not an object"
            )
        record = TrainingAuthorization.from_dict(value)
        if record.authorization_id in seen:
            raise TrainingAuthorizationError(
                f"duplicate training authorization at line {line_number}"
            )
        seen.add(record.authorization_id)
        records.append(record)
    return records


def make_training_authorization(
    *,
    run_name: str,
    training_config: str | Path,
    model_config: str | Path,
    manifest: str | Path,
    corpus: str | Path,
    tokenizer: str | Path,
    max_tokens: int,
    approved_by: str,
    approved_at: str,
    status: AuthorizationStatus = "approved",
    notes: str = "",
) -> TrainingAuthorization:
    training_hash = sha256_file(training_config)
    model_hash = sha256_file(model_config)
    manifest_hash = sha256_file(manifest)
    corpus_hash = sha256_file(corpus)
    tokenizer_hash = sha256_file(tokenizer)
    record = TrainingAuthorization(
        authorization_id=training_authorization_id(
            run_name,
            training_hash,
            model_hash,
            manifest_hash,
            corpus_hash,
            tokenizer_hash,
            max_tokens,
            status,
            approved_at,
        ),
        run_name=run_name,
        training_config_sha256=training_hash,
        model_config_sha256=model_hash,
        manifest_sha256=manifest_hash,
        corpus_sha256=corpus_hash,
        tokenizer_sha256=tokenizer_hash,
        max_tokens=max_tokens,
        status=status,
        approved_by=approved_by,
        approved_at=approved_at,
        notes=notes,
    )
    record.validate()
    return record


def append_training_authorization(path: str | Path, record: TrainingAuthorization) -> None:
    record.validate()
    destination = Path(path)
    existing = load_training_authorizations(destination) if destination.exists() else []
    if any(item.authorization_id == record.authorization_id for item in existing):
        raise TrainingAuthorizationError("authorization_id already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")


def require_training_authorization(
    ledger: str | Path,
    *,
    run_name: str,
    training_config: str | Path,
    model_config: str | Path,
    manifest: str | Path,
    corpus: str | Path,
    tokenizer: str | Path,
    max_tokens: int,
) -> TrainingAuthorization:
    """Require the latest exact approval for a costly training identity."""

    identity = {
        "run_name": run_name,
        "training_config_sha256": sha256_file(training_config),
        "model_config_sha256": sha256_file(model_config),
        "manifest_sha256": sha256_file(manifest),
        "corpus_sha256": sha256_file(corpus),
        "tokenizer_sha256": sha256_file(tokenizer),
        "max_tokens": max_tokens,
    }
    matches = [
        record
        for record in load_training_authorizations(ledger)
        if all(getattr(record, key) == value for key, value in identity.items())
    ]
    if not matches:
        raise TrainingAuthorizationError(
            "no exact authorization exists for this run/config/model/data/tokenizer identity"
        )
    latest = matches[-1]
    if latest.status != "approved":
        raise TrainingAuthorizationError(f"latest training authorization status is {latest.status}")
    return latest
