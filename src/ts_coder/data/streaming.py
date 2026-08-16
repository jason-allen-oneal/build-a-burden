"""Streaming document-shard contracts for scale-out training.

The milestone-1 trainer materializes a small list of examples.  That is useful
for correctness tests, but it cannot be the data interface for a serious run:
the whole corpus must not be resident in one process and a worker must be able
to resume from a reproducible position.  This module intentionally contains no
framework-specific dataloader.  It defines the portable, auditable contract
that a future single-process, DDP, or FSDP loader can implement.

Shard files are newline-delimited JSON records.  The iterator reads one line at
a time, validates the small envelope, and exposes a line offset as the resume
position.  Partitioning is by a stable record-id hash, not process-local order,
so changing worker count does not silently change which examples are seen by a
given global stream.  A future launcher may replace hash partitioning with a
coordinated sampler, but it must preserve the cursor and manifest contracts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class ShardDescriptor:
    """Immutable identity and accounting metadata for one JSONL shard."""

    shard_id: str
    path: str
    records: int
    tokens: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.shard_id or Path(self.shard_id).name != self.shard_id:
            raise ValueError("shard_id must be a non-empty filename-safe identifier")
        if not self.path or Path(self.path).is_absolute():
            raise ValueError("shard path must be a non-empty relative path")
        resolved = Path(self.path)
        if ".." in resolved.parts:
            raise ValueError("shard path must not contain parent traversal")
        if self.records < 0:
            raise ValueError("shard records must not be negative")
        if self.tokens is not None and self.tokens < 0:
            raise ValueError("shard tokens must not be negative")
        if self.sha256 is not None and (
            len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise ValueError("shard sha256 must be a lowercase SHA-256 digest")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ShardDescriptor:
        allowed = {"shard_id", "path", "records", "tokens", "sha256"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown shard descriptor keys: {sorted(unknown)}")
        if not isinstance(value.get("shard_id"), str) or not isinstance(value.get("path"), str):
            raise ValueError("shard_id and path are required strings")
        records = value.get("records")
        if not isinstance(records, int) or isinstance(records, bool):
            raise ValueError("shard records must be an integer")
        tokens = value.get("tokens")
        if tokens is not None and (not isinstance(tokens, int) or isinstance(tokens, bool)):
            raise ValueError("shard tokens must be an integer or null")
        sha256 = value.get("sha256")
        if sha256 is not None and not isinstance(sha256, str):
            raise ValueError("shard sha256 must be a string or null")
        return cls(str(value["shard_id"]), str(value["path"]), records, tokens, sha256)


@dataclass(frozen=True)
class ShardManifest:
    """Versioned shard index consumed by streaming loaders."""

    shards: tuple[ShardDescriptor, ...]
    tokenizer_hash: str
    source_manifest_hash: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported shard manifest schema_version")
        if not self.tokenizer_hash or not self.source_manifest_hash:
            raise ValueError("shard manifest requires tokenizer and source manifest hashes")
        ids = [shard.shard_id for shard in self.shards]
        if len(ids) != len(set(ids)):
            raise ValueError("shard ids must be unique")
        paths = [shard.path for shard in self.shards]
        if len(paths) != len(set(paths)):
            raise ValueError("shard paths must be unique")

    @property
    def records(self) -> int:
        return sum(shard.records for shard in self.shards)

    @property
    def tokens(self) -> int | None:
        if any(shard.tokens is None for shard in self.shards):
            return None
        return sum(int(shard.tokens or 0) for shard in self.shards)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tokenizer_hash": self.tokenizer_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "records": self.records,
            "tokens": self.tokens,
            "shards": [asdict(shard) for shard in self.shards],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ShardManifest:
        allowed = {
            "schema_version",
            "tokenizer_hash",
            "source_manifest_hash",
            "records",
            "tokens",
            "shards",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown shard manifest keys: {sorted(unknown)}")
        if value.get("schema_version", 1) != 1:
            raise ValueError("unsupported shard manifest schema_version")
        if not isinstance(value.get("tokenizer_hash"), str) or not isinstance(
            value.get("source_manifest_hash"), str
        ):
            raise ValueError("shard manifest hashes are required strings")
        raw_shards = value.get("shards")
        if not isinstance(raw_shards, list):
            raise ValueError("shard manifest shards must be a list")
        shards = tuple(
            ShardDescriptor.from_mapping(item) for item in raw_shards if isinstance(item, Mapping)
        )
        if len(shards) != len(raw_shards):
            raise ValueError("each shard descriptor must be a mapping")
        manifest = cls(shards, value["tokenizer_hash"], value["source_manifest_hash"])
        for field in ("records", "tokens"):
            if field not in value:
                continue
            expected = getattr(manifest, field)
            if value[field] != expected:
                raise ValueError(f"shard manifest {field} total does not match descriptors")
        return manifest


@dataclass(frozen=True)
class DataCursor:
    """Portable resume position for a partitioned shard stream.

    ``record_offset`` is the next zero-based JSONL line to inspect within
    ``shard_index``.  It is deliberately a line offset rather than an in-memory
    iterator position.  ``token_offset`` is an objective-loader-defined token
    window offset within that record (zero means the next record). Cursors are
    valid only with the same source manifest, tokenizer, rank, and world size
    recorded by the checkpoint.
    """

    epoch: int = 0
    shard_index: int = 0
    record_offset: int = 0
    token_offset: int = 0
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        for name in ("epoch", "shard_index", "record_offset", "token_offset", "rank", "world_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"cursor {name} must be an integer")
        if (
            self.epoch < 0
            or self.shard_index < 0
            or self.record_offset < 0
            or self.token_offset < 0
        ):
            raise ValueError("cursor positions must not be negative")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("cursor rank/world_size is invalid")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DataCursor:
        expected = {"epoch", "shard_index", "record_offset", "token_offset", "rank", "world_size"}
        if set(value) != expected:
            raise ValueError("data cursor schema is incomplete or has unknown fields")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True)
class StreamRecord:
    """A validated record with enough identity to advance a cursor."""

    shard_index: int
    record_offset: int
    record_id: str
    value: dict[str, Any]


def _record_owner(record_id: str, world_size: int) -> int:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % world_size


def iter_jsonl_records(
    path: str | Path, *, max_line_bytes: int = 16 * 1024 * 1024
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield validated JSONL records without reading the shard into memory."""

    if max_line_bytes < 1:
        raise ValueError("max_line_bytes must be positive")
    shard_path = Path(path)
    if shard_path.is_symlink() or not shard_path.is_file():
        raise ValueError("shard path must be a regular file")
    with shard_path.open("rb") as handle:
        for offset, raw in enumerate(handle):
            if len(raw) > max_line_bytes:
                raise ValueError(f"shard record exceeds max_line_bytes at offset {offset}")
            try:
                decoded = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSONL shard record at offset {offset}") from exc
            if not isinstance(decoded, dict):
                raise ValueError(f"shard record at offset {offset} must be an object")
            record = decoded.get("record")
            if not isinstance(record, dict) or not isinstance(record.get("record_id"), str):
                raise ValueError(f"shard record at offset {offset} lacks record.record_id")
            if not isinstance(decoded.get("text"), str):
                raise ValueError(f"shard record at offset {offset} lacks text")
            yield offset, decoded


class StreamingShardDataset:
    """Deterministic, bounded-memory iterator over a shard manifest."""

    def __init__(
        self,
        manifest: ShardManifest,
        root: str | Path,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_line_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank/world_size is invalid")
        self.manifest = manifest
        self.root = Path(root).resolve()
        self.rank = rank
        self.world_size = world_size
        self.max_line_bytes = max_line_bytes

    def _path(self, descriptor: ShardDescriptor) -> Path:
        path = (self.root / descriptor.path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"shard path escapes root: {descriptor.path}")
        return path

    def iter_records(self, cursor: DataCursor | None = None) -> Iterator[StreamRecord]:
        """Yield records assigned to this rank, starting at ``cursor``."""

        position = cursor or DataCursor(rank=self.rank, world_size=self.world_size)
        if position.rank != self.rank or position.world_size != self.world_size:
            raise ValueError("cursor partition does not match dataset partition")
        if position.shard_index > len(self.manifest.shards):
            raise ValueError("cursor shard_index is outside manifest")
        for shard_index in range(position.shard_index, len(self.manifest.shards)):
            descriptor = self.manifest.shards[shard_index]
            first_offset = position.record_offset if shard_index == position.shard_index else 0
            for record_offset, value in iter_jsonl_records(
                self._path(descriptor), max_line_bytes=self.max_line_bytes
            ):
                if record_offset < first_offset:
                    continue
                record_id = value["record"]["record_id"]
                if _record_owner(record_id, self.world_size) != self.rank:
                    continue
                yield StreamRecord(shard_index, record_offset, record_id, value)

    def cursor_after(
        self, record: StreamRecord, *, token_offset: int = 0, epoch: int = 0
    ) -> DataCursor:
        """Return the next line position after a yielded record."""

        if record.shard_index >= len(self.manifest.shards):
            raise ValueError("record shard index is outside manifest")
        return DataCursor(
            epoch=epoch,
            shard_index=record.shard_index,
            record_offset=record.record_offset + 1,
            token_offset=token_offset,
            rank=self.rank,
            world_size=self.world_size,
        )
