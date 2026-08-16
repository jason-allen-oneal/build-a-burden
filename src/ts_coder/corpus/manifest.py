"""Append-only JSONL corpus manifest records."""

import hashlib
import json
from pathlib import Path


def stable_id(repository_id: str, relative_path: str, content_sha256: str) -> str:
    return hashlib.sha256(
        f"{repository_id}\0{relative_path}\0{content_sha256}".encode()
    ).hexdigest()


def write_manifest(path: Path, records: list[dict], append: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
        for r in sorted(records, key=lambda x: x["record_id"])
    )
    if append:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    else:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
