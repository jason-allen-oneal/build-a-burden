"""Deterministic JSONL document sharding."""

import hashlib
import json
from pathlib import Path


def write_shards(documents: list[dict], output_dir: Path, max_records: int = 1000) -> list[dict]:
    if max_records < 1:
        raise ValueError("max_records must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    ordered = sorted(documents, key=lambda x: x["record_id"])
    for index in range(0, len(ordered), max_records):
        path = output_dir / f"shard-{index // max_records:05d}.jsonl"
        payload = "".join(
            json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n"
            for x in ordered[index : index + max_records]
        )
        path.write_text(payload, encoding="utf-8", newline="\n")
        result.append(
            {
                "path": path.name,
                "records": len(ordered[index : index + max_records]),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )
    return result
