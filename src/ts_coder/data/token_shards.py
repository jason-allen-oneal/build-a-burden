import json
from pathlib import Path


def write_token_shard(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in samples),
        encoding="utf-8",
        newline="\n",
    )


def read_token_shard(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
