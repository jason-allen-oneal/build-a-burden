"""Machine-readable TypeScript syntax evaluation via the local helper."""

from __future__ import annotations

import json
import subprocess  # nosec B404 - invokes only the pinned local helper
from pathlib import Path
from typing import Any

MAX_SOURCE_CHARS = 2_000_000


def _helper() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / "typescript" / "dist" / "parse.js"


def parse_typescript(
    source: str, filename: str = "generated.ts", helper: str | Path | None = None
) -> dict[str, Any]:
    if len(source) > MAX_SOURCE_CHARS:
        return {
            "success": False,
            "diagnostics": [
                {"message": "source exceeds evaluation limit", "code": "SOURCE_TOO_LARGE"}
            ],
        }
    target = Path(helper).resolve() if helper else _helper()
    if target != _helper().resolve():
        return {
            "success": False,
            "diagnostics": [
                {
                    "message": "helper path is outside the project toolchain",
                    "code": "HELPER_UNTRUSTED",
                }
            ],
        }
    if not target.exists():
        return {
            "success": False,
            "diagnostics": [
                {"message": "TypeScript helper is not built", "code": "HELPER_MISSING"}
            ],
        }
    payload = json.dumps({"filename": filename, "source": source})
    try:
        result = subprocess.run(  # nosec B603 B607 - fixed local Node helper, shell disabled
            ["node", str(target)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"success": False, "diagnostics": [{"message": str(exc), "code": "HELPER_ERROR"}]}


def syntax_rate(sources: list[str]) -> float:
    return sum(parse_typescript(s).get("success", False) for s in sources) / max(1, len(sources))
