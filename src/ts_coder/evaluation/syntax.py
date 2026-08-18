"""Machine-readable TypeScript syntax evaluation via the local helper."""

from __future__ import annotations

import json
import subprocess  # nosec B404 - invokes only the pinned local helper
from pathlib import Path
from typing import Any

MAX_SOURCE_CHARS = 2_000_000


def _tool_root(tool_root: str | Path | None = None) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    requested = Path(tool_root) if tool_root is not None else project_root / "tools" / "typescript"
    if requested.is_symlink():
        raise ValueError("TypeScript tool root must not be a symlink")
    candidate = requested.resolve()
    if candidate != project_root and project_root not in candidate.parents:
        raise ValueError("TypeScript tool root must remain inside the project")
    return candidate


def parse_typescript(
    source: str,
    filename: str = "generated.ts",
    helper: str | Path | None = None,
    timeout: int = 10,
    *,
    tool_root: str | Path | None = None,
) -> dict[str, Any]:
    if len(source) > MAX_SOURCE_CHARS:
        return {
            "success": False,
            "diagnostics": [
                {"message": "source exceeds evaluation limit", "code": "SOURCE_TOO_LARGE"}
            ],
        }
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    expected_path = _tool_root(tool_root) / "dist" / "parse.js"
    requested = Path(helper) if helper else expected_path
    if expected_path.is_symlink() or requested.is_symlink():
        return {
            "success": False,
            "diagnostics": [
                {
                    "message": "helper path must not be a symlink",
                    "code": "HELPER_UNTRUSTED",
                }
            ],
        }
    expected = expected_path.resolve()
    target = requested.resolve()
    if target != expected:
        return {
            "success": False,
            "diagnostics": [
                {
                    "message": "helper path is outside the configured project toolchain",
                    "code": "HELPER_UNTRUSTED",
                }
            ],
        }
    if not target.is_file():
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
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0 and not result.stdout.strip():
            message = (
                result.stderr.strip()[-500:] or f"helper exited with status {result.returncode}"
            )
            return {
                "success": False,
                "diagnostics": [{"message": message, "code": "HELPER_ERROR"}],
            }
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise json.JSONDecodeError("helper response must be an object", result.stdout, 0)
        return value
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {
            "success": False,
            "diagnostics": [{"message": str(exc), "code": "HELPER_ERROR"}],
        }


def syntax_rate(sources: list[str]) -> float:
    return sum(parse_typescript(s).get("success", False) for s in sources) / max(1, len(sources))
