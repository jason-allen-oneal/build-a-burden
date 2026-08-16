"""Controlled TypeScript compilation adapter."""

from __future__ import annotations

import json
import subprocess  # nosec B404 - invokes only the pinned local helper
from pathlib import Path
from typing import Any

MAX_SOURCE_CHARS = 2_000_000


def compile_typescript(
    source: str, filename: str = "generated.ts", helper: str | Path | None = None, timeout: int = 10
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
    expected = Path(__file__).resolve().parents[3] / "tools" / "typescript" / "dist" / "compile.js"
    target = Path(helper).resolve() if helper else expected
    if target != expected.resolve():
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
    try:
        result = subprocess.run(  # nosec B603 B607 - fixed local Node helper, shell disabled
            ["node", str(target)],
            input=json.dumps({"filename": filename, "source": source}),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"success": False, "timeout": True, "diagnostics": []}
    except json.JSONDecodeError as exc:
        return {"success": False, "diagnostics": [{"message": str(exc), "code": "HELPER_ERROR"}]}
