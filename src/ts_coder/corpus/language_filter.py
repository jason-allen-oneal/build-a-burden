"""TypeScript-only language boundary checks."""

import re
from pathlib import Path

ACCEPTED_SUFFIXES = {".ts", ".tsx"}


def is_typescript(path: Path, content: str) -> tuple[bool, str]:
    if path.suffix.lower() not in ACCEPTED_SUFFIXES:
        return False, "unsupported-extension"
    if path.name.endswith((".min.ts", ".bundle.ts")):
        return False, "bundled-filename"
    # Extension is necessary but not sufficient: require recognizable TS/JS syntax.
    syntax = re.search(
        r"\b(interface|type|class|function|const|let|enum|namespace|import|export|declare)\b|=>|<[A-Z][\w.]*",
        content,
    )
    if not syntax:
        return False, "no-typescript-syntax"
    return True, "typescript"
