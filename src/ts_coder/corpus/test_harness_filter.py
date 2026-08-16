"""Detect compiler/test-harness scripts that are not ordinary source files.

TypeScript compiler repositories contain a large suite of executable fixture
descriptions (notably fourslash tests).  They are valid ``.ts`` files, but the
``@filename`` and ``verify.*`` directives are harness syntax rather than code a
coding model should learn to emit.  Keep ordinary unit/integration tests and
reject only files with unmistakable harness markers.
"""

from __future__ import annotations

import re
from pathlib import Path

_FILENAME_DIRECTIVE = re.compile(r"^\s*//\s*@filename\s*:", re.IGNORECASE | re.MULTILINE)
_FOURSLASH_REFERENCE = re.compile(
    r"<reference\s+path\s*=\s*[\"'](?:\.\.?/)*fourslash\.ts[\"']",
    re.IGNORECASE,
)
_VERIFY_CALL = re.compile(r"\bverify\.[A-Za-z_$][\w$]*\s*\(")
_HARNESS_CALL = re.compile(r"\b(?:goTo|edit|formatCode|baseline)\.[A-Za-z_$][\w$]*\s*\(")


def test_harness_reasons(path: Path, text: str) -> list[str]:
    """Return stable exclusion reasons for an unmistakable test harness file."""

    normalized = path.as_posix().lower()
    reasons: list[str] = []
    if "/fourslash/" in f"/{normalized}/" or normalized.startswith("fourslash/"):
        reasons.append("test-harness-fourslash-path")
    if _FILENAME_DIRECTIVE.search(text):
        reasons.append("test-harness-filename-directive")
    if _FOURSLASH_REFERENCE.search(text):
        reasons.append("test-harness-fourslash-reference")
    # ``verify.*`` and these editor/harness calls are intentionally paired with
    # a test path or a fourslash reference to avoid rejecting a project's own
    # legitimate ``verify`` helper in ordinary application code.
    test_path = "/test" in f"/{normalized}" or normalized.startswith("test/")
    if test_path and _VERIFY_CALL.search(text):
        reasons.append("test-harness-verify-api")
    if test_path and _HARNESS_CALL.search(text):
        reasons.append("test-harness-editor-api")
    return sorted(set(reasons))
