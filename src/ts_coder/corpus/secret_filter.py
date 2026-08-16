"""Secret detection that never returns or logs matched values."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    "authorization-header": re.compile(
        r"authorization\s*[:=]\s*(?:['\"`])?(?:bearer|basic)\s+"
        r"[^\s'\"`]+(?:['\"`])?",
        re.I,
    ),
    "cloud-credential": re.compile(r"\bAKIA[0-9A-Z]{16}\b|aws_secret_access_key", re.I),
    "database-credential": re.compile(
        r"(?:postgres|mysql|mongodb)(?:ql)?://[^\s:'\"]+:[^\s@'\"]+@", re.I
    ),
    "credential-variable": re.compile(
        r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*"
        r"(?:['\"`][^'\"`]{8,}['\"`]|[A-Za-z0-9_+/=.-]{16,})",
        re.I,
    ),
}
_SAFE_MARKERS = ("EXAMPLE_NOT_A_SECRET", "TEST_TOKEN_DO_NOT_USE", "INVALID_FIXTURE_CREDENTIAL")


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    return -sum(
        (value.count(c) / len(value)) * math.log2(value.count(c) / len(value)) for c in set(value)
    )


def scan_secrets(path: Path, text: str) -> tuple[str, list[dict[str, str]]]:
    if path.name == ".env" or path.suffix.lower() in {".pem", ".key", ".p12"}:
        return "rejected", [
            {
                "category": "credential-file",
                "fingerprint": hashlib.sha256(path.name.encode()).hexdigest()[:12],
            }
        ]
    # Never remove fixture markers from the scan input.  Substring scrubbing
    # turns ``TEST_TOKEN_DO_NOT_USE_with_a_real_suffix`` into an apparently
    # safe value.  The only exemption is the complete marker check below for
    # a plainly named generic token variable.
    scrubbed = text
    findings = []
    for category, pattern in _PATTERNS.items():
        # A plainly named API-key/password field remains review-worthy even when
        # its value is a project marker. Generic ``token`` examples are allowed
        # so fixtures can document redaction without looking like credentials.
        target = text if category == "credential-variable" else scrubbed
        for match in pattern.finditer(target):
            if category == "credential-variable":
                assigned = re.split(r"[:=]", match.group(0), maxsplit=1)[-1]
                assigned = assigned.strip().strip(";,'\"`")
                # Only a complete marker in a generic token variable is exempt.
                # Marker prefixes with any suffix remain suspicious.
                tail_lines = text[match.end() :].splitlines()
                tail = tail_lines[0] if tail_lines else ""
                standalone = re.fullmatch(r"\s*[;,]?\s*(?://.*)?", tail) is not None
                if (
                    assigned in _SAFE_MARKERS
                    and standalone
                    and re.search(r"\btoken\b", match.group(0), re.I)
                ):
                    continue
            findings.append(
                {
                    "category": category,
                    "fingerprint": hashlib.sha256(match.group(0).encode()).hexdigest()[:12],
                }
            )
    for token in re.findall(r"[A-Za-z0-9_+/=-]{32,}", scrubbed):
        if _entropy(token) >= 4.3:
            findings.append(
                {
                    "category": "high-entropy-token",
                    "fingerprint": hashlib.sha256(token.encode()).hexdigest()[:12],
                }
            )
    return ("rejected" if findings else "clean"), findings
