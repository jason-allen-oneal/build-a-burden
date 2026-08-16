"""Conservative, configurable source-license decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ACCEPTED = frozenset(
    {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD", "CC0-1.0", "Unlicense"}
)
_MARKERS = {
    "MIT": ("permission is hereby granted, free of charge",),
    "Apache-2.0": ("apache license", "version 2.0"),
    "ISC": ("permission to use, copy, modify, and/or distribute",),
    "Unlicense": ("this is free and unencumbered software",),
    "CC0-1.0": ("cc0 1.0 universal",),
}


@dataclass(frozen=True)
class LicenseDecision:
    spdx: str
    status: str
    evidence: str


def detect_license(
    repository: Path, accepted: frozenset[str] = DEFAULT_ACCEPTED
) -> LicenseDecision:
    candidates = sorted(
        p
        for p in repository.iterdir()
        if p.is_file() and not p.is_symlink() and p.name.lower().startswith(("license", "copying"))
    )
    if not candidates:
        return LicenseDecision("NOASSERTION", "review", "no-license-file")
    detected: list[tuple[str, str]] = []
    for candidate in candidates:
        with candidate.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(100_001).lower()
        spdx_match = re.search(r"spdx-license-identifier:\s*([A-Za-z0-9.+-]+)", text, re.I)
        spdx = spdx_match.group(1) if spdx_match else "NOASSERTION"
        if spdx == "NOASSERTION":
            for name, markers in _MARKERS.items():
                if all(marker in text for marker in markers):
                    spdx = name
                    break
        detected.append((spdx, candidate.name))
    names = {item[0] for item in detected}
    if len(names) != 1:
        return LicenseDecision("CONFLICTING", "review", ",".join(item[1] for item in detected))
    spdx, evidence = detected[0]
    return LicenseDecision(spdx, "accepted" if spdx in accepted else "review", evidence)
