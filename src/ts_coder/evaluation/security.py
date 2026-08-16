"""Static checks for generated evaluation artifacts and manifests."""

from __future__ import annotations

from pathlib import Path

from ..corpus.secret_filter import scan_secrets


def contains_secret(text: str) -> bool:
    """Return whether the same scanner used during ingestion rejects *text*.

    Keeping evaluation and corpus filtering on one implementation prevents a
    generated artifact from receiving a clean evaluation result merely because
    the two paths recognize different credential patterns.
    """
    status, _ = scan_secrets(Path("generated.ts"), text)
    return status == "rejected"
