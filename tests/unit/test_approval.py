from pathlib import Path

import pytest

from ts_coder.corpus.approval import (
    ApprovalManifestError,
    append_approval,
    load_approval_manifest,
    make_approval,
    require_approved_source,
)


def test_exact_revision_approval_is_required_and_append_only(tmp_path: Path) -> None:
    manifest = tmp_path / "approved.jsonl"
    approved = make_approval(
        source_id="example",
        source_uri="https://example.test/repo",
        commit_sha="a" * 40,
        license_spdx="MIT",
        approved_by="reviewer",
        approved_at="2026-08-14T00:00:00Z",
        scope=["typescript-source"],
    )
    append_approval(manifest, approved)
    assert (
        require_approved_source(
            load_approval_manifest(manifest), "example", "https://example.test/repo", "a" * 40
        )
        == approved
    )
    with pytest.raises(ApprovalManifestError, match="exact URI/revision"):
        require_approved_source(
            load_approval_manifest(manifest), "example", "https://example.test/repo", "b" * 40
        )

    removed = make_approval(
        source_id="example",
        source_uri="https://example.test/repo",
        commit_sha="a" * 40,
        license_spdx="MIT",
        approved_by="reviewer",
        approved_at="2026-08-15T00:00:00Z",
        scope=["typescript-source"],
        status="removed",
        notes="withdrawn for contamination review",
        supersedes=approved.approval_id,
    )
    append_approval(manifest, removed)
    with pytest.raises(ApprovalManifestError, match="removed"):
        require_approved_source(
            load_approval_manifest(manifest), "example", "https://example.test/repo", "a" * 40
        )


def test_approval_rejects_unsafe_uri_and_license() -> None:
    with pytest.raises(ApprovalManifestError, match="source_uri"):
        make_approval(
            source_id="example",
            source_uri="https://user:password@example.test/repo",
            commit_sha="a" * 40,
            license_spdx="MIT",
            approved_by="reviewer",
            approved_at="2026-08-14T00:00:00Z",
            scope=["typescript-source"],
        )
    with pytest.raises(ApprovalManifestError, match="allowlist"):
        make_approval(
            source_id="example",
            source_uri="https://example.test/repo",
            commit_sha="a" * 40,
            license_spdx="GPL-3.0-only",
            approved_by="reviewer",
            approved_at="2026-08-14T00:00:00Z",
            scope=["typescript-source"],
        )


def test_empty_approval_ledger_validates(tmp_path: Path) -> None:
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("\n", encoding="utf-8")
    assert load_approval_manifest(manifest) == []


def test_approval_rejects_unsafe_or_ambiguous_scope() -> None:
    base = {
        "source_id": "example",
        "source_uri": "https://example.test/repo",
        "commit_sha": "a" * 40,
        "license_spdx": "MIT",
        "approved_by": "reviewer",
        "approved_at": "2026-08-14T00:00:00Z",
    }
    with pytest.raises(ApprovalManifestError, match="scope entries"):
        make_approval(scope=["packages/**"], **base)
    with pytest.raises(ApprovalManifestError, match="unsafe path"):
        make_approval(scope=["include:../escape.ts"], **base)
