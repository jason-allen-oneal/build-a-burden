from pathlib import Path

import pytest

from ts_coder.corpus.ingest import ingest_repository, sanitize_source_uri


def test_source_uri_redacts_credentials_and_query():
    assert (
        sanitize_source_uri("https://user:secret@example.test/repo?token=abc")
        == "https://example.test/repo"
    )
    assert sanitize_source_uri("file:/tmp/repo?token=abc#fragment") == "file:///tmp/repo"
    assert sanitize_source_uri("urn:repo?signature=abc") == "urn:repo"


def test_oversized_and_symlink_inputs_are_recorded_as_excluded(tmp_path: Path):
    (tmp_path / "LICENSE").write_text(
        "MIT License\nPermission is hereby granted, free of charge", encoding="utf-8"
    )
    (tmp_path / "large.ts").write_text("const x = 1;\n" * 20, encoding="utf-8")
    (tmp_path / "link.ts").symlink_to("large.ts")
    records, _ = ingest_repository(tmp_path, max_file_bytes=16)
    by_name = {record["relative_path"]: record for record in records}
    assert "oversized" in by_name["large.ts"]["exclusion_reasons"]
    assert "symlink" in by_name["link.ts"]["exclusion_reasons"]


def test_file_count_limit_fails_closed(tmp_path: Path):
    (tmp_path / "LICENSE").write_text("MIT License", encoding="utf-8")
    (tmp_path / "a.ts").write_text("export const a = 1", encoding="utf-8")
    with pytest.raises(ValueError, match="max_files"):
        ingest_repository(tmp_path, max_files=1)


def test_license_policy_cannot_be_weakened(tmp_path: Path):
    (tmp_path / "LICENSE").write_text("MIT License", encoding="utf-8")
    (tmp_path / "source.ts").write_text("export const value = 1;", encoding="utf-8")
    with pytest.raises(ValueError, match="policy-disallowed"):
        ingest_repository(tmp_path, accepted_licenses=frozenset({"GPL-3.0-only"}))
