from pathlib import Path
import shutil

import pytest

from ts_coder.cli import main
from ts_coder.corpus.approval import ApprovalManifestError, append_approval, make_approval


def _config(path: Path) -> None:
    path.write_text(
        """schema_version: 1
seed: 42
sources:
  - path: repositories
    source_id: basic
    source_uri: fixture://basic
    commit_sha: fixture
approval_manifest: approvals.jsonl
require_approval: true
accepted_licenses: [MIT]
output_manifest: manifests/out.jsonl
output_corpus: artifacts/corpus/out
append_manifest: true
max_file_bytes: 262144
max_files: 1000
max_total_bytes: 10000000
max_path_depth: 32
splits: {train: 0.90, validation: 0.05, test: 0.05}
near_duplicate_threshold: 0.90
""",
        encoding="utf-8",
    )


def test_approved_dry_run_requires_exact_ledger_entry(tmp_path: Path, monkeypatch) -> None:
    fixture = Path(__file__).parents[2] / "fixtures" / "repositories" / "basic"
    shutil.copytree(fixture, tmp_path / "repositories" / "basic")
    approval_path = tmp_path / "approvals.jsonl"
    append_approval(
        approval_path,
        make_approval(
            source_id="basic",
            source_uri="fixture://basic",
            commit_sha="fixture",
            license_spdx="MIT",
            approved_by="fixture-reviewer",
            approved_at="2026-08-14T00:00:00Z",
            scope=["typescript-source"],
        ),
    )
    config = tmp_path / "approved.yaml"
    _config(config)
    monkeypatch.chdir(tmp_path)
    assert main(["corpus", "build", "--config", str(config), "--dry-run"]) == 0
    assert not (tmp_path / "manifests" / "out.jsonl").exists()

    approval_path.write_text("\n", encoding="utf-8")
    with pytest.raises(ApprovalManifestError, match="not approved"):
        main(["corpus", "build", "--config", str(config), "--dry-run"])


def test_approval_add_cli_appends_a_valid_scoped_decision(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "corpus",
                "approvals",
                "add",
                "--manifest",
                "approvals.jsonl",
                "--source-id",
                "example-project",
                "--source-uri",
                "https://github.com/example/project",
                "--commit-sha",
                "a" * 40,
                "--license-spdx",
                "MIT",
                "--approved-by",
                "reviewer",
                "--approved-at",
                "2026-08-14T00:00:00Z",
                "--scope",
                "include:src/**/*.ts",
            ]
        )
        == 0
    )
    assert len((tmp_path / "approvals.jsonl").read_text(encoding="utf-8").splitlines()) == 1
