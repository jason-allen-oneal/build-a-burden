from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ts_coder.cli import main
from ts_coder.evaluation.quarantine import verify_quarantine


def _fixture_tree(tmp_path: Path) -> tuple[Path, str]:
    fixture = tmp_path / "fixtures" / "completion.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks": [
                    {
                        "id": "held-out",
                        "prompt": "export function heldOut(value: number): number {\n",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    manifest = tmp_path / "quarantine.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "held-out-evaluation",
                "fixtures": [
                    {
                        "id": "completion-v1",
                        "kind": "completion",
                        "path": "fixtures/completion.json",
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, digest


def test_quarantine_verifies_pinned_fixture(tmp_path: Path) -> None:
    manifest, _ = _fixture_tree(tmp_path)

    report = verify_quarantine(manifest, project_root=tmp_path, corpus={"safe": "const x = 1;"})

    assert report["schema_version"] == 1
    assert report["fixtures_verified"] == 1
    assert report["tasks_verified"] == 1
    assert report["fragments_verified"] == 1
    assert report["corpus_records_checked"] == 1
    assert report["contamination_matches"] == 0
    assert report["fixture_sha256"]["completion-v1"]
    assert report["manifest_sha256"]


def test_quarantine_rejects_fixture_hash_drift(tmp_path: Path) -> None:
    manifest, _ = _fixture_tree(tmp_path)
    fixture = tmp_path / "fixtures" / "completion.json"
    fixture.write_text(fixture.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_quarantine(manifest, project_root=tmp_path)


def test_quarantine_rejects_normalized_prompt_in_corpus(tmp_path: Path) -> None:
    manifest, _ = _fixture_tree(tmp_path)
    corpus = {
        "leaked": (
            "// formatting differs\n"
            "export   function heldOut(value: number): number {\n"
            "  return value;\n}"
        )
    }

    with pytest.raises(ValueError, match="contamination detected"):
        verify_quarantine(manifest, project_root=tmp_path, corpus=corpus)


def test_quarantine_rejects_minified_prompt_tokens(tmp_path: Path) -> None:
    manifest, _ = _fixture_tree(tmp_path)
    corpus = {"leaked": "export function heldOut(value:number):number{return value;}"}

    with pytest.raises(ValueError, match="contamination detected"):
        verify_quarantine(manifest, project_root=tmp_path, corpus=corpus)


def test_repository_quarantine_rejects_isolated_fim_middle() -> None:
    corpus = {"leaked": "return Math.max(minimum, Math.min(value, maximum));"}

    with pytest.raises(ValueError, match="contamination detected"):
        verify_quarantine(
            Path("manifests/evaluation-quarantine.json"),
            project_root=Path.cwd(),
            corpus=corpus,
        )


def test_quarantine_cli_verifies_repository_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evaluation-quarantine", "verify"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["fixtures_verified"] == 2
    assert report["tasks_verified"] == 10
    assert report["fragments_verified"] == 19
