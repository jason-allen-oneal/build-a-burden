from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from ts_coder.corpus import acquire
from ts_coder.corpus.acquire import AcquisitionError, acquire_git_source, verify_snapshot_receipt


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _fixture_bare(
    tmp_path: Path,
    files: dict[str, str] | None = None,
    symlinks: dict[str, str] | None = None,
) -> tuple[Path, str]:
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init")
    _git(work, "config", "user.name", "Fixture")
    _git(work, "config", "user.email", "fixture@example.invalid")
    for name, content in (
        files or {"LICENSE": "MIT License\n", "src/index.ts": "export const x = 1;\n"}
    ).items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for name, target in (symlinks or {}).items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    _git(work, "add", ".")
    _git(work, "commit", "-m", "fixture")
    commit = _git(work, "rev-parse", "HEAD")
    bare = tmp_path / "fixture.git"
    subprocess.run(
        ["git", "clone", "--bare", str(work), str(bare)], check=True, capture_output=True
    )
    return bare, commit


def _local_fetch(source: Path, commit: str):
    def fetch(
        destination: Path,
        source_uri: str,
        commit_sha: str,
        timeout: int,
        max_file_bytes: int,
        max_total_bytes: int,
    ) -> None:
        assert source_uri == "https://github.com/example/project"
        assert commit_sha == commit
        shutil.copytree(source, destination)
        (destination / "FETCH_HEAD").write_text(commit + "\n", encoding="ascii")

    return fetch


def _acquire(tmp_path: Path, commit: str, **overrides):
    values = {
        "source_id": "example-project",
        "source_uri": "https://github.com/example/project",
        "commit_sha": commit,
        "destination_root": tmp_path / "sources",
        "max_file_bytes": 1_000,
        "max_files": 20,
        "max_total_bytes": 10_000,
        "max_path_depth": 8,
    }
    values.update(overrides)
    return acquire_git_source(**values)


def test_acquires_verified_blobs_and_writes_deterministic_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    bare, commit = _fixture_bare(tmp_path)
    monkeypatch.setattr(acquire, "_fetch_commit", _local_fetch(bare, commit))

    result = _acquire(tmp_path, commit)
    snapshot = Path(result["snapshot_path"])
    receipt = Path(result["receipt_path"])
    assert (snapshot / "src/index.ts").read_text(encoding="utf-8") == "export const x = 1;\n"
    assert receipt == snapshot.parent / f"{commit}.receipt.json"
    assert result["file_count"] == 2
    assert result["license_sha256"] == {"LICENSE": hashlib.sha256(b"MIT License\n").hexdigest()}
    assert (
        verify_snapshot_receipt(snapshot, receipt, result["source_uri"], commit)["tree_sha256"]
        == result["tree_sha256"]
    )

    again = _acquire(tmp_path, commit)
    assert again["tree_sha256"] == result["tree_sha256"]
    assert json.loads(receipt.read_text(encoding="utf-8"))["commit_sha"] == commit


def test_snapshot_mutation_and_partial_existing_state_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    bare, commit = _fixture_bare(tmp_path)
    monkeypatch.setattr(acquire, "_fetch_commit", _local_fetch(bare, commit))
    result = _acquire(tmp_path, commit)
    Path(result["snapshot_path"], "src/index.ts").write_text("changed", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="does not match"):
        _acquire(tmp_path, commit)

    other = "a" * 40
    partial = tmp_path / "sources" / "other" / other
    partial.mkdir(parents=True)
    with pytest.raises(AcquisitionError, match="partial pre-existing"):
        _acquire(tmp_path, other, source_id="other")


def test_internal_symlinks_are_materialized_and_fingerprinted(tmp_path: Path, monkeypatch) -> None:
    bare, commit = _fixture_bare(
        tmp_path,
        files={"LICENSE": "MIT License\n", "src/target.ts": "export {};\n"},
        symlinks={"CURRENT.ts": "src/target.ts", "src/UP.ts": "../CURRENT.ts"},
    )
    monkeypatch.setattr(acquire, "_fetch_commit", _local_fetch(bare, commit))
    result = _acquire(tmp_path, commit)
    snapshot = Path(result["snapshot_path"])
    assert (snapshot / "CURRENT.ts").is_symlink()
    assert (snapshot / "CURRENT.ts").readlink().as_posix() == "src/target.ts"
    assert result["file_count"] == 4
    original_hash = result["tree_sha256"]

    (snapshot / "CURRENT.ts").unlink()
    (snapshot / "CURRENT.ts").symlink_to("LICENSE")
    with pytest.raises(AcquisitionError, match="does not match"):
        verify_snapshot_receipt(snapshot, result["receipt_path"], result["source_uri"], commit)
    assert original_hash != acquire._snapshot_fingerprint(snapshot)[0]


def test_dry_run_validates_without_network_or_writes(tmp_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted network acquisition")

    monkeypatch.setattr(acquire, "_fetch_commit", forbidden)
    result = _acquire(tmp_path, "a" * 40, dry_run=True)
    assert result["dry_run"] is True
    assert not (tmp_path / "sources").exists()


def test_fetch_uses_blob_filter_and_os_file_size_quota(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_git(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return b""

    monkeypatch.setattr(acquire, "_run_git", fake_git)
    acquire._fetch_commit(
        tmp_path / "bare",
        "https://github.com/example/project",
        "a" * 40,
        10,
        1_000,
        10_000,
    )
    fetch_args, fetch_kwargs = calls[-1]
    assert "--filter=blob:limit=1001" in fetch_args
    assert fetch_kwargs["file_size_limit"] == acquire._fetch_quota(10_000)
    assert fetch_kwargs["file_size_limit"] > 10_000


def test_fetch_quota_failure_publishes_nothing(tmp_path: Path, monkeypatch) -> None:
    def quota_failure(*args, **kwargs):
        raise AcquisitionError("Git operation failed")

    monkeypatch.setattr(acquire, "_fetch_commit", quota_failure)
    commit = "a" * 40
    with pytest.raises(AcquisitionError, match="Git operation failed"):
        _acquire(tmp_path, commit)
    source_root = tmp_path / "sources" / "example-project"
    assert not (source_root / commit).exists()
    assert not (source_root / f"{commit}.receipt.json").exists()
    assert not list(source_root.glob(".example-project-*"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.update({"unknown": True}),
        lambda receipt: receipt.pop("git_version"),
        lambda receipt: receipt.update({"schema_version": 2}),
        lambda receipt: receipt.update({"source_id": "wrong"}),
        lambda receipt: receipt.update({"retrieved_at": "not-a-timestamp"}),
        lambda receipt: receipt.update({"file_count": 1_000_001}),
        lambda receipt: receipt.update({"total_bytes": 10 * 1024 * 1024 * 1024 + 1}),
    ],
)
def test_receipt_schema_and_hard_bounds_fail_closed(tmp_path: Path, monkeypatch, mutation) -> None:
    bare, commit = _fixture_bare(tmp_path)
    monkeypatch.setattr(acquire, "_fetch_commit", _local_fetch(bare, commit))
    result = _acquire(tmp_path, commit)
    receipt_path = Path(result["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AcquisitionError):
        verify_snapshot_receipt(result["snapshot_path"], receipt_path, result["source_uri"], commit)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"source_uri": "http://github.com/example/project"}, "canonical HTTPS"),
        ({"source_uri": "https://github.com/example/project.git"}, "not canonical"),
        ({"commit_sha": "A" * 40}, "lowercase"),
        ({"source_id": "../escape"}, "source_id"),
        ({"max_files": 0}, "positive"),
    ],
)
def test_rejects_invalid_source_specs(tmp_path: Path, override: dict, message: str) -> None:
    with pytest.raises(AcquisitionError, match=message):
        _acquire(tmp_path, "a" * 40, dry_run=True, **override)


@pytest.mark.parametrize(
    ("limit", "value", "message"),
    [
        ("max_file_bytes", 5, "max_file_bytes"),
        ("max_files", 1, "max_files"),
        ("max_total_bytes", 10, "max_total_bytes"),
        ("max_path_depth", 1, "over-depth"),
    ],
)
def test_materialization_limits_fail_without_publishing(
    tmp_path: Path, monkeypatch, limit: str, value: int, message: str
) -> None:
    bare, commit = _fixture_bare(tmp_path)
    monkeypatch.setattr(acquire, "_fetch_commit", _local_fetch(bare, commit))
    with pytest.raises(AcquisitionError, match=message):
        _acquire(tmp_path, commit, **{limit: value})
    assert not (tmp_path / "sources" / "example-project" / commit).exists()
