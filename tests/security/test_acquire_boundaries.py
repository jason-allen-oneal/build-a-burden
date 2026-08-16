import os
from pathlib import Path

import pytest

from ts_coder.corpus import acquire
from ts_coder.corpus.acquire import AcquisitionError, acquire_git_source


@pytest.mark.parametrize(
    "uri",
    [
        "file:///tmp/repository",
        "ssh://git@github.com/example/project",
        "https://user:secret@github.com/example/project",
        "https://github.com/example/project?token=value",
        "https://github.com/example/project#main",
        "https://evil.example/example/project",
        "https://github.com/example/project/extra",
        "https://github.com/-c/project",
        "https://github.com/example/project%0Ainject",
    ],
)
def test_noncanonical_or_option_like_source_uris_are_rejected(tmp_path: Path, uri: str) -> None:
    with pytest.raises(AcquisitionError):
        acquire_git_source(
            source_id="example",
            source_uri=uri,
            commit_sha="a" * 40,
            destination_root=tmp_path / "sources",
            max_file_bytes=100,
            max_files=10,
            max_total_bytes=1_000,
            max_path_depth=5,
            dry_run=True,
        )


@pytest.mark.parametrize(
    "path",
    [
        b"../escape.ts",
        b"/absolute.ts",
        b"safe/../../escape.ts",
        b"windows\\escape.ts",
        b"a\x00b.ts",
    ],
)
def test_unsafe_tree_paths_are_rejected(path: bytes) -> None:
    with pytest.raises(AcquisitionError, match="path|NUL"):
        acquire._safe_tree_path(path, 8)


@pytest.mark.parametrize(
    ("mode", "kind", "size"),
    [("160000", "commit", "-"), ("100664", "blob", "4")],
)
def test_submodule_and_special_modes_are_rejected(
    tmp_path: Path, monkeypatch, mode: str, kind: str, size: str
) -> None:
    object_id = "a" * 40
    tree = f"{mode} {kind} {object_id} {size}\tunsafe\0".encode()

    def fake_git(arguments, **kwargs):
        assert "ls-tree" in arguments
        return tree

    monkeypatch.setattr(acquire, "_run_git", fake_git)
    with pytest.raises(AcquisitionError, match="submodule or special"):
        acquire._tree_entries(tmp_path, "b" * 40, 1, 10)


def test_receipt_symlink_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "receipt.json"
    link.symlink_to(real)
    with pytest.raises(AcquisitionError, match="regular existing"):
        acquire.verify_snapshot_receipt(
            snapshot, link, "https://github.com/example/project", "a" * 40
        )


def test_oversized_receipt_is_rejected_before_parsing(tmp_path: Path) -> None:
    commit = "a" * 40
    source = tmp_path / "example"
    snapshot = source / commit
    snapshot.mkdir(parents=True)
    receipt = source / f"{commit}.receipt.json"
    receipt.write_bytes(b" " * (acquire._MAX_RECEIPT_BYTES + 1))
    with pytest.raises(AcquisitionError, match="size limit"):
        acquire.verify_snapshot_receipt(
            snapshot, receipt, "https://github.com/example/project", commit
        )


def test_snapshot_symlinked_directory_escape_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (snapshot / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(AcquisitionError, match="relative internal|escapes"):
        acquire._snapshot_fingerprint(snapshot)


@pytest.mark.parametrize(
    "target",
    [b"/etc/passwd", b"../../escape", b"..\\escape", b"bad\x00target", b"\xff"],
)
def test_unsafe_symlink_targets_are_rejected(target: bytes) -> None:
    with pytest.raises(AcquisitionError, match="symlink"):
        acquire._internal_symlink_target(acquire.PurePosixPath("nested/link"), target)


def test_dangling_internal_symlink_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "link").symlink_to("missing.ts")
    with pytest.raises(AcquisitionError, match="dangling"):
        acquire._snapshot_fingerprint(snapshot)


def test_non_utf8_materialized_symlink_target_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    os.symlink(b"\xff", os.fsencode(snapshot / "link"))
    with pytest.raises(AcquisitionError, match="UTF-8"):
        acquire._snapshot_fingerprint(snapshot)


def test_inherited_git_configuration_is_removed(monkeypatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/hostile")
    environment = acquire._git_env()
    assert "GIT_CONFIG_COUNT" not in environment
    assert "GIT_CONFIG_KEY_0" not in environment
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
