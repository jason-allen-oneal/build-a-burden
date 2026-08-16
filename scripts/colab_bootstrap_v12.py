"""Safely extract the filtered ts-coder inputs in a Colab runtime."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ARCHIVE = Path("/content/ts-coder-a100-inputs-v12.tar.gz")
TARGET = Path("/content/ts-coder")
# Replaced by the packaging command before upload.
EXPECTED_SHA256 = "f8656ed8be3fe51cf7e9b8afb30a9ea51b3ed1719190146e5246039a3476f0a9"


def safe_members(tf: tarfile.TarFile, root: Path) -> list[tarfile.TarInfo]:
    resolved_root = root.resolve()
    members: list[tarfile.TarInfo] = []
    for member in tf.getmembers():
        name = Path(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise RuntimeError(f"unsafe archive member: {member.name}")
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise RuntimeError(f"unsupported archive member: {member.name}")
        destination = (root / name).resolve()
        if destination != resolved_root and resolved_root not in destination.parents:
            raise RuntimeError(f"archive escape: {member.name}")
        members.append(member)
    return members


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"archive hash mismatch: {digest}")
    shutil.rmtree(TARGET, ignore_errors=True)
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        members = safe_members(tf, TARGET)
        TARGET.mkdir(parents=True, exist_ok=True)
        tf.extractall(TARGET, members=members)  # noqa: S202 - members are path-validated above
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "tokenizers==0.21.1", "PyYAML==6.0.2"]
    )
    print({"archive_sha256": digest, "extracted": str(TARGET), "members": len(members)})


if __name__ == "__main__":
    main()
