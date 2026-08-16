"""Run metadata and deterministic seed helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess  # nosec B404 - invokes only local Git for metadata
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata() -> dict[str, Any]:
    gpu = []
    if torch.cuda.is_available():
        gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    lockfiles = [Path("uv.lock"), Path("tools/typescript/package-lock.json")]
    lock_digest = hashlib.sha256()
    lockfile_names = []
    for lockfile in lockfiles:
        if lockfile.exists():
            lockfile_names.append(str(lockfile))
            lock_digest.update(lockfile.read_bytes())
    try:
        ram_bytes = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        ram_bytes = None
    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "cuda": torch.version.cuda,
        "ram_bytes": ram_bytes,
        "dependency_lockfiles": lockfile_names,
        "dependency_lock_hash": lock_digest.hexdigest() if lockfile_names else "unavailable",
        "seed_source": "explicit",
        "started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def git_metadata(root: str | Path) -> dict[str, Any]:
    root = str(root)

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(  # nosec B603 B607 - fixed local Git executable
                ["git", "-C", root, *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status == "unavailable" else bool(status),
        "repository_available": commit != "unavailable",
    }


def create_run_dir(root: str | Path, run_id: str, config: dict[str, Any], seed: int) -> Path:
    run = Path(root) / run_id
    (run / "samples").mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    (run / "resolved-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run / "resolved-config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    (run / "environment.json").write_text(
        json.dumps({**environment_metadata(), "seed": seed}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run / "git.json").write_text(
        json.dumps(git_metadata(Path.cwd()), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run
