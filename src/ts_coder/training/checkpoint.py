from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

_REQUIRED_PAYLOAD_KEYS = frozenset({"model", "optimizer", "scheduler", "scaler", "metadata", "rng"})
_MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024 * 1024
_MAX_TENSOR_ELEMENTS = 2_000_000_000
_MAX_TOTAL_TENSOR_ELEMENTS = 4_500_000_000
_MAX_CONTAINER_ITEMS = 1_000_000


@dataclass(frozen=True)
class CheckpointMetadata:
    global_step: int
    tokens_processed: int
    data_cursor: int
    resolved_config: dict[str, Any]
    tokenizer_hash: str
    manifest_hash: str
    git_commit: str
    format_version: int = 1
    # Optional scale-out fields.  They are omitted from the v1 envelope unless
    # provided, preserving compatibility with Milestone-1 checkpoints while
    # giving a distributed loader a typed place to persist its cursor.
    data_position: dict[str, int] | None = None
    shard_manifest_hash: str | None = None
    parallel: dict[str, int] | None = None


def validate_checkpoint_payload(payload: Any, path: str | Path | None = None) -> None:
    """Validate the tensor-only checkpoint envelope before applying state.

    ``weights_only=True`` prevents pickle object execution, but it does not
    make an untrusted artifact honest or small.  This schema and size guard
    rejects unknown envelopes and pathological tensor/container sizes before
    they reach a model or optimizer.  The artifact must still come from a
    trusted run directory; this is not a provenance or signature check.
    """
    if path is not None:
        size = Path(path).stat().st_size
        if size > _MAX_CHECKPOINT_BYTES:
            raise ValueError(f"checkpoint exceeds {_MAX_CHECKPOINT_BYTES} bytes")
    if not isinstance(payload, dict) or set(payload) != _REQUIRED_PAYLOAD_KEYS:
        raise ValueError("checkpoint has an unsupported or incomplete envelope")

    total_elements = 0

    def walk(value: Any, location: str, depth: int = 0) -> None:
        nonlocal total_elements
        if depth > 64:
            raise ValueError(f"checkpoint nesting is too deep at {location}")
        if isinstance(value, torch.Tensor):
            elements = int(value.numel())
            if elements > _MAX_TENSOR_ELEMENTS:
                raise ValueError(f"checkpoint tensor is too large at {location}")
            total_elements += elements
            if total_elements > _MAX_TOTAL_TENSOR_ELEMENTS:
                raise ValueError("checkpoint contains too many tensor elements")
            return
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, dict):
            if len(value) > _MAX_CONTAINER_ITEMS:
                raise ValueError(f"checkpoint mapping is too large at {location}")
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool)):
                    raise ValueError(f"checkpoint key type is unsupported at {location}")
                walk(item, f"{location}.{key}", depth + 1)
            return
        if isinstance(value, (list, tuple)):
            if len(value) > _MAX_CONTAINER_ITEMS:
                raise ValueError(f"checkpoint sequence is too large at {location}")
            for index, item in enumerate(value):
                walk(item, f"{location}[{index}]", depth + 1)
            return
        raise ValueError(f"checkpoint value type is unsupported at {location}")

    walk(payload, "root")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or metadata.get("format_version") not in {1, 2}:
        raise ValueError("unsupported checkpoint format_version")
    required_metadata = {
        "global_step",
        "tokens_processed",
        "data_cursor",
        "resolved_config",
        "tokenizer_hash",
        "manifest_hash",
        "git_commit",
        "format_version",
    }
    optional_metadata = {"data_position", "shard_manifest_hash", "parallel"}
    if set(metadata) - required_metadata - optional_metadata:
        raise ValueError("checkpoint metadata has unknown fields")
    if not required_metadata.issubset(metadata):
        raise ValueError("checkpoint metadata schema is incomplete or has unknown fields")
    if not isinstance(metadata["resolved_config"], dict):
        raise ValueError("checkpoint resolved_config must be a mapping")
    for field in ("global_step", "tokens_processed", "data_cursor"):
        invalid_counter = (
            not isinstance(metadata[field], int)
            or isinstance(metadata[field], bool)
            or metadata[field] < 0
        )
        if invalid_counter:
            raise ValueError(f"checkpoint metadata field {field} is invalid")
    for field in ("tokenizer_hash", "manifest_hash", "git_commit"):
        if not isinstance(metadata[field], str) or len(metadata[field]) > 256:
            raise ValueError(f"checkpoint metadata field {field} is invalid")
    if "shard_manifest_hash" in metadata and (
        not isinstance(metadata["shard_manifest_hash"], str)
        or len(metadata["shard_manifest_hash"]) > 256
    ):
        raise ValueError("checkpoint shard_manifest_hash is invalid")
    if "data_position" in metadata:
        from ts_coder.data.streaming import DataCursor

        if not isinstance(metadata["data_position"], dict):
            raise ValueError("checkpoint data_position must be a mapping")
        DataCursor.from_mapping(metadata["data_position"])
    if "parallel" in metadata:
        parallel = metadata["parallel"]
        if not isinstance(parallel, dict) or set(parallel) != {
            "world_size",
            "rank",
            "local_rank",
        }:
            raise ValueError("checkpoint parallel metadata is incomplete")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in parallel.values()
        ):
            raise ValueError("checkpoint parallel metadata is invalid")
        if parallel["world_size"] < 1 or parallel["rank"] >= parallel["world_size"]:
            raise ValueError("checkpoint parallel rank/world_size is invalid")
    if not isinstance(payload["model"], dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload["model"].items()
    ):
        raise ValueError("checkpoint model state must map string names to tensors")
    if not isinstance(payload["optimizer"], dict) or not {
        "state",
        "param_groups",
    }.issubset(payload["optimizer"]):
        raise ValueError("checkpoint optimizer state is incomplete")
    if payload["scheduler"] is not None and not isinstance(payload["scheduler"], dict):
        raise ValueError("checkpoint scheduler state is invalid")
    if payload["scaler"] is not None and not isinstance(payload["scaler"], dict):
        raise ValueError("checkpoint scaler state is invalid")
    if not isinstance(payload["rng"], dict) or not {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }.issubset(payload["rng"]):
        raise ValueError("checkpoint RNG state is incomplete")


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Load and validate a tensor-only checkpoint before exposing its state."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("checkpoint path must be a regular file")
    if checkpoint_path.stat().st_size > _MAX_CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint exceeds {_MAX_CHECKPOINT_BYTES} bytes")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_payload(payload, checkpoint_path)
    return payload


def _rng_state() -> dict[str, Any]:
    algorithm, state, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        # Keep NumPy RNG state tensor/primitive-only so weights_only loading is safe.
        "numpy": {
            "algorithm": algorithm,
            "state": np.asarray(state, dtype=np.uint32).tolist(),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    metadata: CheckpointMetadata,
    scaler: Any | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    metadata_payload = asdict(metadata)
    # Do not add optional scale-out keys to legacy v1 checkpoints unless the
    # caller explicitly supplied them.
    for optional in ("data_position", "shard_manifest_hash", "parallel"):
        if metadata_payload[optional] is None:
            metadata_payload.pop(optional)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metadata": metadata_payload,
        "rng": _rng_state(),
    }
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: Any | None = None,
    restore_rng: bool = True,
) -> CheckpointMetadata:
    # Full-state checkpoints are serialized with primitive/tensor values only.
    # weights_only=True is the deserialization boundary for caller-supplied paths.
    checkpoint_path = Path(path)
    payload = load_checkpoint_payload(checkpoint_path)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload["scaler"] is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        random.setstate(payload["rng"]["python"])
        numpy_state = payload["rng"]["numpy"]
        np.random.set_state(
            (
                numpy_state["algorithm"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(payload["rng"]["torch_cpu"])
        if torch.cuda.is_available() and payload["rng"]["torch_cuda"]:
            torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
    return CheckpointMetadata(**payload["metadata"])
