"""Fail-closed loading for checkpoints, tokenizers, and model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .data.streaming import ShardManifest, StreamingShardDataset
from .data.token_stream import shard_manifest_hash
from .model.config import ModelConfig
from .model.transformer import Transformer
from .reproducibility import sha256_file
from .tokenizer.special_tokens import SPECIAL_TOKENS
from .tokenizer.trainer import load_tokenizer
from .training.checkpoint import load_checkpoint_payload


@dataclass(frozen=True)
class LoadedModelBundle:
    """A checkpoint and its verified tokenizer/model identity."""

    checkpoint_path: Path
    tokenizer_path: Path
    payload: dict[str, Any]
    metadata: dict[str, Any]
    model: Transformer
    tokenizer: Any
    device: torch.device

    @property
    def resolved_config(self) -> dict[str, Any]:
        value = self.metadata.get("resolved_config", {})
        return value if isinstance(value, dict) else {}

    @property
    def training_config(self) -> dict[str, Any]:
        value = self.resolved_config.get("training", {})
        return value if isinstance(value, dict) else {}


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve a requested runtime device and reject unavailable CUDA."""

    try:
        device = torch.device(requested or "cpu")
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid runtime device: {requested!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def _model_config(metadata: Mapping[str, Any]) -> ModelConfig:
    resolved = metadata.get("resolved_config", {})
    if not isinstance(resolved, Mapping):
        raise ValueError("checkpoint resolved_config must be a mapping")
    raw = resolved.get("model", resolved)
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint model configuration must be a mapping")
    runtime_fields = set(ModelConfig.__dataclass_fields__)
    unknown = set(raw) - runtime_fields - {"schema_version", "training_tokens"}
    if unknown:
        raise ValueError(f"checkpoint model configuration has unknown fields: {sorted(unknown)}")
    supplied = {
        key: value
        for key, value in raw.items()
        if key != "schema_version" and key in runtime_fields
    }
    required = {
        "vocab_size",
        "context_length",
        "layers",
        "hidden_size",
        "attention_heads",
        "kv_heads",
        "ffn_size",
    }
    missing = sorted(required - supplied.keys())
    if missing:
        raise ValueError(f"checkpoint model configuration is missing fields: {missing}")
    return ModelConfig(**supplied)


def _default_tokenizer_path(checkpoint_path: Path) -> Path:
    candidates = [
        checkpoint_path.parent.parent / "tokenizer.json",
        checkpoint_path.parent / "tokenizer.json",
    ]
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return candidates[0]


def _verify_tokenizer(tokenizer_path: Path, metadata: Mapping[str, Any], model: Transformer) -> Any:
    if tokenizer_path.is_symlink() or not tokenizer_path.is_file():
        raise ValueError(f"tokenizer path must be a regular file: {tokenizer_path}")
    expected_hash = metadata.get("tokenizer_hash")
    if not isinstance(expected_hash, str) or not expected_hash or expected_hash == "unavailable":
        raise ValueError("checkpoint does not contain a usable tokenizer hash")
    actual_hash = sha256_file(tokenizer_path)
    if actual_hash != expected_hash:
        raise ValueError("tokenizer hash does not match checkpoint metadata")
    tokenizer = load_tokenizer(tokenizer_path)
    vocabulary_size = int(tokenizer.get_vocab_size())
    if vocabulary_size != model.config.vocab_size:
        raise ValueError(
            "tokenizer/model vocabulary mismatch: "
            f"tokenizer={vocabulary_size}, model={model.config.vocab_size}"
        )
    invalid_special_ids = {
        token: tokenizer.token_to_id(token)
        for expected_id, token in enumerate(SPECIAL_TOKENS)
        if tokenizer.token_to_id(token) != expected_id
    }
    if invalid_special_ids:
        raise ValueError(
            f"tokenizer special-token IDs do not match the canonical order: {invalid_special_ids}"
        )
    return tokenizer


def load_model_bundle(
    checkpoint: str | Path,
    *,
    tokenizer: str | Path | None = None,
    device: str | torch.device | None = None,
) -> LoadedModelBundle:
    """Load a model only when checkpoint and tokenizer identities agree."""

    checkpoint_path = Path(checkpoint)
    payload = load_checkpoint_payload(checkpoint_path)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata is missing")
    model = Transformer(_model_config(metadata))
    model.load_state_dict(payload["model"])
    runtime_device = resolve_device(device)
    model.to(runtime_device)
    model.eval()
    tokenizer_path = (
        Path(tokenizer) if tokenizer is not None else _default_tokenizer_path(checkpoint_path)
    )
    tokenizer_value = _verify_tokenizer(tokenizer_path, metadata, model)
    return LoadedModelBundle(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        payload=payload,
        metadata=metadata,
        model=model,
        tokenizer=tokenizer_value,
        device=runtime_device,
    )


def verify_streaming_data_bundle(
    bundle: LoadedModelBundle,
    *,
    project_root: str | Path = ".",
) -> tuple[Path, Path]:
    """Verify the corpus bytes named by a checkpoint's streaming identity."""

    training = bundle.training_config
    if not training.get("streaming"):
        raise ValueError("checkpoint does not describe a streaming data bundle")
    root = Path(project_root).resolve()

    def project_file(value: object, name: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"checkpoint training configuration lacks {name}")
        candidate = root / value
        if candidate.is_symlink():
            raise ValueError(f"{name} must not be a symlink")
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"{name} escapes the project root")
        if not resolved.is_file():
            raise ValueError(f"{name} must be a regular file: {resolved}")
        return resolved

    corpus_path = project_file(training.get("corpus_artifact"), "corpus_artifact")
    shard_path = project_file(training.get("shard_manifest"), "shard_manifest")
    try:
        value = json.loads(shard_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("streaming shard manifest is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("streaming shard manifest root must be a mapping")
    manifest = ShardManifest.from_mapping(value)
    expected_hash = bundle.metadata.get("shard_manifest_hash")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("checkpoint does not contain a shard manifest identity")
    if expected_hash == "unavailable" or shard_manifest_hash(manifest) != expected_hash:
        raise ValueError("shard manifest hash does not match checkpoint metadata")
    if manifest.tokenizer_hash != bundle.metadata.get("tokenizer_hash"):
        raise ValueError("shard manifest tokenizer hash does not match checkpoint metadata")
    if manifest.source_manifest_hash != bundle.metadata.get("manifest_hash"):
        raise ValueError("shard manifest source hash does not match checkpoint metadata")
    if len(manifest.shards) != 1 or manifest.shards[0].path != corpus_path.name:
        raise ValueError("shard manifest does not identify the configured corpus artifact")
    StreamingShardDataset(manifest, corpus_path.parent, verify_shards=True)
    return corpus_path, shard_path
