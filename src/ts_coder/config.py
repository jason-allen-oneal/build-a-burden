"""Strict YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .model.config import ModelConfig

T = TypeVar("T")


@dataclass(frozen=True)
class TokenizerConfig:
    type: str = "byte_level_bpe"
    vocab_size: int = 4096
    min_frequency: int = 2

    def __post_init__(self) -> None:
        if self.type != "byte_level_bpe" or self.vocab_size < 256 or self.min_frequency < 1:
            raise ValueError("invalid tokenizer configuration")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 2
    sequence_length: int = 128
    max_steps: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1
    grad_accumulation: int = 1
    grad_clip: float = 1.0
    precision: str = "fp32"
    causal_fraction: float = 0.5
    fim_fraction: float = 0.5
    checkpoint_interval: int = 10

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "sequence_length",
            "max_steps",
            "warmup_steps",
            "grad_accumulation",
            "checkpoint_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer values")
        if not 0 <= self.causal_fraction <= 1 or not 0 <= self.fim_fraction <= 1:
            raise ValueError("objective fractions must be in [0, 1]")
        if abs(self.causal_fraction + self.fim_fraction - 1) > 1e-6:
            raise ValueError("causal_fraction + fim_fraction must equal 1")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")


def _strict(cls: type[T], values: dict[str, Any]) -> T:
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown configuration keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_yaml(path: str | Path, cls: type[T]) -> T:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    return _strict(cls, values)


def resolve_config(path: str | Path) -> dict[str, Any]:
    """Load a config file with a recognized section, rejecting unknown sections."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {"model": ModelConfig, "tokenizer": TokenizerConfig, "training": TrainingConfig}
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    return {name: vars(_strict(allowed[name], section or {})) for name, section in values.items()}
