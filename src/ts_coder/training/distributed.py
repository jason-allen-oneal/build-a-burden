"""Distributed-training configuration contracts.

Milestone 1 intentionally runs one process.  This module makes the future
launcher boundary explicit without pretending that a local ``Trainer`` is
DDP/FSDP-capable: a non-single strategy must be rejected by the current CLI
until process-group setup, rank-aware checkpointing, and reduction semantics
are wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DistributedConfig:
    strategy: str = "single"
    backend: str = "gloo"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    timeout_seconds: int = 1800
    data_partition: str = "stable_record_hash"

    def __post_init__(self) -> None:
        if self.strategy not in {"single", "ddp", "fsdp"}:
            raise ValueError("distributed strategy must be single, ddp, or fsdp")
        if self.backend not in {"gloo", "nccl", "mpi"}:
            raise ValueError("distributed backend must be gloo, nccl, or mpi")
        if self.world_size < 1 or self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("distributed world_size/rank is invalid")
        if self.local_rank < 0 or self.timeout_seconds <= 0:
            raise ValueError("distributed local_rank/timeout_seconds is invalid")
        if self.data_partition != "stable_record_hash":
            raise ValueError("only stable_record_hash data partitioning is supported")
        if self.strategy == "single" and (self.world_size != 1 or self.rank != 0):
            raise ValueError("single strategy requires world_size=1 and rank=0")
        if self.strategy != "single" and self.world_size < 2:
            raise ValueError("ddp/fsdp strategy requires world_size >= 2")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> DistributedConfig:
        values = dict(value or {})
        allowed = {
            "strategy",
            "backend",
            "world_size",
            "rank",
            "local_rank",
            "timeout_seconds",
            "data_partition",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown distributed configuration keys: {sorted(unknown)}")
        return cls(**values)


def validate_distributed_config(value: Mapping[str, Any] | None) -> DistributedConfig:
    """Validate a config section without initializing a process group."""

    return DistributedConfig.from_mapping(value)
