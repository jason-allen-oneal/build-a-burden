"""Declarative limits for future container execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: int = 10
    cpu_seconds: int = 2
    memory_mb: int = 512
    pids: int = 64
    file_size_mb: int = 32
    network: bool = False

    def __post_init__(self) -> None:
        if (
            min(
                self.timeout_seconds, self.cpu_seconds, self.memory_mb, self.pids, self.file_size_mb
            )
            <= 0
        ):
            raise ValueError("sandbox limits must be positive")
        if self.network:
            raise ValueError("network-enabled evaluation is prohibited")
