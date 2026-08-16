"""No-op-safe container interface. External code execution is not enabled by default."""

from __future__ import annotations

from .limits import SandboxLimits


class SandboxUnavailable(RuntimeError):
    pass


def run_in_sandbox(*_args, limits: SandboxLimits | None = None, **_kwargs):
    _ = limits or SandboxLimits()
    raise SandboxUnavailable(
        "sandbox runner is a Milestone 2+ integration; refusing host execution"
    )
