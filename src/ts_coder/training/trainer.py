from __future__ import annotations

import math
import signal
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..data.streaming import DataCursor
from .checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from .metrics import (
    TrainingMetrics,
    append_metrics,
    input_token_count,
    supervised_token_count,
)


@dataclass
class TrainerState:
    global_step: int = 0
    tokens_processed: int = 0
    data_cursor: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        device: str | torch.device = "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        metrics_path: Path | None = None,
        precision: str = "fp32",
    ) -> None:
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.metrics_path = metrics_path
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        self.precision = precision
        self.state = TrainerState()
        self.data_position: dict[str, int] | None = None
        self.shard_manifest_hash: str | None = None
        self.parallel: dict[str, int] | None = None
        self.objective_examples = {"causal": 0, "fim": 0}
        self.objective_tokens = {"causal": 0, "fim": 0}

    def train_steps(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        max_steps: int,
        max_tokens: int | None = None,
        checkpoint_interval_tokens: int | None = None,
        sample_interval_tokens: int | None = None,
        validation_interval_tokens: int | None = None,
        checkpoint_callback: Callable[[TrainerState], None] | None = None,
        sample_callback: Callable[[TrainerState], None] | None = None,
        validation_callback: Callable[[TrainerState], None] | None = None,
        interrupt_callback: Callable[[TrainerState], None] | None = None,
    ) -> list[TrainingMetrics]:
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        for name, interval in (
            ("checkpoint_interval_tokens", checkpoint_interval_tokens),
            ("sample_interval_tokens", sample_interval_tokens),
            ("validation_interval_tokens", validation_interval_tokens),
        ):
            if interval is not None and interval <= 0:
                raise ValueError(f"{name} must be positive")
        self.model.train()
        iterator = iter(batches)
        produced: list[TrainingMetrics] = []
        next_checkpoint = _next_boundary(self.state.tokens_processed, checkpoint_interval_tokens)
        next_sample = _next_boundary(self.state.tokens_processed, sample_interval_tokens)
        next_validation = _next_boundary(self.state.tokens_processed, validation_interval_tokens)
        try:
            while self.state.global_step < max_steps and (
                max_tokens is None or self.state.tokens_processed < max_tokens
            ):
                metrics = self._train_step(iterator)
                if metrics is None:
                    return produced
                produced.append(metrics)
                if self.metrics_path is not None:
                    append_metrics(self.metrics_path, metrics)
                if next_checkpoint is not None and self.state.tokens_processed >= next_checkpoint:
                    if checkpoint_callback is not None:
                        checkpoint_callback(self.state)
                    next_checkpoint = _next_boundary(
                        self.state.tokens_processed, checkpoint_interval_tokens
                    )
                if next_sample is not None and self.state.tokens_processed >= next_sample:
                    if sample_callback is not None:
                        sample_callback(self.state)
                    next_sample = _next_boundary(
                        self.state.tokens_processed, sample_interval_tokens
                    )
                if next_validation is not None and self.state.tokens_processed >= next_validation:
                    if validation_callback is not None:
                        validation_callback(self.state)
                    next_validation = _next_boundary(
                        self.state.tokens_processed, validation_interval_tokens
                    )
        except KeyboardInterrupt:
            if interrupt_callback is not None:
                interrupt_callback(self.state)
            raise
        return produced

    def _train_step(self, iterator: Iterator[Mapping[str, torch.Tensor]]) -> TrainingMetrics | None:
        started = time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        batch_tokens = 0
        padded_tokens = 0
        loss_tokens = 0
        batches_consumed = 0
        sequence_length = 0
        pending_data_position = self.data_position
        pending_shard_manifest_hash = self.shard_manifest_hash
        pending_objective_examples = dict(self.objective_examples)
        pending_objective_tokens = dict(self.objective_tokens)
        for _ in range(self.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                if batches_consumed == 0:
                    return None
                break
            batch_position = getattr(batch, "data_position", None)
            if batch_position is not None:
                if not isinstance(batch_position, dict):
                    raise ValueError("streaming batch data_position must be a mapping")
                DataCursor.from_mapping(batch_position)
                pending_data_position = dict(batch_position)
            batch_shard_hash = getattr(batch, "shard_manifest_hash", None)
            if batch_shard_hash is not None:
                if not isinstance(batch_shard_hash, str) or not batch_shard_hash:
                    raise ValueError("streaming batch shard_manifest_hash is invalid")
                if (
                    pending_shard_manifest_hash is not None
                    and pending_shard_manifest_hash != batch_shard_hash
                ):
                    raise ValueError("streaming batches changed shard manifest identity")
                pending_shard_manifest_hash = batch_shard_hash
            batch_objectives = getattr(batch, "objective_counts", None)
            batch_objective_tokens = getattr(batch, "objective_token_counts", None)
            if batch_objectives is not None:
                if not isinstance(batch_objectives, dict):
                    raise ValueError("streaming batch objective_counts is invalid")
                for objective in ("causal", "fim"):
                    pending_objective_examples[objective] += int(batch_objectives.get(objective, 0))
            if batch_objective_tokens is not None:
                if not isinstance(batch_objective_tokens, dict):
                    raise ValueError("streaming batch objective_token_counts is invalid")
                for objective in ("causal", "fim"):
                    pending_objective_tokens[objective] += int(
                        batch_objective_tokens.get(objective, 0)
                    )
            prepared = {key: value.to(self.device) for key, value in batch.items()}
            autocast = (
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.precision == "bf16"
                else nullcontext()
            )
            with autocast:
                output = self.model(**prepared)
            if output.loss is None or not torch.isfinite(output.loss):
                raise FloatingPointError("training loss is missing, NaN, or infinite")
            batch_loss_tokens = supervised_token_count(prepared)
            if batch_loss_tokens:
                (output.loss * batch_loss_tokens).backward()
            accumulated_loss += float(output.loss.detach()) * batch_loss_tokens
            loss_tokens += batch_loss_tokens
            batch_tokens += input_token_count(prepared)
            padded_tokens += int(prepared["input_ids"].numel())
            sequence_length = int(prepared["input_ids"].shape[1])
            batches_consumed += 1
        if loss_tokens <= 0:
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("training step contains no supervised target tokens")
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(loss_tokens)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError("gradient norm is NaN or infinite")
        with _defer_sigint():
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.state.global_step += 1
            self.state.tokens_processed += batch_tokens
            self.state.data_cursor += batches_consumed
            self.data_position = pending_data_position
            self.shard_manifest_hash = pending_shard_manifest_hash
            self.objective_examples = pending_objective_examples
            self.objective_tokens = pending_objective_tokens
        elapsed = max(time.monotonic() - started, 1e-9)
        return TrainingMetrics(
            self.state.global_step,
            self.state.tokens_processed,
            accumulated_loss / loss_tokens,
            self.optimizer.param_groups[0]["lr"],
            float(grad_norm),
            batch_tokens / elapsed,
            batch_tokens,
            sequence_length,
            elapsed,
            loss_tokens,
            padded_tokens,
            max(padded_tokens - batch_tokens, 0),
        )

    @torch.no_grad()
    def evaluate_batches(self, batches: Iterable[Mapping[str, torch.Tensor]]) -> float | None:
        """Evaluate masked causal loss without changing optimizer or RNG state."""
        was_training = self.model.training
        self.model.eval()
        weighted_loss = 0.0
        loss_tokens = 0
        try:
            for batch in batches:
                prepared = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(**prepared)
                if output.loss is not None and torch.isfinite(output.loss):
                    count = supervised_token_count(prepared)
                    weighted_loss += float(output.loss) * count
                    loss_tokens += count
        finally:
            self.model.train(was_training)
        return weighted_loss / loss_tokens if loss_tokens else None

    def save(
        self,
        path: Path,
        resolved_config: dict,
        tokenizer_hash: str,
        manifest_hash: str,
        git_commit: str,
        data_position: dict[str, int] | None = None,
        shard_manifest_hash: str | None = None,
        parallel: dict[str, int] | None = None,
    ) -> Path:
        if data_position is None:
            data_position = self.data_position
        if shard_manifest_hash is None:
            shard_manifest_hash = self.shard_manifest_hash
        if parallel is None:
            parallel = self.parallel
        format_version = (
            2
            if any(value is not None for value in (data_position, shard_manifest_hash, parallel))
            else 1
        )
        metadata = CheckpointMetadata(
            self.state.global_step,
            self.state.tokens_processed,
            self.state.data_cursor,
            resolved_config,
            tokenizer_hash,
            manifest_hash,
            git_commit,
            format_version,
            data_position,
            shard_manifest_hash,
            parallel,
        )
        return save_checkpoint(path, self.model, self.optimizer, self.scheduler, metadata)

    def resume(self, path: Path) -> CheckpointMetadata:
        metadata = load_checkpoint(path, self.model, self.optimizer, self.scheduler)
        self.state = TrainerState(
            metadata.global_step, metadata.tokens_processed, metadata.data_cursor
        )
        self.data_position = metadata.data_position
        self.shard_manifest_hash = metadata.shard_manifest_hash
        self.parallel = metadata.parallel
        return metadata


def _next_boundary(current: int, interval: int | None) -> int | None:
    return None if interval is None else ((current // interval) + 1) * interval


@contextmanager
def _defer_sigint():
    """Defer process SIGINT until a training state commit is complete."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    if previous == signal.SIG_IGN:
        yield
        return
    pending = False

    def defer(_signum, _frame) -> None:
        nonlocal pending
        pending = True

    signal.signal(signal.SIGINT, defer)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if pending:
            if callable(previous):
                previous(signal.SIGINT, None)
            else:
                raise KeyboardInterrupt
