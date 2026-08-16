from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainingMetrics:
    step: int
    tokens_processed: int
    loss: float
    learning_rate: float
    gradient_norm: float
    tokens_per_second: float
    batch_tokens: int
    sequence_length: int
    wall_clock_seconds: float
    # ``batch_tokens`` is the number of input tokens consumed.  ``loss_tokens``
    # is the number of next-token targets that contributed to the loss.  They
    # differ by one per unpadded sequence and by all padding/masked positions;
    # keeping both prevents throughput and optimization accounting from being
    # conflated at scale.
    loss_tokens: int = 0
    # ``padded_tokens`` is the tensor size presented to the model, including
    # padding.  ``padding_tokens`` makes the cost of short per-record windows
    # visible instead of hiding it behind the actual-token throughput number.
    # These fields are optional for backwards-compatible loading of older
    # metrics records.
    padded_tokens: int = 0
    padding_tokens: int = 0

    @property
    def token_utilization(self) -> float:
        """Return the fraction of model positions carrying input tokens."""

        if self.padded_tokens <= 0:
            return 0.0
        return self.batch_tokens / self.padded_tokens


def input_token_count(batch: dict[str, torch.Tensor]) -> int:
    """Count consumed (non-padding) input positions for progress accounting."""

    input_ids = batch.get("input_ids")
    if input_ids is None:
        raise ValueError("training batch requires input_ids")
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        return int(attention_mask.bool().sum())
    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        if loss_mask.shape != input_ids.shape:
            raise ValueError("loss_mask must match input_ids")
        return int(loss_mask.bool().sum())
    return int(input_ids.numel())


def supervised_token_count(batch: dict[str, torch.Tensor]) -> int:
    """Count exactly the next-token positions used by ``Transformer.forward``."""

    labels = batch.get("labels")
    if labels is None:
        return 0
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must be [batch, sequence] with at least two positions")
    valid = labels[:, 1:].ne(-100)
    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        if loss_mask.shape != labels.shape:
            raise ValueError("loss_mask must match labels")
        valid &= loss_mask[:, 1:].bool()
    return int(valid.sum())


def append_metrics(path: Path, metrics: TrainingMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(metrics)
    # Keep the derived ratio in the machine-readable record as well as on the
    # dataclass.  ``asdict`` intentionally serializes fields only, not
    # properties, and a run report should make padding waste visible without
    # requiring consumers to recompute it.
    payload["token_utilization"] = metrics.token_utilization
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
