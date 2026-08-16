import pytest
import torch

from ts_coder.cli import (
    _bounded_validation_batches,
    _resolve_training_device,
    _validate_training_config,
)


def test_training_device_defaults_to_cpu() -> None:
    assert _resolve_training_device(None) == torch.device("cpu")
    assert _resolve_training_device("cpu") == torch.device("cpu")


def test_training_device_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="invalid training device"):
        _resolve_training_device("not-a-device")


def test_training_device_fails_closed_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        _resolve_training_device("cuda")


def test_validation_budget_bounds_by_actual_input_tokens() -> None:
    batches = [
        {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        },
        {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        },
    ]
    bounded = list(_bounded_validation_batches(batches, 3))
    assert len(bounded) == 2
    bounded_two = list(_bounded_validation_batches(batches, 2))
    assert len(bounded_two) == 1
    assert bounded_two[0] is batches[0]


def test_validation_budget_rejects_nonpositive_value() -> None:
    with pytest.raises(ValueError, match="validation_max_tokens"):
        list(_bounded_validation_batches([], 0))


def test_training_config_accepts_validation_budget() -> None:
    _validate_training_config(
        {
            "sequence_length": 128,
            "objectives": {"causal_fraction": 0.5, "fim_fraction": 0.5},
            "validation_max_tokens": 1024,
        }
    )


def test_training_config_accepts_streaming_step_estimate() -> None:
    _validate_training_config(
        {
            "sequence_length": 4096,
            "objectives": {"causal_fraction": 0.5, "fim_fraction": 0.5},
            "streaming": True,
            "streaming_tokens_per_step_estimate": 300,
        }
    )


def test_training_config_rejects_nonpositive_streaming_step_estimate() -> None:
    with pytest.raises(ValueError, match="streaming_tokens_per_step_estimate"):
        _validate_training_config(
            {
                "sequence_length": 4096,
                "objectives": {"causal_fraction": 0.5, "fim_fraction": 0.5},
                "streaming_tokens_per_step_estimate": 0,
            }
        )
