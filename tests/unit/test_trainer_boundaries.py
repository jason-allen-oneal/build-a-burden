import pytest
import torch

from ts_coder.model import ModelConfig, Transformer
from ts_coder.training.optimizer import build_adamw
from ts_coder.training.trainer import Trainer


def make_model() -> Transformer:
    return Transformer(
        ModelConfig(
            vocab_size=32,
            context_length=16,
            layers=1,
            hidden_size=16,
            attention_heads=2,
            kv_heads=1,
            ffn_size=32,
        )
    )


def batch(*, supervised=True):
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    return {
        "input_ids": ids,
        "labels": ids,
        "attention_mask": torch.ones_like(ids),
        "loss_mask": torch.ones_like(ids) if supervised else torch.zeros_like(ids),
    }


def test_partial_gradient_accumulation_commits_consumed_batch() -> None:
    model = make_model()
    trainer = Trainer(
        model,
        build_adamw(model, 1e-3),
        gradient_accumulation_steps=2,
    )

    metrics = trainer.train_steps(iter([batch()]), max_steps=1)

    assert len(metrics) == 1
    assert trainer.state.global_step == 1
    assert trainer.state.data_cursor == 1


def test_zero_supervision_never_updates_optimizer_or_state() -> None:
    model = make_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trainer = Trainer(model, build_adamw(model, 1e-3))

    with pytest.raises(ValueError, match="no supervised"):
        trainer.train_steps(iter([batch(supervised=False)]), max_steps=1)

    assert trainer.state.global_step == 0
    assert trainer.state.tokens_processed == 0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_trainer_rejects_invalid_gradient_clip(value) -> None:
    model = make_model()
    with pytest.raises(ValueError, match="max_grad_norm"):
        Trainer(model, build_adamw(model, 1e-3), max_grad_norm=value)
