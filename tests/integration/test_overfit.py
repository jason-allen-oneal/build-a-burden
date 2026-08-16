import torch

from ts_coder.model import ModelConfig, Transformer
from ts_coder.training.optimizer import build_adamw
from ts_coder.training.trainer import Trainer


def test_development_model_overfits_one_fixed_batch():
    model = Transformer(
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
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "loss_mask": torch.ones((1, 6), dtype=torch.bool),
    }
    trainer = Trainer(model, build_adamw(model, 1e-2))
    initial = float(model(**batch).loss)
    trainer.train_steps([batch] * 20, max_steps=20)
    final = float(model(**batch).loss)
    assert final < initial
