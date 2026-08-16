import pytest

from ts_coder.config import ModelConfig, TrainingConfig, load_yaml
from ts_coder.training.distributed import DistributedConfig, validate_distributed_config


def test_model_validation():
    assert ModelConfig().hidden_size == 256
    assert ModelConfig().use_sdpa is True
    assert ModelConfig(initializer_range=0.01, rms_norm_eps=1e-5).initializer_range == 0.01
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=257)


def test_objective_fraction_validation(tmp_path):
    with pytest.raises(ValueError):
        TrainingConfig(causal_fraction=0.2, fim_fraction=0.2)
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(path, ModelConfig)


def test_distributed_config_is_explicit_and_fail_closed() -> None:
    assert validate_distributed_config(None).world_size == 1
    distributed = DistributedConfig(
        strategy="ddp", backend="nccl", world_size=2, rank=1, local_rank=1
    )
    assert distributed.rank == 1
    with pytest.raises(ValueError, match="world_size"):
        DistributedConfig(strategy="single", world_size=2)
    with pytest.raises(ValueError, match="unknown distributed"):
        validate_distributed_config({"not_a_real_key": True})
