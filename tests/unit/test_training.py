from __future__ import annotations

import os
import json
import signal

import pytest
import torch
import torch.nn.functional as F

from ts_coder.cli import _corpus_documents, _validate_resume_identity
from ts_coder.model import ModelConfig, Transformer
from ts_coder.training.optimizer import build_adamw
from ts_coder.training.scheduler import build_cosine_scheduler
from ts_coder.training.checkpoint import load_checkpoint
from ts_coder.training.metrics import append_metrics, input_token_count, supervised_token_count
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


def batches(count: int):
    generator = torch.Generator().manual_seed(3)
    for _ in range(count):
        ids = torch.randint(0, 32, (2, 6), generator=generator)
        yield {"input_ids": ids, "labels": ids, "loss_mask": torch.ones_like(ids, dtype=torch.bool)}


def test_training_checkpoint_resume_preserves_cursor_and_state(tmp_path) -> None:
    torch.manual_seed(2)
    model = make_model()
    optimizer = build_adamw(model, 1e-3)
    scheduler = build_cosine_scheduler(optimizer, 1, 4)
    trainer = Trainer(model, optimizer, scheduler)
    metrics = trainer.train_steps(batches(2), max_steps=2)
    assert len(metrics) == 2
    assert trainer.state.data_cursor == 2
    checkpoint = trainer.save(
        tmp_path / "checkpoint.pt", {"model": "tiny"}, "tok", "manifest", "dirty"
    )
    safe_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert safe_payload["metadata"]["format_version"] == 1

    restored_model = make_model()
    restored_optimizer = build_adamw(restored_model, 1e-3)
    restored_scheduler = build_cosine_scheduler(restored_optimizer, 1, 4)
    restored = Trainer(restored_model, restored_optimizer, restored_scheduler)
    metadata = restored.resume(checkpoint)
    assert (metadata.global_step, metadata.data_cursor, metadata.tokens_processed) == (2, 2, 24)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            model.state_dict().values(), restored_model.state_dict().values(), strict=True
        )
    )
    more = restored.train_steps(batches(1), max_steps=3)
    assert len(more) == 1
    assert restored.state.data_cursor == 3


def test_gradient_accumulation_advances_data_cursor() -> None:
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3), gradient_accumulation_steps=2)
    trainer.train_steps(batches(2), max_steps=1)
    assert trainer.state.global_step == 1
    assert trainer.state.data_cursor == 2
    assert trainer.state.tokens_processed == 24


def test_token_accounting_distinguishes_inputs_from_next_token_targets() -> None:
    batch = next(batches(1))
    assert input_token_count(batch) == 12
    assert supervised_token_count(batch) == 10
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3))
    metrics = trainer.train_steps(iter([batch]), max_steps=1)
    assert metrics[0].batch_tokens == 12
    assert metrics[0].loss_tokens == 10
    assert metrics[0].padded_tokens == 12
    assert metrics[0].padding_tokens == 0
    assert metrics[0].token_utilization == 1.0


def test_training_metrics_report_short_window_padding() -> None:
    model = make_model()
    ids = torch.tensor([[3, 7, 0, 0], [11, 2, 0, 0]])
    batch = {
        "input_ids": ids,
        "labels": ids,
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.long),
        "loss_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.long),
    }
    trainer = Trainer(model, build_adamw(model, 1e-3))
    metrics = trainer.train_steps(iter([batch]), max_steps=1)
    assert metrics[0].batch_tokens == 4
    assert metrics[0].padded_tokens == 8
    assert metrics[0].padding_tokens == 4
    assert metrics[0].token_utilization == 0.5


def test_metrics_jsonl_persists_padding_efficiency(tmp_path) -> None:
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3))
    metric = trainer.train_steps(iter([next(batches(1))]), max_steps=1)[0]
    path = tmp_path / "metrics.jsonl"
    append_metrics(path, metric)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["padded_tokens"] == 12
    assert payload["padding_tokens"] == 0
    assert payload["token_utilization"] == 1.0


def test_token_budget_stops_repeated_stream() -> None:
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3))
    batch = next(batches(1))
    metrics = trainer.train_steps(iter([batch] * 10), max_steps=10, max_tokens=20)
    assert len(metrics) == 2
    assert trainer.state.tokens_processed == 24


def test_bfloat16_autocast_path_is_supported_on_cpu() -> None:
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3), precision="bf16")
    metrics = trainer.train_steps(batches(1), max_steps=1)
    assert len(metrics) == 1


def test_token_interval_callbacks_fire_after_crossing_boundaries() -> None:
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3))
    checkpoints: list[int] = []
    samples: list[int] = []
    validations: list[int] = []
    trainer.train_steps(
        batches(4),
        max_steps=4,
        checkpoint_interval_tokens=20,
        sample_interval_tokens=30,
        validation_interval_tokens=15,
        checkpoint_callback=lambda state: checkpoints.append(state.tokens_processed),
        sample_callback=lambda state: samples.append(state.tokens_processed),
        validation_callback=lambda state: validations.append(state.tokens_processed),
    )
    assert checkpoints == [24, 48]
    assert samples == [36]
    assert validations == [24, 36, 48]


def test_keyboard_interrupt_callback_saves_resumable_cursor(tmp_path) -> None:
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3))
    checkpoint = tmp_path / "interrupted.pt"

    def interrupted_stream():
        yield next(batches(1))
        raise KeyboardInterrupt

    def save_interrupt(_state) -> None:
        trainer.save(checkpoint, {"model": "tiny"}, "tok", "manifest", "dirty")

    with pytest.raises(KeyboardInterrupt):
        trainer.train_steps(interrupted_stream(), max_steps=3, interrupt_callback=save_interrupt)

    restored_model = make_model()
    restored = Trainer(restored_model, build_adamw(restored_model, 1e-3))
    restored.resume(checkpoint)
    assert restored.state.global_step == 1
    assert restored.state.data_cursor == 1
    assert restored.state.tokens_processed == 12


def test_interrupt_inside_accumulation_does_not_advance_cursor(tmp_path) -> None:
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3), gradient_accumulation_steps=2)
    checkpoint = tmp_path / "partial-step.pt"

    def partial_step_stream():
        yield next(batches(1))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        trainer.train_steps(
            partial_step_stream(),
            max_steps=1,
            interrupt_callback=lambda _state: trainer.save(
                checkpoint, {"model": "tiny"}, "tok", "manifest", "dirty"
            ),
        )
    restored_model = make_model()
    restored = Trainer(restored_model, build_adamw(restored_model, 1e-3))
    restored.resume(checkpoint)
    assert restored.state.global_step == 0
    assert restored.state.data_cursor == 0
    assert restored.state.tokens_processed == 0


def test_sigint_from_optimizer_step_checkpoints_post_commit_state(tmp_path, monkeypatch) -> None:
    model = make_model()
    optimizer = build_adamw(model, 1e-3)
    trainer = Trainer(model, optimizer)
    checkpoint = tmp_path / "optimizer-sigint.pt"
    original_step = optimizer.step

    def step_with_sigint(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", step_with_sigint)
    with pytest.raises(KeyboardInterrupt):
        trainer.train_steps(
            batches(1),
            max_steps=1,
            interrupt_callback=lambda _state: trainer.save(
                checkpoint, {"model": "tiny"}, "tok", "manifest", "dirty"
            ),
        )
    restored_model = make_model()
    restored = Trainer(restored_model, build_adamw(restored_model, 1e-3))
    restored.resume(checkpoint)
    assert restored.state.global_step == 1
    assert restored.state.data_cursor == 1
    assert restored.state.tokens_processed == 12


def test_keyboard_interrupt_raised_by_optimizer_is_uncommitted(tmp_path, monkeypatch) -> None:
    model = make_model()
    optimizer = build_adamw(model, 1e-3)
    trainer = Trainer(model, optimizer)
    checkpoint = tmp_path / "optimizer-exception.pt"

    def interrupted_step(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(optimizer, "step", interrupted_step)
    with pytest.raises(KeyboardInterrupt):
        trainer.train_steps(
            batches(1),
            max_steps=1,
            interrupt_callback=lambda _state: trainer.save(
                checkpoint, {"model": "tiny"}, "tok", "manifest", "dirty"
            ),
        )
    restored_model = make_model()
    restored = Trainer(restored_model, build_adamw(restored_model, 1e-3))
    restored.resume(checkpoint)
    assert restored.state.global_step == 0
    assert restored.state.data_cursor == 0
    assert restored.state.tokens_processed == 0


def test_checkpoint_schema_rejects_incomplete_tensor_envelope(tmp_path) -> None:
    path = tmp_path / "malformed.pt"
    torch.save({"model": {}}, path)
    model = make_model()
    with torch.no_grad():
        try:
            load_checkpoint(path, model)
        except ValueError as exc:
            assert "checkpoint" in str(exc)
        else:
            raise AssertionError("malformed checkpoint was accepted")


def test_checkpoint_v2_persists_shard_cursor_and_parallel_identity(tmp_path) -> None:
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3))
    checkpoint = trainer.save(
        tmp_path / "scale-out.pt",
        {"model": {"hidden_size": 16}, "training": {}},
        "tokenizer",
        "manifest",
        "dirty",
        data_position={
            "epoch": 1,
            "shard_index": 2,
            "record_offset": 7,
            "token_offset": 128,
            "rank": 1,
            "world_size": 2,
        },
        shard_manifest_hash="shard-manifest",
        parallel={"world_size": 2, "rank": 1, "local_rank": 1},
    )
    restored_model = make_model()
    restored = Trainer(restored_model, build_adamw(restored_model, 1e-3))
    metadata = restored.resume(checkpoint)
    assert metadata.format_version == 2
    assert metadata.data_position is not None
    assert metadata.data_position["record_offset"] == 7
    assert metadata.parallel == {"world_size": 2, "rank": 1, "local_rank": 1}


def test_configured_corpus_artifact_is_loaded_instead_of_default(tmp_path) -> None:
    configured = tmp_path / "chosen.jsonl"
    configured.write_text('{"text":"chosen","record":{"split":"train"}}\n', encoding="utf-8")
    assert _corpus_documents(tmp_path, "chosen.jsonl")[0]["text"] == "chosen"


def test_model_applies_exactly_one_causal_shift() -> None:
    torch.manual_seed(9)
    model = make_model()
    ids = torch.tensor([[3, 7, 11, 2]])
    output = model(ids, labels=ids, loss_mask=torch.ones_like(ids, dtype=torch.bool))
    expected = F.cross_entropy(output.logits[:, :-1].reshape(-1, 32), ids[:, 1:].reshape(-1))
    assert output.loss is not None
    assert torch.allclose(output.loss, expected)


def test_resume_identity_rejects_hash_and_critical_config_mismatch(tmp_path) -> None:
    model = make_model()
    trainer = Trainer(model, build_adamw(model, 1e-3))
    model_config = {"hidden_size": 16}
    training_config = {
        "corpus_artifact": "corpus.jsonl",
        "manifest": "manifest.jsonl",
        "tokenizer_dir": "tokenizer",
        "seed": 42,
        "sequence_length": 16,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "objectives": {"causal_fraction": 0.5, "fim_fraction": 0.5},
    }
    checkpoint = trainer.save(
        tmp_path / "resume.pt",
        {"model": model_config, "training": training_config},
        "tokenizer-hash",
        "manifest-hash",
        "dirty",
    )
    _validate_resume_identity(
        checkpoint, model_config, training_config, "tokenizer-hash", "manifest-hash"
    )
    with pytest.raises(ValueError, match="tokenizer hash"):
        _validate_resume_identity(
            checkpoint, model_config, training_config, "wrong", "manifest-hash"
        )
    changed = {**training_config, "sequence_length": 8}
    with pytest.raises(ValueError, match="data-critical"):
        _validate_resume_identity(
            checkpoint, model_config, changed, "tokenizer-hash", "manifest-hash"
        )
    changed_microbatch = {**training_config, "micro_batch_size": 2}
    with pytest.raises(ValueError, match="micro_batch_size"):
        _validate_resume_identity(
            checkpoint,
            model_config,
            changed_microbatch,
            "tokenizer-hash",
            "manifest-hash",
        )
