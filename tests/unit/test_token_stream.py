from __future__ import annotations

from ts_coder.data.streaming import (
    ShardDescriptor,
    ShardManifest,
    StreamRecord,
    StreamingShardDataset,
)
from ts_coder.data.token_stream import (
    TokenizedStreamingBatcher,
    TokenizedStreamingDataset,
    is_compiler_harness_record,
    shard_manifest_hash,
)
from ts_coder.model import ModelConfig, Transformer
from ts_coder.training.optimizer import build_adamw
from ts_coder.training.trainer import Trainer


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    def encode(self, text: str) -> FakeEncoding:
        # The fixture uses one byte-sized token per character.  It is enough
        # to test windowing and resume semantics without training a tokenizer.
        return FakeEncoding([index + 1 for index, _ in enumerate(text)])


def make_dataset(tmp_path) -> TokenizedStreamingDataset:
    shard = tmp_path / "shard-00000.jsonl"
    shard.write_text(
        '{"record":{"record_id":"first","split":"train"},"text":"abcdefghij"}\n'
        '{"record":{"record_id":"second","split":"train"},"text":"klmnopqrst"}\n'
        '{"record":{"record_id":"held-out","split":"validation"},"text":"uvwxyz"}\n',
        encoding="utf-8",
    )
    manifest = ShardManifest(
        (ShardDescriptor("shard-00000", shard.name, records=3),),
        "tokenizer-hash",
        "source-hash",
    )
    shards = StreamingShardDataset(manifest, tmp_path)
    return TokenizedStreamingDataset(
        shards,
        FakeTokenizer(),
        context_length=4,
        pad_id=0,
        seed=7,
        fim_fraction=0.0,
        split="train",
        tokenizer_hash="tokenizer-hash",
    )


def test_compiler_harness_filter_is_narrow_and_does_not_remove_decorators() -> None:
    harness = StreamRecord(
        0,
        0,
        "harness",
        {
            "record": {"record_id": "harness"},
            "text": "// @Filename: /a.ts\nexport const value = 1;",
        },
    )
    baseline = StreamRecord(
        0,
        1,
        "baseline",
        {
            "record": {"record_id": "baseline"},
            "text": "// @BaselineFile: /a.ts\nexport const value = 1;",
        },
    )
    decorator = StreamRecord(
        0,
        2,
        "decorator",
        {
            "record": {"record_id": "decorator"},
            "text": "@Column()\nvalue!: string;",
        },
    )
    ordinary_comment = StreamRecord(
        0,
        3,
        "ordinary-comment",
        {
            "record": {"record_id": "ordinary-comment"},
            "text": "// Filename: user.ts\nexport const value = 1;",
        },
    )

    assert is_compiler_harness_record(harness)
    assert is_compiler_harness_record(baseline)
    assert not is_compiler_harness_record(decorator)
    assert not is_compiler_harness_record(ordinary_comment)


def test_compiler_harness_records_can_be_excluded_from_stream_view(tmp_path) -> None:
    shard = tmp_path / "shard-00000.jsonl"
    shard.write_text(
        '{"record":{"record_id":"harness","split":"train"},'
        '"text":"// @Filename: /a.ts\\nexport const x = 1;"}\n'
        '{"record":{"record_id":"ordinary","split":"train"},'
        '"text":"@Column()\\nexport const x = 1;"}\n',
        encoding="utf-8",
    )
    manifest = ShardManifest(
        (ShardDescriptor("shard-00000", shard.name, records=2),),
        "tokenizer-hash",
        "source-hash",
    )
    dataset = TokenizedStreamingDataset(
        StreamingShardDataset(manifest, tmp_path),
        FakeTokenizer(),
        context_length=16,
        pad_id=0,
        split="train",
        tokenizer_hash="tokenizer-hash",
        exclude_compiler_harness=True,
    )

    examples = list(dataset.iter_examples())
    assert examples
    assert {example.record_id for example in examples} == {"ordinary"}


def test_tokenized_stream_is_bounded_and_resumes_inside_a_record(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    examples = list(dataset.iter_examples())
    assert len(examples) == 6
    assert examples[0].record_id == "first"
    assert examples[0].token_start == 0
    assert examples[0].next_cursor.token_offset == 3
    assert examples[2].next_cursor.record_offset == 1
    assert examples[-1].record_id == "second"

    resumed = list(dataset.iter_examples(examples[0].next_cursor))
    assert [(item.record_id, item.token_start) for item in resumed] == [
        ("first", 3),
        ("first", 6),
        ("second", 0),
        ("second", 3),
        ("second", 6),
    ]


def test_batch_metadata_is_model_compatible_and_hash_pinned(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batcher = TokenizedStreamingBatcher(dataset, batch_size=2)
    batch = next(iter(batcher))
    assert set(batch) == {"input_ids", "labels", "attention_mask", "loss_mask"}
    assert batch["input_ids"].shape == (2, 4)
    assert batch.data_position["token_offset"] == 6
    assert batch.shard_manifest_hash == shard_manifest_hash(dataset.shards.manifest)
    assert batch.tokenizer_hash == "tokenizer-hash"
    assert batch.objective_counts == {"causal": 2, "fim": 0}
    assert batch.objective_token_counts == {"causal": 6, "fim": 0}
    assert batch.actual_input_tokens == 8
    assert batch.padded_input_tokens == 8
    assert batch.padding_tokens == 0


def test_stream_objective_selection_is_reproducible(tmp_path) -> None:
    first = make_dataset(tmp_path)
    first.fim_fraction = 1.0
    second = make_dataset(tmp_path)
    second.fim_fraction = 1.0
    first_examples = list(first.iter_examples())
    second_examples = list(second.iter_examples())
    assert [(item.record_id, item.token_start, item.objective) for item in first_examples] == [
        (item.record_id, item.token_start, item.objective) for item in second_examples
    ]
    assert all(item.objective == "fim" for item in first_examples)


def test_trainer_persists_stream_cursor_after_committed_step(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batch = next(iter(TokenizedStreamingBatcher(dataset, batch_size=1)))
    model = Transformer(
        ModelConfig(
            vocab_size=32,
            context_length=4,
            layers=1,
            hidden_size=16,
            attention_heads=2,
            kv_heads=1,
            ffn_size=32,
        )
    )
    trainer = Trainer(model, build_adamw(model, 1e-3))
    trainer.train_steps(iter([batch]), max_steps=1)
    checkpoint = trainer.save(
        tmp_path / "stream-checkpoint.pt",
        {"model": "tiny", "training": {}},
        "tokenizer-hash",
        "source-hash",
        "working-tree",
    )
    restored_model = Transformer(
        ModelConfig(
            vocab_size=32,
            context_length=4,
            layers=1,
            hidden_size=16,
            attention_heads=2,
            kv_heads=1,
            ffn_size=32,
        )
    )
    restored = Trainer(restored_model, build_adamw(restored_model, 1e-3))
    metadata = restored.resume(checkpoint)
    assert metadata.data_position == batch.data_position
    assert metadata.shard_manifest_hash == batch.shard_manifest_hash
    assert restored.data_position == batch.data_position
    assert restored.shard_manifest_hash == batch.shard_manifest_hash
