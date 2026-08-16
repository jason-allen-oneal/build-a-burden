from __future__ import annotations

from ts_coder.data.packed_stream import PackedTokenBlockBatcher
from ts_coder.data.streaming import ShardDescriptor, ShardManifest, StreamingShardDataset
from ts_coder.data.token_stream import TokenizedStreamingDataset


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    def encode(self, text: str) -> FakeEncoding:
        return FakeEncoding([index + 1 for index, _ in enumerate(text)])


def make_dataset(tmp_path) -> TokenizedStreamingDataset:
    shard = tmp_path / "shard-00000.jsonl"
    shard.write_text(
        '{"record":{"record_id":"first","repository_id":"repo-a","split":"train"},"text":"abcdefghij"}\n'
        '{"record":{"record_id":"second","repository_id":"repo-a","split":"train"},"text":"klmnopqrst"}\n'
        '{"record":{"record_id":"other","repository_id":"repo-b","split":"train"},"text":"uvwxyz"}\n',
        encoding="utf-8",
    )
    manifest = ShardManifest(
        (ShardDescriptor("shard-00000", shard.name, records=3),),
        "tokenizer-hash",
        "source-hash",
    )
    return TokenizedStreamingDataset(
        StreamingShardDataset(manifest, tmp_path),
        FakeTokenizer(),
        context_length=4,
        pad_id=0,
        seed=7,
        fim_fraction=0.0,
        split="train",
        tokenizer_hash="tokenizer-hash",
    )


def test_packed_blocks_fill_context_and_never_cross_repository(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batches = list(PackedTokenBlockBatcher(dataset, eos_id=99, batch_size=1, epochs=1))
    assert batches
    assert all(batch["input_ids"].shape == (1, 4) for batch in batches)
    assert all(batch.actual_input_tokens <= 4 for batch in batches)
    for batch in batches:
        ids = set(batch.record_ids)
        assert not ({"first", "second"} <= ids and "other" in ids)
    assert any(batch.record_ids == ("first", "second") for batch in batches)
    assert any(batch.record_ids == ("other",) for batch in batches)
    assert batches[-1].padding_tokens > 0


def test_packed_blocks_handle_exact_boundary_before_repository_change(tmp_path) -> None:
    shard = tmp_path / "shard-00000.jsonl"
    shard.write_text(
        '{"record":{"record_id":"exact","repository_id":"repo-a",'
        '"split":"train"},"text":"abc"}\n'
        '{"record":{"record_id":"next","repository_id":"repo-b",'
        '"split":"train"},"text":"d"}\n',
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
        context_length=4,
        pad_id=0,
        split="train",
        tokenizer_hash="tokenizer-hash",
    )

    batches = list(PackedTokenBlockBatcher(dataset, eos_id=99, batch_size=1, epochs=1))

    assert [batch.record_ids for batch in batches] == [("exact",), ("next",)]
    assert batches[0]["loss_mask"].tolist() == [[0, 1, 1, 1]]
    assert batches[0].objective_token_counts == {"causal": 3, "fim": 0}
    assert batches[1].objective_token_counts == {"causal": 1, "fim": 0}


def test_packed_batch_objective_counts_match_shifted_loss_targets(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batches = list(PackedTokenBlockBatcher(dataset, eos_id=99, batch_size=2, epochs=1))

    for batch in batches:
        assert sum(batch.objective_token_counts.values()) == int(batch["loss_mask"][:, 1:].sum())


def test_packed_stream_applies_compiler_harness_training_view(tmp_path) -> None:
    shard = tmp_path / "shard-00000.jsonl"
    shard.write_text(
        '{"record":{"record_id":"harness","repository_id":"repo-a",'
        '"split":"train"},"text":"// @Filename: /a.ts\\nexport const x = 1;"}\n'
        '{"record":{"record_id":"ordinary","repository_id":"repo-a",'
        '"split":"train"},"text":"export const y = 2;"}\n',
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

    batches = list(PackedTokenBlockBatcher(dataset, eos_id=99, batch_size=1, epochs=1))

    assert batches
    assert all(batch.record_ids == ("ordinary",) for batch in batches)


def test_packed_cursor_resumes_without_replaying_a_block(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    first = list(PackedTokenBlockBatcher(dataset, eos_id=99, batch_size=1, epochs=1))[0]
    resumed = list(
        PackedTokenBlockBatcher(
            dataset,
            eos_id=99,
            batch_size=1,
            cursor=type(dataset.initial_cursor()).from_mapping(first.data_position),
            epochs=1,
        )
    )
    assert resumed
    assert resumed[0].data_position != first.data_position
    assert resumed[0]["input_ids"].tolist() != first["input_ids"].tolist()
