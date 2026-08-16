import pytest

from ts_coder.data.causal import causal_example, causal_examples
from ts_coder.data.fim import make_fim, reconstruct
from ts_coder.data.objectives import use_fim
from ts_coder.data.streaming import (
    DataCursor,
    ShardDescriptor,
    ShardManifest,
    StreamingShardDataset,
    iter_jsonl_records,
)


def test_fim_deterministic_and_reconstructs_unicode() -> None:
    text = "export const café = `hello ${name}`;"
    one = make_fim(text, "record", 7, 2, 8)
    two = make_fim(text, "record", 7, 2, 8)
    assert one == two and reconstruct(one) == text
    assert one.serialized.startswith("<fim_prefix>")


def test_causal_padding_masks_loss() -> None:
    sample = causal_example([1, 2, 3], 4, 0)
    assert sample["input_ids"] == [1, 2, 3, 0]
    assert sample["labels"] == [1, 2, 3, 0]
    assert sample["loss_mask"] == [1, 1, 1, 0]


def test_causal_examples_cover_later_tokens() -> None:
    samples = causal_examples(list(range(1, 12)), 4, 0)
    assert len(samples) == 4
    assert samples[0]["input_ids"] == [1, 2, 3, 4]
    assert samples[1]["input_ids"] == [4, 5, 6, 7]
    assert samples[3]["labels"] == [10, 11, 0, 0]


def test_objective_selection_is_deterministic_and_honors_fraction() -> None:
    ids = [f"record-{index}" for index in range(100_000)]
    selected = [sample_id for sample_id in ids if use_fim(sample_id, 71, 0.37)]
    assert 36_000 <= len(selected) <= 38_000
    assert (
        selected == [sample_id for sample_id in reversed(ids) if use_fim(sample_id, 71, 0.37)][::-1]
    )
    assert not any(use_fim(sample_id, 71, 0.0) for sample_id in ids[:10])
    assert all(use_fim(sample_id, 71, 1.0) for sample_id in ids[:10])


def test_fim_rejects_text_that_cannot_provide_a_safe_span() -> None:
    token = "<fim_prefix>"
    with pytest.raises(ValueError, match="special-token"):
        make_fim(token, "special", 1, min_span=len(token), max_span=len(token))


def test_streaming_shards_validate_and_resume_by_line_offset(tmp_path) -> None:
    shard = tmp_path / "shard-00000.jsonl"
    shard.write_text(
        "".join(
            f'{{"record":{{"record_id":"record-{index}"}},"text":"export const x{index} = {index};"}}\n'
            for index in range(4)
        ),
        encoding="utf-8",
    )
    descriptor = ShardDescriptor("shard-00000", shard.name, records=4)
    manifest = ShardManifest((descriptor,), "tokenizer", "source")
    dataset = StreamingShardDataset(manifest, tmp_path)
    records = list(dataset.iter_records())
    assert [record.record_id for record in records] == [
        "record-0",
        "record-1",
        "record-2",
        "record-3",
    ]
    cursor = dataset.cursor_after(records[1], token_offset=17, epoch=2)
    assert cursor == DataCursor(2, 0, 2, 17, 0, 1)
    resumed = list(dataset.iter_records(cursor))
    assert [record.record_id for record in resumed] == ["record-2", "record-3"]
    assert list(iter_jsonl_records(shard))[0][0] == 0


def test_streaming_shards_partition_by_stable_record_id(tmp_path) -> None:
    shard = tmp_path / "shard.jsonl"
    shard.write_text(
        "".join(
            f'{{"record":{{"record_id":"record-{index}"}},"text":"x"}}\n' for index in range(20)
        ),
        encoding="utf-8",
    )
    manifest = ShardManifest((ShardDescriptor("shard", shard.name, 20),), "tok", "source")
    workers = [
        StreamingShardDataset(manifest, tmp_path, rank=rank, world_size=2) for rank in range(2)
    ]
    first = {record.record_id for record in workers[0].iter_records()}
    second = {record.record_id for record in workers[1].iter_records()}
    assert first and second and first.isdisjoint(second)
    assert first | second == {f"record-{index}" for index in range(20)}


def test_streaming_shard_rejects_path_escape_and_malformed_records(tmp_path) -> None:
    with pytest.raises(ValueError, match="parent traversal"):
        ShardDescriptor("bad", "../bad.jsonl", 1)
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        list(iter_jsonl_records(malformed))
