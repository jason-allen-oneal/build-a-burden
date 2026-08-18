import hashlib
import json
from types import SimpleNamespace

import pytest

from ts_coder.data.packed_stream import PackedTokenBlockBatcher
from ts_coder.data.streaming import (
    DataCursor,
    ShardDescriptor,
    ShardManifest,
    StreamingShardDataset,
)
from ts_coder.data.token_stream import TokenizedStreamingDataset


class FakeTokenizer:
    def encode(self, text: str):
        return SimpleNamespace(ids=[(byte % 23) + 1 for byte in text.encode("utf-8")])


def write_shard(path, *, split="train"):
    payload = (
        json.dumps(
            {
                "record": {
                    "record_id": "record-1",
                    "repository_id": "repo-1",
                    "split": split,
                },
                "text": "const value: number = 1;",
            },
            sort_keys=True,
        )
        + "\n"
    )
    path.write_text(payload, encoding="utf-8")
    return payload


def make_dataset(tmp_path):
    shard = tmp_path / "documents.jsonl"
    payload = write_shard(shard)
    descriptor = ShardDescriptor(
        "documents.jsonl",
        "documents.jsonl",
        1,
        sha256=hashlib.sha256(payload.encode()).hexdigest(),
    )
    manifest = ShardManifest((descriptor,), "tokenizer", "manifest")
    return shard, StreamingShardDataset(manifest, tmp_path)


def test_streaming_dataset_rejects_changed_shard_bytes(tmp_path) -> None:
    shard, dataset = make_dataset(tmp_path)
    assert list(dataset.iter_records())
    shard.write_text(shard.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="record count mismatch|sha256 mismatch"):
        StreamingShardDataset(dataset.manifest, tmp_path)


def test_token_stream_resume_after_final_record_advances_epoch(tmp_path) -> None:
    _shard, shards = make_dataset(tmp_path)
    dataset = TokenizedStreamingDataset(
        shards,
        FakeTokenizer(),
        context_length=16,
        pad_id=0,
        tokenizer_hash="tokenizer",
    )
    cursor = DataCursor(epoch=0, shard_index=0, record_offset=1)

    example = next(dataset.iter_examples(cursor, epochs=None))

    assert example.next_cursor.epoch == 1


def test_packed_stream_resume_after_final_record_advances_epoch(tmp_path) -> None:
    _shard, shards = make_dataset(tmp_path)
    dataset = TokenizedStreamingDataset(
        shards,
        FakeTokenizer(),
        context_length=16,
        pad_id=0,
        tokenizer_hash="tokenizer",
    )
    cursor = DataCursor(epoch=0, shard_index=0, record_offset=1)
    batcher = PackedTokenBlockBatcher(
        dataset,
        eos_id=31,
        batch_size=1,
        cursor=cursor,
        epochs=None,
    )

    batch = next(iter(batcher))

    assert batch.data_position["epoch"] == 1
