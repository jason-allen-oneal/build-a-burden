"""Bounded-memory tokenization and batching for immutable JSONL shards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .causal import causal_example
from .fim import make_fim
from .objectives import use_fim
from .streaming import DataCursor, ShardManifest, StreamingShardDataset, StreamRecord


def is_compiler_harness_record(record: StreamRecord) -> bool:
    text = record.value.get("text", "")
    if not isinstance(text, str):
        return False
    return any(
        line.lstrip().startswith(("// @Filename:", "// @BaselineFile:"))
        for line in text.splitlines()
    )


class TextTokenizer(Protocol):
    def encode(self, text: str) -> Any: ...


def shard_manifest_hash(manifest: ShardManifest) -> str:
    encoded = json.dumps(manifest.to_mapping(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    loss_mask: tuple[int, ...]
    record_id: str
    objective: str
    shard_index: int
    record_offset: int
    token_start: int
    next_cursor: DataCursor

    def as_lists(self) -> dict[str, list[int]]:
        return {
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "attention_mask": list(self.attention_mask),
            "loss_mask": list(self.loss_mask),
        }


class TokenizedBatch(dict[str, torch.Tensor]):
    data_position: dict[str, int]
    shard_manifest_hash: str
    tokenizer_hash: str
    record_ids: tuple[str, ...]
    objective_counts: dict[str, int]
    objective_token_counts: dict[str, int]
    actual_input_tokens: int
    padded_input_tokens: int
    padding_tokens: int

    def __init__(
        self,
        examples: list[TokenizedExample],
        *,
        shard_hash: str,
        tokenizer_hash: str,
    ) -> None:
        if not examples:
            raise ValueError("cannot construct an empty tokenized batch")
        fields = examples[0].as_lists()
        if any(example.as_lists().keys() != fields.keys() for example in examples[1:]):
            raise ValueError("tokenized examples have incompatible fields")
        super().__init__(
            {
                key: torch.tensor(
                    [example.as_lists()[key] for example in examples], dtype=torch.long
                )
                for key in fields
            }
        )
        self.data_position = examples[-1].next_cursor.as_dict()
        self.shard_manifest_hash = shard_hash
        self.tokenizer_hash = tokenizer_hash
        self.record_ids = tuple(example.record_id for example in examples)
        self.objective_counts = {
            "causal": sum(example.objective == "causal" for example in examples),
            "fim": sum(example.objective == "fim" for example in examples),
        }
        self.objective_token_counts = {
            "causal": sum(
                sum(example.loss_mask[1:]) for example in examples if example.objective == "causal"
            ),
            "fim": sum(
                sum(example.loss_mask[1:]) for example in examples if example.objective == "fim"
            ),
        }
        self.actual_input_tokens = int(self["attention_mask"].bool().sum())
        self.padded_input_tokens = int(self["input_ids"].numel())
        self.padding_tokens = max(self.padded_input_tokens - self.actual_input_tokens, 0)


class TokenizedStreamingDataset:
    def __init__(
        self,
        shards: StreamingShardDataset,
        tokenizer: TextTokenizer,
        *,
        context_length: int,
        pad_id: int,
        seed: int = 42,
        fim_fraction: float = 0.0,
        fim_min_span: int = 1,
        fim_max_span: int = 128,
        split: str | None = "train",
        tokenizer_hash: str = "unavailable",
        exclude_compiler_harness: bool = False,
    ) -> None:
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        if pad_id < 0:
            raise ValueError("pad_id must not be negative")
        if not 0.0 <= fim_fraction <= 1.0:
            raise ValueError("fim_fraction must be in [0, 1]")
        if fim_min_span < 1 or fim_max_span < fim_min_span:
            raise ValueError("invalid FIM span lengths")
        if not tokenizer_hash:
            raise ValueError("tokenizer_hash must not be empty")
        self.shards = shards
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.pad_id = pad_id
        self.seed = seed
        self.fim_fraction = fim_fraction
        self.fim_min_span = fim_min_span
        self.fim_max_span = fim_max_span
        self.split = split
        self.tokenizer_hash = tokenizer_hash
        self.exclude_compiler_harness = bool(exclude_compiler_harness)
        self.shard_hash = shard_manifest_hash(shards.manifest)

    @property
    def rank(self) -> int:
        return self.shards.rank

    @property
    def world_size(self) -> int:
        return self.shards.world_size

    def initial_cursor(self) -> DataCursor:
        return DataCursor(rank=self.rank, world_size=self.world_size)

    def include_record(self, record: StreamRecord) -> bool:
        return not (self.exclude_compiler_harness and is_compiler_harness_record(record))

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)
        values = getattr(encoded, "ids", encoded)
        if not isinstance(values, (list, tuple)):
            raise ValueError("tokenizer.encode must return an object with list-like ids")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("tokenizer returned a non-integer token id")
        ids = [int(value) for value in values]
        if any(value < 0 for value in ids):
            raise ValueError("tokenizer returned a negative token id")
        return ids

    def _record_tokens(self, record: StreamRecord) -> tuple[list[int], str]:
        text = record.value["text"]
        objective = "causal"
        source = text
        if use_fim(record.record_id, self.seed, self.fim_fraction):
            try:
                source = make_fim(
                    text,
                    record.record_id,
                    self.seed,
                    min_span=self.fim_min_span,
                    max_span=self.fim_max_span,
                ).serialized
                objective = "fim"
            except ValueError:
                pass
        return self._encode(source), objective

    def _record_examples(
        self, record: StreamRecord, start_offset: int, epoch: int
    ) -> Iterator[TokenizedExample]:
        token_ids, objective = self._record_tokens(record)
        stride = self.context_length - 1
        for token_start in range(0, max(len(token_ids) - 1, 0), stride):
            if token_start < start_offset:
                continue
            window = token_ids[token_start : token_start + self.context_length]
            if len(window) < 2:
                continue
            values = causal_example(window, self.context_length, self.pad_id)
            next_start = token_start + stride
            if next_start < len(token_ids) - 1:
                next_cursor = DataCursor(
                    epoch=epoch,
                    shard_index=record.shard_index,
                    record_offset=record.record_offset,
                    token_offset=next_start,
                    rank=self.rank,
                    world_size=self.world_size,
                )
            else:
                next_cursor = DataCursor(
                    epoch=epoch,
                    shard_index=record.shard_index,
                    record_offset=record.record_offset + 1,
                    token_offset=0,
                    rank=self.rank,
                    world_size=self.world_size,
                )
            yield TokenizedExample(
                tuple(values["input_ids"]),
                tuple(values["labels"]),
                tuple(values["attention_mask"]),
                tuple(values["loss_mask"]),
                record.record_id,
                objective,
                record.shard_index,
                record.record_offset,
                token_start,
                next_cursor,
            )

    @staticmethod
    def _at_epoch_start(position: DataCursor) -> bool:
        return (
            position.shard_index == 0 and position.record_offset == 0 and position.token_offset == 0
        )

    def iter_examples(
        self,
        cursor: DataCursor | None = None,
        *,
        epochs: int | None = 1,
    ) -> Iterator[TokenizedExample]:
        if epochs is not None and epochs < 1:
            raise ValueError("epochs must be positive or null")
        position = cursor or self.initial_cursor()
        if position.rank != self.rank or position.world_size != self.world_size:
            raise ValueError("cursor partition does not match dataset partition")
        completed = 0
        while epochs is None or completed < epochs:
            yielded = False
            started_at_epoch_start = self._at_epoch_start(position)
            for record in self.shards.iter_records(position):
                if self.split is not None and record.value["record"].get("split") != self.split:
                    continue
                if not self.include_record(record):
                    continue
                start_offset = (
                    position.token_offset
                    if record.shard_index == position.shard_index
                    and record.record_offset == position.record_offset
                    else 0
                )
                for example in self._record_examples(record, start_offset, position.epoch):
                    yielded = True
                    yield example
            completed += 1
            if epochs is not None and completed >= epochs:
                break
            if not yielded and started_at_epoch_start:
                raise ValueError("cannot repeat an empty tokenized shard stream")
            position = DataCursor(
                epoch=position.epoch + 1,
                shard_index=0,
                record_offset=0,
                token_offset=0,
                rank=self.rank,
                world_size=self.world_size,
            )


class TokenizedStreamingBatcher:
    def __init__(
        self,
        dataset: TokenizedStreamingDataset,
        *,
        batch_size: int,
        cursor: DataCursor | None = None,
        epochs: int | None = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if epochs is not None and epochs < 1:
            raise ValueError("epochs must be positive or null")
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self._cursor = cursor or dataset.initial_cursor()
        if self._cursor.rank != dataset.rank or self._cursor.world_size != dataset.world_size:
            raise ValueError("cursor partition does not match dataset partition")

    @property
    def cursor(self) -> DataCursor:
        return self._cursor

    def seek(self, cursor: DataCursor) -> None:
        if cursor.rank != self.dataset.rank or cursor.world_size != self.dataset.world_size:
            raise ValueError("cursor partition does not match dataset partition")
        self._cursor = cursor

    def __iter__(self) -> Iterator[TokenizedBatch]:
        examples: list[TokenizedExample] = []
        stream = self.dataset.iter_examples(self._cursor, epochs=self.epochs)
        for example in stream:
            examples.append(example)
            if len(examples) < self.batch_size:
                continue
            batch = TokenizedBatch(
                examples,
                shard_hash=self.dataset.shard_hash,
                tokenizer_hash=self.dataset.tokenizer_hash,
            )
            yield batch
            self._cursor = examples[-1].next_cursor
            examples = []
        if examples:
            batch = TokenizedBatch(
                examples,
                shard_hash=self.dataset.shard_hash,
                tokenizer_hash=self.dataset.tokenizer_hash,
            )
            yield batch
            self._cursor = examples[-1].next_cursor


__all__ = [
    "TextTokenizer",
    "TokenizedBatch",
    "TokenizedExample",
    "TokenizedStreamingBatcher",
    "TokenizedStreamingDataset",
    "is_compiler_harness_record",
    "shard_manifest_hash",
]
