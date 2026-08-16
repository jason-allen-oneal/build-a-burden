"""Bounded-memory repository-local token packing for GPU training.

The record-oriented stream is deliberately conservative, but it wastes most of
an A100 sequence when source files are short.  This adapter packs serialized
records into full context blocks while never putting two repositories in the
same block.  It keeps one record's token list and one block in memory, emits an
explicit EOS between records, and carries the exact next source cursor as
checkpoint metadata.

Packing is opt-in.  The original :mod:`token_stream` path remains the reference
fallback and is preferable when debugging provenance or cursor behavior.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from .streaming import DataCursor, StreamRecord
from .token_stream import TokenizedStreamingDataset


@dataclass(frozen=True)
class PackedBlock:
    """One full or padded block and the cursor after its final source token."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    loss_mask: tuple[int, ...]
    record_ids: tuple[str, ...]
    objective_counts: dict[str, int]
    objective_token_counts: dict[str, int]
    next_cursor: DataCursor


class PackedTokenBatch(dict[str, torch.Tensor]):
    """Model-compatible batch carrying packed-stream metadata out of band."""

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
        self, blocks: list[PackedBlock], *, shard_hash: str, tokenizer_hash: str, pad_id: int
    ) -> None:
        if not blocks:
            raise ValueError("cannot construct an empty packed batch")
        width = len(blocks[0].input_ids)
        if width < 2 or any(len(block.input_ids) != width for block in blocks):
            raise ValueError("packed blocks must have the same width")
        super().__init__(
            {
                "input_ids": torch.tensor([block.input_ids for block in blocks], dtype=torch.long),
                "labels": torch.tensor([block.labels for block in blocks], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [block.attention_mask for block in blocks], dtype=torch.long
                ),
                "loss_mask": torch.tensor([block.loss_mask for block in blocks], dtype=torch.long),
            }
        )
        self.data_position = blocks[-1].next_cursor.as_dict()
        self.shard_manifest_hash = shard_hash
        self.tokenizer_hash = tokenizer_hash
        self.record_ids = tuple(record_id for block in blocks for record_id in block.record_ids)
        self.objective_counts = {
            objective: sum(block.objective_counts.get(objective, 0) for block in blocks)
            for objective in ("causal", "fim")
        }
        self.objective_token_counts = {
            objective: sum(block.objective_token_counts.get(objective, 0) for block in blocks)
            for objective in ("causal", "fim")
        }
        self.actual_input_tokens = int(self["attention_mask"].bool().sum())
        self.padded_input_tokens = int(self["input_ids"].numel())
        self.padding_tokens = max(self.padded_input_tokens - self.actual_input_tokens, 0)
        # Keep the constructor's pad_id explicit so accidental pad collisions
        # are caught in tests and during future tokenizer changes.
        if pad_id < 0:
            raise ValueError("pad_id must not be negative")


class PackedTokenBlockBatcher:
    """Pack tokenized records into repository-local context blocks.

    ``batch_size`` is the microbatch dimension.  Each block is independently
    causal, so blocks from different repositories may share a tensor batch but
    can never attend to one another.  A block never crosses a repository
    boundary; a partial block is padded and flushed before the next repository.
    """

    def __init__(
        self,
        dataset: TokenizedStreamingDataset,
        *,
        eos_id: int,
        batch_size: int,
        cursor: DataCursor | None = None,
        epochs: int | None = 1,
    ) -> None:
        if eos_id < 0:
            raise ValueError("eos_id must not be negative")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if epochs is not None and epochs < 1:
            raise ValueError("epochs must be positive or null")
        self.dataset = dataset
        self.eos_id = eos_id
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

    def _record_tokens(self, record: StreamRecord) -> tuple[list[int], str]:
        values, objective = self.dataset._record_tokens(record)  # noqa: SLF001
        return [*values, self.eos_id], objective

    def _blocks(self, cursor: DataCursor, epochs: int | None) -> Iterator[PackedBlock]:
        position = cursor
        completed = 0
        while epochs is None or completed < epochs:
            records = self.dataset.shards.iter_records(position)
            pending: tuple[StreamRecord, list[int], str, int] | None = None
            block_ids: list[int] = []
            block_mask: list[int] = []
            block_loss: list[int] = []
            block_records: list[str] = []
            block_objectives = {"causal": 0, "fim": 0}
            block_token_objectives: list[str] = []
            block_repo: str | None = None
            last_cursor = position
            yielded = False

            while True:
                if pending is None:
                    try:
                        record = next(records)
                    except StopIteration:
                        record = None
                    if record is None:
                        if block_ids:
                            yield self._make_block(
                                block_ids,
                                block_mask,
                                block_loss,
                                block_records,
                                block_objectives,
                                block_token_objectives,
                                last_cursor,
                            )
                            block_ids, block_mask, block_loss, block_records = [], [], [], []
                            block_objectives = {"causal": 0, "fim": 0}
                            block_token_objectives = []
                            block_repo = None
                            yielded = True
                        break
                    if (
                        self.dataset.split is not None
                        and record.value["record"].get("split") != self.dataset.split
                    ):
                        continue
                    if not self.dataset.include_record(record):
                        continue
                    token_ids, objective = self._record_tokens(record)
                    start = (
                        position.token_offset
                        if (
                            record.shard_index == position.shard_index
                            and record.record_offset == position.record_offset
                        )
                        else 0
                    )
                    pending = (record, token_ids, objective, start)
                record, token_ids, objective, start = pending
                repository_id = str(record.value["record"].get("repository_id", ""))
                if block_repo is not None and repository_id != block_repo:
                    if block_ids:
                        yield self._make_block(
                            block_ids,
                            block_mask,
                            block_loss,
                            block_records,
                            block_objectives,
                            block_token_objectives,
                            last_cursor,
                        )
                    block_ids, block_mask, block_loss, block_records = [], [], [], []
                    block_objectives = {"causal": 0, "fim": 0}
                    block_token_objectives = []
                    block_repo = None
                    yielded = True
                    continue
                block_repo = repository_id
                available = self.dataset.context_length - len(block_ids)
                take = min(available, len(token_ids) - start)
                if take <= 0:
                    pending = None
                    continue
                block_ids.extend(token_ids[start : start + take])
                block_mask.extend([1] * take)
                block_loss.extend([1] * take)
                block_token_objectives.extend([objective] * take)
                if record.record_id not in block_records:
                    block_records.append(record.record_id)
                    block_objectives[objective] += 1
                next_start = start + take
                if next_start < len(token_ids):
                    last_cursor = DataCursor(
                        epoch=position.epoch,
                        shard_index=record.shard_index,
                        record_offset=record.record_offset,
                        token_offset=next_start,
                        rank=self.dataset.rank,
                        world_size=self.dataset.world_size,
                    )
                    pending = (record, token_ids, objective, next_start)
                else:
                    last_cursor = DataCursor(
                        epoch=position.epoch,
                        shard_index=record.shard_index,
                        record_offset=record.record_offset + 1,
                        token_offset=0,
                        rank=self.dataset.rank,
                        world_size=self.dataset.world_size,
                    )
                    pending = None
                if len(block_ids) >= self.dataset.context_length:
                    yield self._make_block(
                        block_ids,
                        block_mask,
                        block_loss,
                        block_records,
                        block_objectives,
                        block_token_objectives,
                        last_cursor,
                    )
                    block_ids, block_mask, block_loss, block_records = [], [], [], []
                    block_objectives = {"causal": 0, "fim": 0}
                    block_token_objectives = []
                    block_repo = None
                    yielded = True

            completed += 1
            if epochs is not None and completed >= epochs:
                break
            if not yielded:
                raise ValueError("cannot repeat an empty packed token stream")
            position = DataCursor(
                epoch=position.epoch + 1,
                shard_index=0,
                record_offset=0,
                token_offset=0,
                rank=self.dataset.rank,
                world_size=self.dataset.world_size,
            )

    def _make_block(
        self,
        ids: list[int],
        mask: list[int],
        loss: list[int],
        records: list[str],
        objectives: dict[str, int],
        token_objectives: list[str],
        cursor: DataCursor,
    ) -> PackedBlock:
        width = self.dataset.context_length
        if len(ids) < 2:
            raise ValueError("packed block must contain at least two tokens")
        if len(mask) != len(ids) or len(loss) != len(ids):
            raise ValueError("packed block masks must match token count")
        if len(token_objectives) != len(ids):
            raise ValueError("packed block objective labels must match token count")
        pad = width - len(ids)
        # The first token conditions the block and is not a next-token target.
        # The trainer shifts masks once more, so this also documents the
        # accounting contract at the data boundary without changing model
        # semantics for callers that provide their own masks.
        block_loss = [0, *([1] * (len(ids) - 1))]
        shifted_objectives = token_objectives[1:]
        exact_objective_tokens = {
            objective: shifted_objectives.count(objective) for objective in ("causal", "fim")
        }
        return PackedBlock(
            tuple(ids + [self.dataset.pad_id] * pad),
            tuple(ids + [self.dataset.pad_id] * pad),
            tuple(mask + [0] * pad),
            tuple(block_loss + [0] * pad),
            tuple(records),
            dict(objectives),
            exact_objective_tokens,
            cursor,
        )

    def __iter__(self) -> Iterator[PackedTokenBatch]:
        blocks: list[PackedBlock] = []
        for block in self._blocks(self._cursor, self.epochs):
            blocks.append(block)
            if len(blocks) < self.batch_size:
                continue
            batch = PackedTokenBatch(
                blocks,
                shard_hash=self.dataset.shard_hash,
                tokenizer_hash=self.dataset.tokenizer_hash,
                pad_id=self.dataset.pad_id,
            )
            yield batch
            self._cursor = blocks[-1].next_cursor
            blocks = []
        if blocks:
            batch = PackedTokenBatch(
                blocks,
                shard_hash=self.dataset.shard_hash,
                tokenizer_hash=self.dataset.tokenizer_hash,
                pad_id=self.dataset.pad_id,
            )
            yield batch
            self._cursor = blocks[-1].next_cursor


__all__ = ["PackedBlock", "PackedTokenBatch", "PackedTokenBlockBatcher"]
