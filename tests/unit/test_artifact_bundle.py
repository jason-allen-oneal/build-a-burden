import hashlib
import json
from dataclasses import asdict

import pytest

from ts_coder.artifacts import load_model_bundle, verify_streaming_data_bundle
from ts_coder.data.streaming import ShardDescriptor, ShardManifest
from ts_coder.data.token_stream import shard_manifest_hash
from ts_coder.model import ModelConfig, Transformer
from ts_coder.reproducibility import sha256_file
from ts_coder.tokenizer.special_tokens import SPECIAL_TOKENS


class FakeTokenizer:
    def __init__(self, vocab_size: int, *, special_offset: int = 0) -> None:
        self._vocab_size = vocab_size
        self._special = {
            token: index + special_offset for index, token in enumerate(SPECIAL_TOKENS)
        }

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def token_to_id(self, token: str):
        return self._special.get(token)


def make_payload(tokenizer_path, *, training=None, shard_hash=None):
    config = ModelConfig(
        vocab_size=32,
        context_length=16,
        layers=1,
        hidden_size=16,
        attention_heads=2,
        kv_heads=1,
        ffn_size=32,
    )
    model = Transformer(config)
    metadata = {
        "resolved_config": {
            "model": {"schema_version": 1, **asdict(config)},
            "training": training or {},
        },
        "tokenizer_hash": sha256_file(tokenizer_path),
        "manifest_hash": "a" * 64,
    }
    if shard_hash is not None:
        metadata["shard_manifest_hash"] = shard_hash
    return {"model": model.state_dict(), "metadata": metadata}


def load_fixture_bundle(tmp_path, monkeypatch, *, payload=None):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("fixture tokenizer", encoding="utf-8")
    checkpoint = tmp_path / "latest"
    checkpoint.write_bytes(b"checkpoint")
    value = payload or make_payload(tokenizer_path)
    monkeypatch.setattr("ts_coder.artifacts.load_checkpoint_payload", lambda _path: value)
    monkeypatch.setattr("ts_coder.artifacts.load_tokenizer", lambda _path: FakeTokenizer(32))
    return load_model_bundle(checkpoint, tokenizer=tokenizer_path), tokenizer_path


def test_bundle_requires_matching_tokenizer_identity(tmp_path, monkeypatch) -> None:
    bundle, tokenizer_path = load_fixture_bundle(tmp_path, monkeypatch)

    assert bundle.model.config.vocab_size == 32
    assert bundle.tokenizer_path == tokenizer_path


def test_bundle_rejects_tokenizer_hash_mismatch(tmp_path, monkeypatch) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("fixture tokenizer", encoding="utf-8")
    checkpoint = tmp_path / "latest"
    checkpoint.write_bytes(b"checkpoint")
    payload = make_payload(tokenizer_path)
    tokenizer_path.write_text("tampered tokenizer", encoding="utf-8")
    monkeypatch.setattr("ts_coder.artifacts.load_checkpoint_payload", lambda _path: payload)
    monkeypatch.setattr("ts_coder.artifacts.load_tokenizer", lambda _path: FakeTokenizer(32))

    with pytest.raises(ValueError, match="tokenizer hash"):
        load_model_bundle(checkpoint, tokenizer=tokenizer_path)


def test_bundle_rejects_vocabulary_mismatch(tmp_path, monkeypatch) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("fixture tokenizer", encoding="utf-8")
    checkpoint = tmp_path / "latest"
    checkpoint.write_bytes(b"checkpoint")
    payload = make_payload(tokenizer_path)
    monkeypatch.setattr("ts_coder.artifacts.load_checkpoint_payload", lambda _path: payload)
    monkeypatch.setattr("ts_coder.artifacts.load_tokenizer", lambda _path: FakeTokenizer(31))

    with pytest.raises(ValueError, match="vocabulary mismatch"):
        load_model_bundle(checkpoint, tokenizer=tokenizer_path)


def test_streaming_bundle_binds_manifest_to_corpus_bytes(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "documents.jsonl"
    corpus_payload = (
        json.dumps({"record": {"record_id": "one", "split": "train"}, "text": "const one = 1;"})
        + "\n"
    )
    corpus.write_text(corpus_payload, encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("fixture tokenizer", encoding="utf-8")
    descriptor = ShardDescriptor(
        "documents.jsonl",
        "documents.jsonl",
        1,
        sha256=hashlib.sha256(corpus_payload.encode()).hexdigest(),
    )
    manifest = ShardManifest(
        (descriptor,),
        sha256_file(tokenizer_path),
        "a" * 64,
    )
    shard_path = tmp_path / "shards.json"
    shard_path.write_text(json.dumps(manifest.to_mapping()), encoding="utf-8")

    payload = make_payload(
        tokenizer_path,
        training={
            "streaming": True,
            "corpus_artifact": "documents.jsonl",
            "shard_manifest": "shards.json",
        },
        shard_hash=shard_manifest_hash(manifest),
    )
    bundle, _ = load_fixture_bundle(tmp_path, monkeypatch, payload=payload)

    assert verify_streaming_data_bundle(bundle, project_root=tmp_path) == (corpus, shard_path)

    corpus.write_text(corpus_payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record count mismatch|sha256 mismatch"):
        verify_streaming_data_bundle(bundle, project_root=tmp_path)


def test_bundle_rejects_noncanonical_special_token_ids(tmp_path, monkeypatch) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("fixture tokenizer", encoding="utf-8")
    checkpoint = tmp_path / "latest"
    checkpoint.write_bytes(b"checkpoint")
    payload = make_payload(tokenizer_path)
    monkeypatch.setattr("ts_coder.artifacts.load_checkpoint_payload", lambda _path: payload)
    monkeypatch.setattr(
        "ts_coder.artifacts.load_tokenizer",
        lambda _path: FakeTokenizer(32, special_offset=1),
    )

    with pytest.raises(ValueError, match="special-token IDs"):
        load_model_bundle(checkpoint, tokenizer=tokenizer_path)
