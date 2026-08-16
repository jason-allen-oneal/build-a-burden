from pathlib import Path
import pytest
from ts_coder.tokenizer.metrics import evaluate_tokenizer
from ts_coder.tokenizer.trainer import train_tokenizer, load_tokenizer


def test_train_roundtrip_and_hash(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    texts = [
        "export const café = `hello ${name}`;\n",
        "const value = foo?.bar ?? 1;\n",
        "export const View=()=> <main>ok</main>;\n",
    ]
    path = tmp_path / "tokenizer.json"
    metadata = train_tokenizer(texts, path, vocab_size=300, min_frequency=1)
    tok = load_tokenizer(path)
    metrics = evaluate_tokenizer(tok, texts)
    assert metadata["sha256"] and metrics["round_trip_success_rate"] == 1.0
    assert not metrics["special_token_collision"]
