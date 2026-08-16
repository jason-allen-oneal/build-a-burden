"""Train a byte-level BPE tokenizer only from supplied approved text."""

import hashlib
import json
from pathlib import Path

from .special_tokens import SPECIAL_TOKENS, validate_special_tokens


def train_tokenizer(
    texts: list[str], output_path: Path, *, vocab_size: int = 4096, min_frequency: int = 2
) -> dict:
    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("vocab_size too small")
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")
    validate_special_tokens()
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))  # nosec B106 - model control token, not a secret
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(iter(texts), trainer=trainer, length=len(texts))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_path), pretty=True)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "type": "byte_level_bpe",
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tok.get_vocab_size(),
        "min_frequency": min_frequency,
        "sha256": digest,
        "special_tokens": list(SPECIAL_TOKENS),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))
