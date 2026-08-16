"""Tokenizer quality measurements."""

import re

from .special_tokens import SPECIAL_TOKENS


def evaluate_tokenizer(tokenizer, texts: list[str]) -> dict:
    totals = {"bytes": 0, "chars": 0, "tokens": 0, "lines": 0, "lexical": 0}
    roundtrips = 0
    used = set()
    for text in texts:
        enc = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids, skip_special_tokens=False)
        totals["bytes"] += len(text.encode())
        totals["chars"] += len(text)
        totals["tokens"] += len(enc.ids)
        totals["lines"] += max(1, len(text.splitlines()))
        totals["lexical"] += len(re.findall(r"[A-Za-z_$][\w$]*|\d+|\S", text))
        used.update(enc.ids)
        roundtrips += decoded == text
    n = max(totals["tokens"], 1)
    special = [tokenizer.token_to_id(x) for x in SPECIAL_TOKENS]
    operators = [
        "=>",
        "?.",
        "??",
        "??=",
        "!==",
        "===",
        "&&",
        "||",
        "satisfies",
        "as const",
        "keyof",
        "infer",
        "extends",
    ]
    operator_token_counts = {
        operator: len(tokenizer.encode(operator).ids) for operator in operators
    }
    identifiers = re.findall(r"[A-Za-z_$][\w$]*", "\n".join(texts))
    identifier_fragmentation = sum(
        len(tokenizer.encode(identifier).ids) for identifier in identifiers
    ) / max(len(identifiers), 1)
    import_paths = re.findall(r"(?:from|import)\s+[\"']([^\"']+)[\"']", "\n".join(texts))
    import_path_fragmentation = sum(len(tokenizer.encode(path).ids) for path in import_paths) / max(
        len(import_paths), 1
    )
    return {
        "documents": len(texts),
        "bytes_per_token": totals["bytes"] / n,
        "characters_per_token": totals["chars"] / n,
        "tokens_per_line": totals["tokens"] / max(totals["lines"], 1),
        "tokens_per_lexical_token": totals["tokens"] / max(totals["lexical"], 1),
        "round_trip_success_rate": roundtrips / max(len(texts), 1),
        "vocabulary_utilization": len(used) / max(tokenizer.get_vocab_size(), 1),
        "special_token_ids": dict(zip(SPECIAL_TOKENS, special, strict=False)),
        "special_token_collision": len(special) != len(set(special)) or None in special,
        "operator_token_counts": operator_token_counts,
        "identifier_fragmentation": identifier_fragmentation,
        "import_path_fragmentation": import_path_fragmentation,
        "tsx_present": any("<" in text and ">" in text for text in texts),
        "template_literal_present": any("`" in text and "${" in text for text in texts),
    }
