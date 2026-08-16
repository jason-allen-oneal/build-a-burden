"""Stable tokenizer control vocabulary."""

SPECIAL_TOKENS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<repo>",
    "</repo>",
    "<file>",
    "</file>",
    "<fim_prefix>",
    "<fim_suffix>",
    "<fim_middle>",
    "<diagnostic>",
    "</diagnostic>",
    "<test>",
    "</test>",
    "<patch>",
    "</patch>",
)


def validate_special_tokens(tokens: tuple[str, ...] = SPECIAL_TOKENS) -> None:
    if len(tokens) != len(set(tokens)):
        raise ValueError("special tokens must be distinct")
    if any(not (x.startswith("<") and x.endswith(">")) for x in tokens):
        raise ValueError("invalid special token")
