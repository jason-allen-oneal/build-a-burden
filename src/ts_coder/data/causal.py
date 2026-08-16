def causal_example(token_ids: list[int], context_length: int, pad_id: int) -> dict[str, list[int]]:
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    inputs = token_ids[:context_length]
    labels = inputs.copy()
    wanted = context_length - len(inputs)
    inputs += [pad_id] * wanted
    labels += [pad_id] * wanted
    return {
        "input_ids": inputs,
        "labels": labels,
        "attention_mask": [int(x != pad_id) for x in inputs],
        "loss_mask": [int(x != pad_id) for x in labels],
    }


def causal_examples(
    token_ids: list[int], context_length: int, pad_id: int
) -> list[dict[str, list[int]]]:
    """Cover a complete token stream with prediction windows."""
    if len(token_ids) < 2:
        return []
    return [
        causal_example(token_ids[start : start + context_length], context_length, pad_id)
        for start in range(0, len(token_ids) - 1, context_length - 1)
    ]
