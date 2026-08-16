from __future__ import annotations

import torch

from .transformer import Transformer


def _apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor, penalty: float) -> None:
    """Apply the standard sign-preserving repetition penalty in place.

    ``history`` includes the prompt and all tokens generated so far.  This is
    intentionally an inference-time control, not a change to the model or its
    evaluation loss.  Keeping the operation here makes callers able to report
    the exact decoding policy used for a sample.
    """
    if penalty == 1.0:
        return
    for batch_index in range(history.shape[0]):
        seen = history[batch_index].unique()
        values = logits[batch_index, seen]
        logits[batch_index, seen] = torch.where(
            values < 0,
            values * penalty,
            values / penalty,
        )


def _mask_repeated_ngrams(logits: torch.Tensor, history: torch.Tensor, ngram_size: int) -> None:
    """Ban tokens that would repeat an already-emitted n-gram.

    If every vocabulary entry is banned for a row, the highest unmasked
    candidate is retained.  That fallback keeps sampling finite for tiny
    vocabularies and adversarial prompts; it does not pretend the constraint
    can always be satisfied.
    """
    if ngram_size <= 0:
        return
    vocabulary_size = logits.shape[-1]
    for batch_index in range(history.shape[0]):
        tokens = history[batch_index].tolist()
        if len(tokens) + 1 < ngram_size:
            continue
        prefix = tuple(tokens[-(ngram_size - 1) :]) if ngram_size > 1 else ()
        banned: set[int] = set()
        for start in range(len(tokens) - ngram_size + 1):
            if tuple(tokens[start : start + ngram_size - 1]) == prefix:
                banned.add(tokens[start + ngram_size - 1])
        banned = {token for token in banned if 0 <= token < vocabulary_size}
        if not banned:
            continue
        row = logits[batch_index]
        finite = torch.isfinite(row)
        keep = int(row.masked_fill(~finite, float("-inf")).argmax())
        keep_value = row[keep].clone()
        row[list(banned)] = float("-inf")
        if not torch.isfinite(row).any():
            row[keep] = keep_value if torch.isfinite(keep_value) else 0.0


@torch.inference_mode()
def generate(
    model: Transformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_ids: set[int] | None = None,
    generator: torch.Generator | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one prompt token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative")
    if temperature < 0:
        raise ValueError("temperature must not be negative")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must not be negative")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must not be negative")
    model.eval()
    result = input_ids
    cache = None
    current = input_ids
    for _ in range(max_new_tokens):
        output = model(current, cache=cache, use_cache=True)
        cache = output.cache
        logits = output.logits[:, -1]
        _apply_repetition_penalty(logits, result, repetition_penalty)
        _mask_repeated_ngrams(logits, result, no_repeat_ngram_size)
        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                threshold = logits.topk(min(top_k, logits.shape[-1])).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            if top_p is not None:
                sorted_logits, indices = logits.sort(descending=True)
                cumulative = sorted_logits.softmax(-1).cumsum(-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                logits.scatter_(1, indices, sorted_logits.masked_fill(remove, float("-inf")))
            next_token = torch.multinomial(logits.softmax(-1), 1, generator=generator)
        result = torch.cat((result, next_token), dim=1)
        if stop_ids and all(token.item() in stop_ids for token in next_token):
            break
        current = next_token
    return result
