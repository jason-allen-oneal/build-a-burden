from types import SimpleNamespace

import pytest
import torch

from ts_coder.model.generation import generate


class FixedLogitsModel:
    """Small deterministic model double for decoding-policy tests."""

    def __init__(self, logits: list[float]) -> None:
        self.next_logits = torch.tensor(logits, dtype=torch.float32)

    def eval(self) -> "FixedLogitsModel":
        return self

    def __call__(self, input_ids, cache=None, use_cache=False):
        batch, length = input_ids.shape
        logits = self.next_logits.expand(batch, length, -1).clone()
        return SimpleNamespace(logits=logits, cache=None)


def test_repetition_penalty_can_change_greedy_choice() -> None:
    model = FixedLogitsModel([0.0, 0.0, 10.0, 9.0])
    prompt = torch.tensor([[2]])

    unpenalized = generate(model, prompt, max_new_tokens=1)
    penalized = generate(model, prompt, max_new_tokens=1, repetition_penalty=2.0)
    baseline_sampler = generate(
        model,
        prompt,
        max_new_tokens=1,
        temperature=1.0,
        generator=torch.Generator().manual_seed(7),
    )
    disabled_top_k = generate(
        model,
        prompt,
        max_new_tokens=1,
        temperature=1.0,
        top_k=0,
        generator=torch.Generator().manual_seed(7),
    )

    assert unpenalized.tolist() == [[2, 2]]
    assert penalized.tolist() == [[2, 3]]
    assert disabled_top_k.tolist() == baseline_sampler.tolist()


def test_no_repeat_ngram_masks_a_repeated_bigram() -> None:
    model = FixedLogitsModel([0.0, 0.0, 10.0, 9.0])
    prompt = torch.tensor([[1, 2, 1]])

    output = generate(model, prompt, max_new_tokens=1, no_repeat_ngram_size=2)

    assert output.tolist() == [[1, 2, 1, 3]]


def test_no_repeat_fallback_keeps_logits_finite_when_all_tokens_are_seen() -> None:
    model = FixedLogitsModel([10.0, 9.0])
    prompt = torch.tensor([[0, 1]])

    output = generate(model, prompt, max_new_tokens=1, no_repeat_ngram_size=1)

    assert output.tolist() == [[0, 1, 0]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_new_tokens": 1, "temperature": -0.1}, "temperature"),
        ({"max_new_tokens": 1, "top_k": -1}, "top_k"),
        ({"max_new_tokens": 1, "top_p": 0.0}, "top_p"),
        ({"max_new_tokens": 1, "repetition_penalty": 0.0}, "repetition_penalty"),
        ({"max_new_tokens": 1, "no_repeat_ngram_size": -1}, "no_repeat_ngram_size"),
    ],
)
def test_generation_rejects_invalid_controls(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate(FixedLogitsModel([1.0, 0.0]), torch.tensor([[0]]), **kwargs)
