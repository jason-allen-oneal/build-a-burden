import torch

from ts_coder.model import ModelConfig, Transformer
from ts_coder.model.generation import generate


def test_cached_and_uncached_greedy_generation_match() -> None:
    torch.manual_seed(7)
    model = Transformer(
        ModelConfig(
            vocab_size=32,
            context_length=16,
            layers=2,
            hidden_size=16,
            attention_heads=2,
            kv_heads=1,
            ffn_size=32,
        )
    )
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    cached = generate(model, prompt, max_new_tokens=4, use_kv_cache=True)
    uncached = generate(model, prompt, max_new_tokens=4, use_kv_cache=False)

    assert torch.equal(cached, uncached)
