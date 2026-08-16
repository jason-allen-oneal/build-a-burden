from __future__ import annotations

import dataclasses

import torch
import pytest

from ts_coder.model import ModelConfig, Transformer, count_parameters
from ts_coder.model import attention as attention_module
from ts_coder.model.rmsnorm import RMSNorm
from ts_coder.model.generation import generate
from ts_coder.model.rope import RotaryEmbedding, apply_rope, rotary_frequencies


def config(*, use_sdpa: bool = True) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        context_length=32,
        layers=2,
        hidden_size=32,
        attention_heads=4,
        kv_heads=2,
        ffn_size=64,
        use_sdpa=use_sdpa,
    )


def test_shapes_finite_backward_and_parameter_count() -> None:
    model = Transformer(config())
    ids = torch.randint(0, 64, (2, 8))
    output = model(ids, labels=ids)
    assert output.logits.shape == (2, 8, 64)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_weight_tying_and_deterministic_initialization() -> None:
    torch.manual_seed(17)
    first = Transformer(config())
    torch.manual_seed(17)
    second = Transformer(config())
    assert first.output.weight.data_ptr() == first.token_embedding.weight.data_ptr()
    assert all(
        torch.equal(a, b)
        for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True)
    )


def test_no_future_token_leakage() -> None:
    torch.manual_seed(1)
    model = Transformer(config()).eval()
    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = original.clone()
    changed[:, 4:] = torch.tensor([20, 21])
    with torch.no_grad():
        a, b = model(original).logits, model(changed).logits
    torch.testing.assert_close(a[:, :4], b[:, :4], atol=1e-6, rtol=1e-5)


def test_padding_and_loss_masks() -> None:
    model = Transformer(config()).eval()
    ids_a = torch.tensor([[1, 2, 3, 4, 5]])
    ids_b = torch.tensor([[1, 2, 3, 40, 41]])
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        a = model(ids_a, attention_mask=mask).logits
        b = model(ids_b, attention_mask=mask).logits
    torch.testing.assert_close(a[:, :3], b[:, :3])
    loss_a = model(ids_a, labels=ids_a, loss_mask=mask).loss
    altered_labels = ids_a.clone()
    altered_labels[:, 3:] = 33
    loss_b = model(ids_a, labels=altered_labels, loss_mask=mask).loss
    torch.testing.assert_close(loss_a, loss_b)


def test_kv_cache_matches_full_forward() -> None:
    model = Transformer(config()).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        full = model(ids).logits
        cache = None
        pieces = []
        for index in range(ids.shape[1]):
            output = model(ids[:, index : index + 1], cache=cache, use_cache=True)
            cache = output.cache
            pieces.append(output.logits)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=1e-5, rtol=1e-5)


def test_sdpa_gqa_matches_explicit_attention_fallback() -> None:
    torch.manual_seed(9)
    fast = Transformer(config()).eval()
    fallback = Transformer(config(use_sdpa=False)).eval()
    fallback.load_state_dict(fast.state_dict())
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        fast_logits = fast(ids).logits
        fallback_logits = fallback(ids).logits
    torch.testing.assert_close(fast_logits, fallback_logits, atol=1e-5, rtol=1e-5)


def test_sdpa_cached_gqa_matches_explicit_attention_fallback() -> None:
    torch.manual_seed(19)
    fast = Transformer(config()).eval()
    fallback = Transformer(config(use_sdpa=False)).eval()
    fallback.load_state_dict(fast.state_dict())
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        fast_cache = None
        fallback_cache = None
        fast_pieces = []
        fallback_pieces = []
        for index in range(ids.shape[1]):
            fast_output = fast(ids[:, index : index + 1], cache=fast_cache, use_cache=True)
            fallback_output = fallback(
                ids[:, index : index + 1], cache=fallback_cache, use_cache=True
            )
            fast_cache = fast_output.cache
            fallback_cache = fallback_output.cache
            fast_pieces.append(fast_output.logits)
            fallback_pieces.append(fallback_output.logits)
    torch.testing.assert_close(
        torch.cat(fast_pieces, dim=1), torch.cat(fallback_pieces, dim=1), atol=1e-5, rtol=1e-5
    )


def test_sdpa_gqa_api_fallback_matches_explicit(monkeypatch) -> None:
    """Old PyTorch/API backends must retain a correct materialized fallback."""
    torch.manual_seed(23)
    fast = Transformer(config()).eval()
    fallback = Transformer(config(use_sdpa=False)).eval()
    fallback.load_state_dict(fast.state_dict())
    ids = torch.tensor([[1, 2, 3, 4]])
    monkeypatch.setattr(attention_module, "_SDPA_GQA_AVAILABLE", False)
    with torch.no_grad():
        fast_logits = fast(ids).logits
        fallback_logits = fallback(ids).logits
    torch.testing.assert_close(fast_logits, fallback_logits, atol=1e-5, rtol=1e-5)


def test_rope_cache_is_shared_and_device_dtype_aware() -> None:
    model = Transformer(config()).eval()
    positions = torch.arange(6)
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3, 4, 5, 6]]))
    assert model.rotary_embedding.cache_entries == 1
    cos_f32, sin_f32 = model.rotary_embedding(positions, dtype=torch.float32)
    cos_bf16, sin_bf16 = model.rotary_embedding(positions, dtype=torch.bfloat16)
    assert model.rotary_embedding.cache_entries == 2
    assert cos_f32.dtype is torch.float32 and sin_f32.dtype is torch.float32
    assert cos_bf16.dtype is torch.bfloat16 and sin_bf16.dtype is torch.bfloat16
    torch.testing.assert_close(cos_f32, cos_bf16.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(sin_f32, sin_bf16.float(), atol=5e-3, rtol=5e-3)


def test_rope_cache_has_no_checkpoint_state() -> None:
    rope = RotaryEmbedding(8)
    rope(torch.arange(4), dtype=torch.float32)
    assert all("rotary" not in key and "inv_freq" not in key for key in rope.state_dict())


def test_gradient_checkpointing_matches_regular_training_forward() -> None:
    torch.manual_seed(42)
    regular_config = config(use_sdpa=False)
    checkpoint_config = dataclasses.replace(regular_config, gradient_checkpointing=True)
    regular = Transformer(regular_config).train()
    checkpointed = Transformer(checkpoint_config).train()
    checkpointed.load_state_dict(regular.state_dict())
    ids = torch.randint(0, regular_config.vocab_size, (2, 12))
    regular_loss = regular(ids, labels=ids).loss
    checkpoint_loss = checkpointed(ids, labels=ids).loss
    assert regular_loss is not None and checkpoint_loss is not None
    regular_loss.backward()
    checkpoint_loss.backward()
    torch.testing.assert_close(regular_loss, checkpoint_loss, rtol=1e-5, atol=1e-5)
    for left, right in zip(regular.parameters(), checkpointed.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)


def test_rmsnorm_and_rope() -> None:
    x = torch.randn(2, 3, 4, 8)
    normed = RMSNorm(8)(x)
    assert normed.shape == x.shape
    cos, sin = rotary_frequencies(8, torch.arange(4))
    rotated = apply_rope(x, cos, sin)
    assert rotated.shape == x.shape
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1), atol=1e-5, rtol=1e-5)


def test_save_load_equivalence(tmp_path) -> None:
    model = Transformer(config()).eval()
    ids = torch.tensor([[1, 2, 3]])
    expected = model(ids).logits.detach()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = Transformer(config()).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(expected, restored(ids).logits)


def test_generation_rejects_empty_prompt() -> None:
    model = Transformer(config()).eval()
    with pytest.raises(ValueError, match="at least one prompt token"):
        generate(model, torch.empty((1, 0), dtype=torch.long), max_new_tokens=1)
