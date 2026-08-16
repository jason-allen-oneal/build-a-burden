from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as checkpoint_block

from .attention import KVCache
from .block import TransformerBlock
from .config import ModelConfig
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    cache: list[KVCache] | None = None


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        head_dim = config.hidden_size // config.attention_heads
        # RoPE is position-only and identical across layers. Keeping one lazy
        # device/dtype-aware cache avoids rebuilding the same table per block.
        self.rotary_embedding = RotaryEmbedding(head_dim, config.rope_theta)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.final_norm = RMSNorm(config.hidden_size, getattr(config, "rms_norm_eps", 1e-6))
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tied_embeddings:
            self.output.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=getattr(self.config, "initializer_range", 0.02),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        cache: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> TransformerOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        past_len = 0 if cache is None else cache[0][0].shape[2]
        if past_len + input_ids.shape[1] > self.config.context_length:
            raise ValueError("sequence exceeds context length")
        x = self.token_embedding(input_ids)
        positions = torch.arange(
            past_len,
            past_len + input_ids.shape[1],
            device=input_ids.device,
        )
        rotary_values = self.rotary_embedding(positions, dtype=x.dtype)
        new_caches: list[KVCache] = []
        for index, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[index]
            # Checkpointing is only valid for the training path without a KV
            # cache. Capturing the mask in the closure keeps the block API
            # unchanged while dropping intermediate activations between layers.
            if (
                self.training
                and self.config.gradient_checkpointing
                and cache is None
                and not use_cache
            ):
                x = checkpoint_block(
                    lambda hidden, current_block=block: current_block(
                        hidden, attention_mask, None, False, rotary_values
                    )[0],
                    x,
                    use_reentrant=False,
                )
                new_cache = None
            else:
                x, new_cache = block(
                    x,
                    attention_mask,
                    layer_cache,
                    use_cache,
                    rotary_values,
                )
            if new_cache is not None:
                new_caches.append(new_cache)
        logits = self.output(self.final_norm(x)).float()
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids")
            if labels.shape[1] < 2:
                raise ValueError("causal language-model loss requires at least two tokens")
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            token_loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shifted_labels)
            valid = shifted_labels.ne(-100)
            if loss_mask is not None:
                if loss_mask.shape != labels.shape:
                    raise ValueError("loss_mask must match labels")
                valid &= loss_mask[:, 1:].bool()
            denominator = valid.sum().clamp_min(1)
            loss = (token_loss * valid).sum() / denominator
        return TransformerOutput(logits, loss, new_caches if use_cache else None)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
