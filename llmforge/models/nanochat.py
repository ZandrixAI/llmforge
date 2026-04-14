# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "nanochat"
    hidden_size: int = 1280
    num_hidden_layers: int = 20
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    vocab_size: int = 65536
    max_position_embeddings: int = 2048
    intermediate_size: int = 5120  # 4 * hidden_size
    rope_theta: float = 10000.0


def rms_norm(x):
    """Functional RMSNorm with no learnable parameters."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


def apply_rotary_emb(x, offset, base=10000.0, freqs=None):
    """Apply RoPE with blocked layout.

    Args:
        x: Input tensor in (B, H, T, D) format
        offset: Position offset for KV caching
        base: RoPE base frequency (default 10000.0)
        freqs: Precomputed negated frequencies (optional)

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    head_dim = x.shape[-1]

    if freqs is None:
        half_D = head_dim // 2
        freqs = -torch.exp(
            torch.arange(0.0, half_D, dtype=torch.float32, device=x.device)
            * (math.log(base) / half_D)
        )

    B, H, T, D = x.shape
    positions = torch.arange(offset, offset + T, dtype=torch.float32, device=x.device)
    angles = torch.outer(positions, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 : D]

    rx1 = x1 * cos - x2 * sin
    rx2 = x2 * cos + x1 * sin

    return torch.cat([rx1, rx2], dim=-1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.rope_theta = args.rope_theta

        self.c_q = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.c_k = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.c_v = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        half_D = self.head_dim // 2
        self._rope_freqs = -torch.exp(
            torch.arange(0.0, half_D, dtype=torch.float32)
            * (math.log(self.rope_theta) / half_D)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        queries = self.c_q(x)
        keys = self.c_k(x)
        values = self.c_v(x)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        offset = cache.offset if cache is not None else 0
        freqs = self._rope_freqs.to(x.device)
        queries = apply_rotary_emb(
            queries, offset=offset, base=self.rope_theta, freqs=freqs
        )
        keys = apply_rotary_emb(keys, offset=offset, base=self.rope_theta, freqs=freqs)

        queries = rms_norm(queries)
        keys = rms_norm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(1, 2).reshape(B, L, self.hidden_size)
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.c_fc = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.c_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x) ** 2
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = Attention(args)
        self.mlp = MLP(args)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        h = x + self.attn(rms_norm(x), mask=mask, cache=cache)
        out = h + self.mlp(rms_norm(h))
        return out


class NanoChatModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ) -> torch.Tensor:
        h = self.wte(inputs)
        h = rms_norm(h)

        if cache is None:
            cache = [None] * len(self.h)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.h, cache):
            h = layer(h, mask=mask, cache=c)

        h = rms_norm(h)

        return h


def softcap(logits, cap=15.0):
    return cap * torch.tanh(logits / cap)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = NanoChatModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ) -> torch.Tensor:
        out = self.transformer(inputs, cache=cache)
        logits = self.lm_head(out)

        logits = softcap(logits)

        return logits

    @property
    def layers(self):
        return self.transformer.h
