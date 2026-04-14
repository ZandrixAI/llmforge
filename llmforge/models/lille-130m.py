# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    block_size: int
    layer_norm_eps: float
    n_embd: int
    n_head: int
    n_kv_heads: int
    n_layer: int
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool = True


class Lille130mAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.n_head
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.n_embd // args.n_head
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            args.n_embd, (args.n_head + 2 * args.n_kv_heads) * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(args.n_head * self.head_dim, args.n_embd, bias=False)

        self.norm = RMSNorm(args.n_embd, eps=args.layer_norm_eps)

        self.rope = initialize_rope(
            args.n_embd // args.n_head, args.rope_theta, traditional=True
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv_proj(self.norm(x))

        q_size = self.n_head * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        queries, keys, values = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        queries = queries.reshape(B, L, self.n_head, -1).transpose(1, 2)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(output)


class Lille130mMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 256 * round(int(8 * args.n_embd / 3) / 256)

        self.norm = RMSNorm(args.n_embd, eps=args.layer_norm_eps)
        self.gate_proj = nn.Linear(args.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(args.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, args.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.down_proj(swiglu(self.gate_proj(h), self.up_proj(h)))


class Lille130Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Lille130mAttention(args)
        self.feed_forward = Lille130mMLP(args)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        h = x + self.attention(x, mask, cache)
        out = h + self.feed_forward(h)
        return out


class Lille130(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.n_embd)
        self.layers = nn.ModuleList(
            [Lille130Block(args=args) for _ in range(args.n_layer)]
        )
        self.norm = RMSNorm(args.n_embd, eps=args.layer_norm_eps)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return F.linear(self.norm(h), self.tok_embeddings.weight)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = Lille130(args)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        return self.transformer(inputs, cache=cache)

    @property
    def layers(self):
        return self.transformer.layers

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if "rotary_emb" not in k}
