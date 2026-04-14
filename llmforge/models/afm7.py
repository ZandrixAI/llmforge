# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ConcatenateKVCache, KVCache
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
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_kv_reuse_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_dim_scale_factor: float = 3.25
    rope_theta: float = 50000
    rms_norm_eps: float = 1e-5


class FusedLinear(nn.Linear):
    def __init__(self, input_dims, output_dims):
        *indices, output_dims = accumulate(output_dims)
        self.indices = indices
        super().__init__(input_dims, output_dims, bias=False)

    @property
    def input_dims(self):
        return self.weight.shape[-1]

    @property
    def output_dims(self):
        indices = [0] + self.indices + [self.weight.shape[0]]
        return [indices[i] - indices[i - 1] for i in range(1, len(indices))]

    def forward(self, x):
        x = super().forward(x)
        return torch.split(x, self.indices, dim=-1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_dim
        self.n_heads = n_heads = args.num_heads
        self.n_kv_heads = n_kv_heads = args.num_kv_heads
        self.head_dim = head_dim = args.hidden_dim // n_heads
        self.scale = head_dim**-0.5

        self.qkv_proj = FusedLinear(
            dim, [n_heads * head_dim] + 2 * [n_kv_heads * head_dim]
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            True,
        )
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.quant_key_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.quant_value_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        queries, keys, values = self.qkv_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(1, 2)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)

        if cache is not None:
            queries = self.q_norm(self.rope(queries, offset=cache.offset))
            keys = self.k_norm(self.rope(keys, offset=cache.offset))
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.q_norm(self.rope(queries))
            keys = self.k_norm(self.rope(keys))

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(output)


class KVReuseAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_dim
        self.n_heads = n_heads = args.num_heads
        self.head_dim = head_dim = args.hidden_dim // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            True,
        )
        self.q_norm = RMSNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        _, _, S, _ = keys.shape

        queries = self.q_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(1, 2)
        queries = self.q_norm(self.rope(queries, offset=S - L))

        output = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=self.scale, mask=mask
        )

        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_dim
        hidden_dim = int(dim * args.hidden_dim_scale_factor)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        g = self.gate_proj(x)
        x = self.up_proj(x)
        return self.down_proj(swiglu(g, x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class KVReuseTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = KVReuseAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), keys, values, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class AFMModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        self.embedding = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(args)
                for _ in range(args.num_layers - args.num_kv_reuse_layers)
            ]
        )
        self.kv_reuse_layers = nn.ModuleList(
            [KVReuseTransformerBlock(args) for _ in range(args.num_kv_reuse_layers)]
        )
        self.output_norm = RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        h = self.embedding(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
            cache[-1] = ConcatenateKVCache()

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        keys, values = cache[-1].state
        for layer in self.kv_reuse_layers:
            h = layer(h, keys, values, mask)

        return self.output_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AFMModel(args)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = F.linear(out, self.model.embedding.weight)
        return out

    def make_cache(self):
        return [KVCache() for _ in range(len(self.model.layers))]

    @property
    def layers(self):
        return self.model.layers + self.model.kv_reuse_layers
