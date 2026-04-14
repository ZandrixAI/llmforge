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
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    intermediate_size_mlp: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    no_rope_layers: list
    use_qk_norm: bool


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_rope):
        super().__init__()
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.use_rope = use_rope
        if use_rope:
            self.rope = initialize_rope(
                self.head_dim,
                args.rope_theta,
                traditional=True,
            )
        self.use_qk_norm = args.use_qk_norm
        self.rms_norm_eps = args.rms_norm_eps

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        if self.use_qk_norm:
            queries = queries * torch.rsqrt(
                queries.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps
            )
            keys = keys * torch.rsqrt(
                keys.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps
            )
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, intermediate_size, activation=nn.SiLU):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, use_rope):
        super().__init__()
        self.self_attn = Attention(args, use_rope)

        self.feed_forward = MLP(
            args.hidden_size,
            args.intermediate_size_mlp,
        )

        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.post_attention_layernorm(h))
        return h + r


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(args=args, use_rope=args.no_rope_layers[i])
                for i in range(args.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)

        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.output = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        h = self.model(inputs, cache)
        if self.tie_word_embeddings:
            return F.linear(h, self.model.embed_tokens.weight)
        else:
            return self.output(h)

    @property
    def layers(self):
        return self.model.layers
