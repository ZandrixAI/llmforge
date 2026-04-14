# Copyright © 2023-2024 Apple Inc.

import sys
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask
from .rope_utils import initialize_rope

try:
    import hf_olmo
except ImportError:
    print("To run olmo install ai2-olmo: pip install ai2-olmo")
    sys.exit(1)


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
    d_model: int
    n_layers: int
    mlp_hidden_size: int
    n_heads: int
    vocab_size: int
    embedding_size: int
    rope_theta: float = 10000
    rope_traditional: bool = False
    mlp_ratio: int = 4
    weight_tying: bool = False

    def __post_init__(self):
        self.mlp_hidden_size = (
            self.mlp_hidden_size
            if self.mlp_hidden_size is not None
            else self.mlp_ratio * self.d_model
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        dim = args.d_model

        self.ff_proj = nn.Linear(dim, args.mlp_hidden_size, bias=False)
        self.ff_out = nn.Linear(args.mlp_hidden_size // 2, dim, bias=False)

        self.att_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False)

        head_dim = dim // self.n_heads
        self.scale = head_dim**-0.5

        self.att_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=None,
            max_position_embeddings=None,
        )

        self.args = args

    def attend(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        queries, keys, values = torch.split(self.att_proj(x), D, dim=-1)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(1, 2)
        keys = keys.reshape(B, L, self.n_heads, -1).transpose(1, 2)
        values = values.reshape(B, L, self.n_heads, -1).transpose(1, 2)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(-1, -2)
        if mask is not None:
            scores += mask
        scores = torch.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)
        output = (scores @ values).transpose(1, 2).reshape(B, L, -1)
        return self.attn_out(output)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        r = self.attend(self.att_norm(x), mask, cache)
        h = x + r

        x1, x2 = torch.split(
            self.ff_proj(self.ff_norm(h)), self.ff_proj.out_features // 2, dim=-1
        )

        out = h + self.ff_out(swiglu(x2, x1))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_layers = args.n_layers
        self.weight_tying = args.weight_tying

        self.wte = nn.Embedding(args.embedding_size, args.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.n_layers)]
        )
        if not self.weight_tying:
            self.ff_out = nn.Linear(args.d_model, args.embedding_size, bias=False)
        self.norm = nn.LayerNorm(args.d_model, elementwise_affine=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        h = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.blocks)

        mask = create_attention_mask(h, cache[0])

        for block, c in zip(self.blocks, cache):
            h = block(h, mask, c)

        h = self.norm(h)

        if self.weight_tying:
            return F.linear(h, self.wte.weight), cache

        return self.ff_out(h)


class OlmoModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.transformer = Transformer(args)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        return self.transformer(inputs, cache)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = OlmoModel(args)
        self.args = args

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        return self.model(inputs, cache)

    @property
    def layers(self):
        return self.model.transformer.blocks
