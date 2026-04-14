# Copyright © 2023-2024 Apple Inc.

import inspect
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchMLP


@dataclass
class ModelArgs:
    model_type: str
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32
    num_experts_per_tok: int = 2
    num_local_experts: int = 4

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class RoPEAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int, rotary_dim: int):
        super().__init__()

        self.num_heads = num_heads

        self.Wqkv = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def forward(self, x, mask=None, cache=None):
        qkv = self.Wqkv(x)
        queries, keys, values = torch.split(qkv, qkv.shape[-1] // 3, dim=-1)

        # Extract some shapes
        num_heads = self.num_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, num_heads, -1).transpose(1, 2)
        keys = keys.reshape(B, L, num_heads, -1).transpose(1, 2)
        values = values.reshape(B, L, num_heads, -1).transpose(1, 2)

        queries = queries.to(torch.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])

        output = scaled_dot_product_attention(
            queries.to(torch.float32),
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        ).to(values.dtype)
        output = output.transpose(1, 2).reshape(B, L, -1)

        return self.out_proj(output)


class MOE(nn.Module):
    def __init__(self, args: ModelArgs, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchMLP(
            self.dim, self.hidden_dim, self.num_experts, bias=True
        )
        self.gate = nn.Linear(args.model_dim, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.gate(x)

        k = self.num_experts_per_tok
        inds = torch.topk(-gates, k=k, dim=-1).indices.detach()
        scores = torch.gather(gates, -1, inds)
        scores = torch.softmax(scores, dim=-1)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(dim=-2)

        return y


class ParallelBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        dims = config.model_dim
        mlp_dims = dims * 4
        self.mixer = RoPEAttention(dims, config.num_heads, config.rotary_dim)
        self.ln = nn.LayerNorm(dims)
        self.moe = MOE(config, dims, mlp_dims)

    def forward(self, x, mask, cache):
        h = self.ln(x)
        attn_h = self.mixer(h, mask, cache)
        ff_h = self.moe(h)
        return attn_h + ff_h + x


class TransformerDecoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embd = Embd(config)
        self.h = nn.ModuleList(
            [ParallelBlock(config) for i in range(config.num_layers)]
        )

    def forward(self, x, mask, cache):
        x = self.embd(x)
        if cache is None:
            cache = [None] * len(self.h)

        for layer, c in zip(self.h, cache):
            x = layer(x, mask, c)
        return x


class Embd(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(config.num_vocab, config.model_dim)

    def forward(self, x):
        return self.wte(x)


class OutputHead(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(config.model_dim)
        self.linear = nn.Linear(config.model_dim, config.num_vocab)

    def forward(self, inputs):
        return self.linear(self.ln(inputs))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        self.transformer = TransformerDecoder(config)
        self.lm_head = OutputHead(config)
        self.args = config

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        cache=None,
    ) -> torch.Tensor:

        if mask is None:
            mask = create_attention_mask(x, cache)

        y = self.transformer(x, mask, cache)
        return self.lm_head(y)

    def sanitize(self, weights):
        if "transformer.h.0.moe.mlp.0.fc1.weight" not in weights:
            return weights
        for l in range(self.args.num_layers):
            prefix = f"transformer.h.{l}"
            for n in ["fc1", "fc2"]:
                for k in ["weight", "scales", "biases", "bias"]:
                    if f"{prefix}.moe.mlp.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.moe.mlp.{e}.{n}.{k}")
                            for e in range(self.args.num_local_experts)
                        ]
                        weights[f"{prefix}.moe.switch_mlp.{n}.{k}"] = torch.stack(
                            to_join
                        )
        return weights

    @property
    def layers(self):
        return self.transformer.h
