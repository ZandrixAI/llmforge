# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

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
    head_dim: int
    num_transformer_layers: int
    model_dim: int
    vocab_size: int
    ffn_dim_divisor: int
    num_query_heads: List
    num_kv_heads: List
    ffn_multipliers: List
    ffn_with_glu: bool = True
    normalize_qk_projections: bool = True
    share_input_output_layers: bool = True
    rms_norm_eps: float = 1e-6
    rope_freq_constant: float = 10000


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.head_dim = head_dim = args.head_dim
        self.layer_id = layer_id
        self.model_dim = model_dim = args.model_dim

        self.n_heads = n_heads = args.num_query_heads[layer_id]
        self.n_kv_heads = n_kv_heads = args.num_kv_heads[layer_id]
        self.scale = head_dim**-0.5

        op_size = (n_heads + (n_kv_heads * 2)) * head_dim
        self.qkv_proj = nn.Linear(model_dim, op_size, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, model_dim, bias=False)

        self.normalize_qk_projections = args.normalize_qk_projections

        if self.normalize_qk_projections:
            self.q_norm = RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            head_dim, args.rope_freq_constant, False, None, None
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(
            B, L, self.n_heads + (self.n_kv_heads * 2), self.head_dim
        ).transpose(1, 2)

        queries, keys, values = torch.split(
            qkv, [self.n_heads, self.n_kv_heads, self.n_kv_heads], dim=1
        )

        if self.normalize_qk_projections:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

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


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        dim = args.model_dim
        ffn_multiplier = args.ffn_multipliers[layer_id]

        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * args.model_dim,
                divisor=args.ffn_dim_divisor,
            )
        )

        self.proj_1 = nn.Linear(dim, 2 * intermediate_dim, bias=False)
        self.proj_2 = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        x = self.proj_1(x)
        gate, x = torch.split(x, x.shape[-1] // 2, dim=-1)
        return self.proj_2(swiglu(gate, x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        dim = args.model_dim
        self.attn = Attention(args, layer_id=layer_id)
        self.ffn = MLP(args, layer_id=layer_id)
        self.ffn_norm = RMSNorm(dim, eps=args.rms_norm_eps)
        self.attn_norm = RMSNorm(dim, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        r = self.attn(self.attn_norm(x), mask, cache)
        h = x + r
        r = self.ffn(self.ffn_norm(h))
        out = h + r
        return out


class OpenELMModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_transformer_layers = args.num_transformer_layers
        assert self.vocab_size > 0
        self.token_embeddings = nn.Embedding(args.vocab_size, args.model_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(args, layer_id=layer_id)
                for layer_id in range(self.num_transformer_layers)
            ]
        )
        self.norm = RMSNorm(args.model_dim, eps=args.rms_norm_eps)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        h = self.token_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = OpenELMModel(args)
        if not args.share_input_output_layers:
            self.lm_head = nn.Linear(args.model_dim, args.vocab_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        if self.args.share_input_output_layers:
            out = F.linear(out, self.transformer.token_embeddings.weight)
        else:
            out = self.lm_head(out)

        return out

    @property
    def layers(self):
        return self.transformer.layers
