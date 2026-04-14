# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _compute_gate(
    query: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    gate_logits = query @ weight[:, None, :].transpose(-1, -2)
    gate_logits = gate_logits + bias[..., None, None]
    return torch.sigmoid(gate_logits)


def _silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def _mix_attention(
    gate: torch.Tensor, attn_global: torch.Tensor, attn_local: torch.Tensor
) -> torch.Tensor:
    return gate * attn_global + (1 - gate) * attn_local


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: int
    num_key_value_heads: int
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    loop_num: int = 2
    loop_window_size: int = 64


class LoopGateProjection(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.zeros((num_heads, head_dim)))
        self.bias = nn.Parameter(torch.zeros((num_heads,)))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return _compute_gate(query, self.weight, self.bias)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def get_qkv(
        self, x: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(1, 2)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        return queries, keys, values

    def attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_silu_mul(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


class IQuestLoopCoderModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.loop_num == 2, f"Only loop_num=2 is supported, got {args.loop_num}"
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        )
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.gate_projections = nn.ModuleList(
            [
                LoopGateProjection(args.num_attention_heads, args.head_dim)
                for _ in range(args.num_hidden_layers)
            ]
        )
        self.loop_num = args.loop_num
        self.loop_window_size = args.loop_window_size

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[List[Any]] = None,
    ):
        B, L = inputs.shape[:2]
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * (2 * len(self.layers))

        mask = create_attention_mask(h, cache[0])
        window_mask = create_attention_mask(
            h, cache[len(self.layers)], window_size=self.loop_window_size
        )

        loop1_kv = []
        for layer, c in zip(self.layers, cache):
            h_norm = layer.input_layernorm(h)
            offset = c.offset if c is not None else 0
            q1, k1, v1 = layer.self_attn.get_qkv(h_norm, offset)

            if c is not None:
                k1, v1 = c.update_and_fetch(k1, v1)
            loop1_kv.append((k1, v1))

            out = layer.self_attn.attention(q1, k1, v1, mask, cache=c)
            r = layer.self_attn.o_proj(out.transpose(1, 2).reshape(B, L, -1))
            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        for layer, gate_proj, c, (k1, v1) in zip(
            self.layers, self.gate_projections, cache[len(self.layers) :], loop1_kv
        ):
            h_norm = layer.input_layernorm(h)
            offset = c.offset if c is not None else 0
            q2, k2, v2 = layer.self_attn.get_qkv(h_norm, offset)
            gate = gate_proj(q2)
            attn_global = layer.self_attn.attention(q2, k1, v1, mask, cache=c)

            if c is not None:
                k2, v2 = c.update_and_fetch(k2, v2)
            attn_local = layer.self_attn.attention(
                q2,
                k2,
                v2,
                window_mask,
                cache=c,
            )

            mixed = _mix_attention(gate, attn_global, attn_local)
            r = layer.self_attn.o_proj(mixed.transpose(1, 2).reshape(B, L, -1))
            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = IQuestLoopCoderModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = F.linear(out, self.model.embed_tokens.weight)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers] + [
            RotatingKVCache(max_size=self.args.loop_window_size) for _ in self.layers
        ]
