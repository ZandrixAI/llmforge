# Copyright © 2023-2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ArraysCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    attention_bias: bool
    conv1d_width: int
    hidden_size: int
    intermediate_size: int
    logits_soft_cap: float
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    attention_window_size: int
    vocab_size: int
    embeddings_scale_by_sqrt_dim: bool = True
    block_types: Optional[List[str]] = None
    _block_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.block_types is None:
            self.block_types = self._block_types


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dims))
        self.eps = eps

    def forward(self, x):
        return torch.ops.aten.rmsnorm(x, 1.0 + self.weight, self.eps)


def rnn_scan(x, a, h0):
    assert x.ndim == 3
    assert a.shape == x.shape[-a.ndim :]
    assert a.dtype == x.dtype

    if x.shape[1] == 1:
        if h0 is None:
            return x, x[:, 0]
        else:
            y = a * h0[:, None] + x
            return y, y[:, -1]
    else:
        if h0 is not None:
            h_t = h0
        else:
            B, _, D = x.shape
            h_t = torch.zeros((B, D), dtype=x.dtype, device=x.device)

        y = torch.zeros_like(x)
        for t in range(x.shape[1]):
            h_t = a[:, t] * h_t + x[:, t]
            y[:, t] = h_t

    return y, h_t


class Conv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((channels, kernel_size, 1)))
        self.bias = nn.Parameter(torch.zeros((channels,)))

    def forward(self, x, cache=None):
        B, L, C = x.shape
        groups, K, _ = self.weight.shape

        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        else:
            x = F.pad(x, (0, 0, K - 1, 0))

        x_t = x.transpose(1, 2)
        w_t = self.weight.transpose(-1, -2)
        y = F.conv1d(x_t, w_t, groups=groups)
        y = y.transpose(1, 2)
        y = y + self.bias

        return y, x[:, -K + 1 :, :]


class RGLRU(nn.Module):
    """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

    def __init__(
        self,
        width: int,
        num_heads: int,
    ):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.head_dim = self.width // self.num_heads

        self.recurrent_param = nn.Parameter(torch.zeros((self.width,)))

        self.input_gate_weight = nn.Parameter(
            torch.zeros(
                (self.num_heads, self.head_dim, self.head_dim),
            )
        )
        self.input_gate_bias = nn.Parameter(
            torch.zeros((self.num_heads, self.head_dim))
        )

        self.recurrent_gate_weight = nn.Parameter(
            torch.zeros(
                (self.num_heads, self.head_dim, self.head_dim),
            )
        )
        self.recurrent_gate_bias = nn.Parameter(
            torch.zeros((self.num_heads, self.head_dim))
        )

    def forward(
        self,
        x: torch.Tensor,
        cache=None,
    ):
        B, L, _ = x.shape

        def apply_block_linear(h, w, b):
            h = h.reshape((B, L, self.num_heads, self.head_dim))
            h = (h.transpose(1, 2) @ w).transpose(1, 2) + b
            return torch.sigmoid(h.flatten(2, 3))

        gate_x = apply_block_linear(x, self.input_gate_weight, self.input_gate_bias)
        gate_a = apply_block_linear(
            x, self.recurrent_gate_weight, self.recurrent_gate_bias
        )

        log_a = -8.0 * gate_a * F.softplus(self.recurrent_param)
        a = torch.exp(log_a)
        a_square = torch.exp(2 * log_a)

        gated_x = x * gate_x

        multiplier = torch.sqrt(1 - a_square)
        if cache is None:
            multiplier[:, 0, :] = 1.0
        normalized_x = gated_x * multiplier.to(x.dtype)

        y, last_h = rnn_scan(
            x=normalized_x,
            a=a,
            h0=cache,
        )

        return y, last_h


class RecurrentBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        lru_width: int = None,
        conv1d_temporal_width: int = 4,
    ):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.lru_width = lru_width or width
        self.conv1d_temporal_width = conv1d_temporal_width

        self.linear_y = nn.Linear(width, self.lru_width)
        self.linear_x = nn.Linear(width, self.lru_width)
        self.linear_out = nn.Linear(self.lru_width, width)
        self.conv_1d = Conv1d(
            channels=self.lru_width,
            kernel_size=self.conv1d_temporal_width,
        )
        self.rg_lru = RGLRU(
            width=self.lru_width,
            num_heads=self.num_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache=None,
        mask=None,
    ):
        y = self.linear_y(x)
        y = F.gelu(y)

        x = self.linear_x(x)
        if cache is None:
            cache = [None, None]
        x, cache[0] = self.conv_1d(x=x, cache=cache[0])
        x, cache[1] = self.rg_lru(x=x, cache=cache[1])

        x = x * y
        x = self.linear_out(x)

        return x


class LocalAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        window_size: int,
    ):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (width // num_heads) ** (-0.5)

        self.head_dim = self.width // self.num_heads
        self.q_proj = nn.Linear(self.width, self.width, bias=False)
        self.k_proj = nn.Linear(self.width, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.width, bias=True)
        self.rope = RoPE(
            self.head_dim // 2,
            traditional=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache=None,
        mask=None,
    ):
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        keys = keys.reshape(B, L, 1, -1).transpose(1, 2)
        values = values.reshape(B, L, 1, -1).transpose(1, 2)

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
        return self.o_proj(output)


class MLPBlock(nn.Module):
    def __init__(self, width: int, expanded_width: int):
        super().__init__()
        self.up_proj = nn.Linear(width, expanded_width // 2)
        self.gate_proj = nn.Linear(width, expanded_width // 2)
        self.down_proj = nn.Linear(expanded_width // 2, width)

    def forward(self, x: torch.Tensor):
        gate = self.gate_proj(x)
        x = self.up_proj(x)
        return self.down_proj(F.gelu(gate) * x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        width: int,
        mlp_expanded_width: int,
        num_heads: int,
        attention_window_size: int,
        temporal_block_type: str,
        lru_width: Optional[int] = None,
        conv1d_temporal_width: int = 4,
    ):
        super().__init__()
        self.width = width
        self.mlp_expanded_width = mlp_expanded_width
        self.num_heads = num_heads
        self.attention_window_size = attention_window_size
        self.temporal_block_type = temporal_block_type
        self.lru_width = lru_width
        self.conv1d_temporal_width = conv1d_temporal_width

        self.temporal_pre_norm = RMSNorm(width)
        if self.temporal_block_type == "recurrent":
            self.temporal_block = RecurrentBlock(
                width=self.width,
                num_heads=self.num_heads,
                lru_width=self.lru_width,
                conv1d_temporal_width=self.conv1d_temporal_width,
            )
        else:
            self.temporal_block = LocalAttentionBlock(
                width=self.width,
                num_heads=self.num_heads,
                window_size=self.attention_window_size,
            )

        self.channel_pre_norm = RMSNorm(width)
        self.mlp_block = MLPBlock(
            width=self.width,
            expanded_width=self.mlp_expanded_width,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache=None,
        mask=None,
    ):
        raw_x = x

        inputs_normalized = self.temporal_pre_norm(raw_x)

        x = self.temporal_block(inputs_normalized, cache=cache, mask=mask)
        residual = x + raw_x

        x = self.channel_pre_norm(residual)
        x = self.mlp_block(x)

        x = x + residual

        return x


class Griffin(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.scale_by_sqrt_dim = config.embeddings_scale_by_sqrt_dim
        block_types = config.block_types

        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    width=config.hidden_size,
                    mlp_expanded_width=config.intermediate_size,
                    num_heads=config.num_attention_heads,
                    attention_window_size=config.attention_window_size,
                    temporal_block_type=block_types[i % len(block_types)],
                    lru_width=None,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.window_size = config.attention_window_size
        self.swa_idx = block_types.index("attention")

    def forward(
        self,
        tokens,
        cache=None,
    ):
        x = self.embed_tokens(tokens)
        if self.scale_by_sqrt_dim:
            x = x * math.sqrt(x.shape[-1])

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(
            x, cache[self.swa_idx], window_size=self.window_size
        )

        for i, block in enumerate(self.layers):
            x = block(x, mask=mask, cache=cache[i])

        return self.final_norm(x)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.args = config
        self.model = Griffin(config)
        self.model_type = config.model_type
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, cache=None) -> torch.Tensor:
        logits = self.model(tokens, cache=cache)
        if hasattr(self, "lm_head"):
            logits = self.lm_head(logits)
        else:
            logits = F.linear(logits, self.model.embed_tokens.weight)

        c = self.args.logits_soft_cap
        if c:
            logits = torch.tanh(logits / c) * c
        return logits

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv_1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.transpose(-1, -2)
        if "lm_head.weight" not in weights:
            del self.lm_head
        return weights

    def make_cache(self):
        cache = []
        for layer in self.layers:
            if layer.temporal_block_type == "recurrent":
                cache.append(ArraysCache(size=2))
            else:
                cache.append(RotatingKVCache(max_size=self.args.attention_window_size))
        return cache


from .rope_utils import RoPE
