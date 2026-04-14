# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ArraysCache, CacheList, KVCache, RotatingKVCache
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
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float
    sliding_window: int
    sliding_window_layers: List[int]
    conv_window: int
    rms_norm_eps: float
    model_type: str = "baichuan_m1"
    num_swa_attention_heads: Optional[int] = None
    num_swa_key_value_heads: Optional[int] = None
    tie_word_embeddings: bool = False


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            raise ValueError("Layer index must be provided to Attention module.")

        self.is_swa = layer_idx in config.sliding_window_layers
        self.num_heads = (
            config.num_swa_attention_heads
            if self.is_swa and config.num_swa_attention_heads
            else config.num_attention_heads
        )
        self.num_kv_heads = (
            config.num_swa_key_value_heads
            if self.is_swa and config.num_swa_key_value_heads
            else config.num_key_value_heads
        )

        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size

        self.scale = self.head_dim**-0.5

        self.W_pack = nn.Linear(
            config.hidden_size,
            self.hidden_size + 2 * self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.rope = initialize_rope(
            self.head_dim,
            base=config.rope_theta,
            traditional=False,
        )

        self.conv_window = config.conv_window
        assert self.conv_window == 2
        self.conv_k = torch.zeros((1, 1, self.num_kv_heads, 1, self.conv_window))
        self.conv_v = torch.zeros((1, 1, self.num_kv_heads, 1, self.conv_window))

    def _custom_convolution(self, u, weights, state=None):
        B, H, L, D = u.shape
        weights = weights.reshape((1, H, self.conv_window, 1, 1))
        w0 = weights[:, :, 0]
        w1 = weights[:, :, 1]
        if state is None:
            state = torch.zeros((B, H, 1, D), dtype=u.dtype, device=u.device)
        if L > 1:
            u_prev = torch.cat([state, u[:, :, :-1]], dim=2)
        else:
            u_prev = state
        return u_prev * w0 + u * w1

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, cache: Any = None
    ) -> torch.Tensor:
        B, L, D = x.shape

        proj = self.W_pack(x)
        q, k, v = torch.split(
            proj,
            [D, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim],
            dim=-1,
        )

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if cache is None:
            cache = (None, None)

        if cache[0] is not None:
            offset = cache[1].offset
            last_k, last_v = cache[0][0], cache[0][1]
        else:
            offset = 0
            last_k, last_v = None, None

        k_init = k
        v_init = v
        k = self._custom_convolution(k, self.conv_k, state=last_k)
        v = self._custom_convolution(v, self.conv_v, state=last_v)
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache[0] is not None:
            k, v = cache[1].update_and_fetch(k, v)
            if L > 0:
                cache[0][0] = k_init[:, :, -1:, :]
                cache[0][1] = v_init[:, :, -1:, :]

        out = scaled_dot_product_attention(
            q, k, v, cache=cache[1], scale=self.scale, mask=mask
        )
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, cache: Any = None
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + r
        r = self.mlp(self.post_attention_layernorm(x))
        return x + r


class BaichuanModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.sliding_window = config.sliding_window
        self.first_swa_idx = None
        if config.sliding_window_layers:
            self.first_swa_idx = config.sliding_window_layers[0]

        self.first_global_idx = None
        self.swa_layers = set(config.sliding_window_layers)
        for i in range(config.num_hidden_layers):
            if i in self.swa_layers:
                continue
            self.first_global_idx = i
            break

    def forward(self, inputs: torch.Tensor, cache: Any = None) -> torch.Tensor:
        x = self.embed_tokens(inputs)

        if cache is None:
            cache = [(None, None)] * len(self.layers)

        if self.first_global_idx is None:
            c_global = None
        else:
            c_global = cache[self.first_global_idx][1]

        if self.first_swa_idx is None:
            c_swa = None
        else:
            c_swa = cache[self.first_swa_idx][1]

        global_mask = create_attention_mask(x, c_global)
        swa_mask = create_attention_mask(x, c_swa, window_size=self.sliding_window)

        for l, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = swa_mask if l in self.swa_layers else global_mask
            x = layer(x, mask, c)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = BaichuanModel(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> List[Any]:
        caches = []
        for i, layer in enumerate(self.model.layers):
            is_swa = i in self.config.sliding_window_layers
            conv_cache = ArraysCache(size=2)
            if is_swa:
                kv_cache = RotatingKVCache(max_size=self.config.sliding_window)
            else:
                kv_cache = KVCache()
            caches.append(CacheList(conv_cache, kv_cache))
        return caches

    def sanitize(self, weights: dict) -> dict:
        is_quantized = "lm_head.scales" in weights
        if not is_quantized and "lm_head.weight" in weights:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            w = w.to(torch.float32)
            norm = torch.norm(w, dim=-1, keepdim=True)
            w = (w / (norm + 1e-7)).to(dtype)
            weights["lm_head.weight"] = w
        return weights

    def forward(self, inputs: torch.Tensor, cache: Any = None) -> torch.Tensor:
        outputs = self.model(inputs, cache)
        return self.lm_head(outputs)

    @property
    def layers(self) -> List[nn.Module]:
        return self.model.layers
