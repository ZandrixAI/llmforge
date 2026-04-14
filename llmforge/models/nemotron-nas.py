# Copyright © 2025 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .rope_utils import initialize_rope


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


@dataclass(frozen=True)
class AttentionConfig:
    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: Optional[list[str]] = None
    n_heads_in_group: Optional[int] = None
    window_length: Optional[int] = None
    num_sink_tokens: Optional[int] = None
    use_prefill_window_in_sink_attention: bool = False
    unshifted_sink: bool = False

    def __post_init__(self):
        if self.no_op or self.replace_with_linear:
            object.__setattr__(self, "n_heads_in_group", None)
            object.__setattr__(self, "window_length", None)
            object.__setattr__(self, "num_sink_tokens", None)
        elif not self.no_op:
            if self.n_heads_in_group is None:
                raise ValueError(
                    "n_heads_in_group must be specified for active attention blocks"
                )
            if self.n_heads_in_group <= 0:
                raise ValueError(
                    f"n_heads_in_group must be positive, got {self.n_heads_in_group}"
                )


@dataclass(frozen=True)
class FFNConfig:
    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: Optional[list[str]] = None
    ffn_mult: Optional[float] = None

    def __post_init__(self):
        if self.no_op or self.replace_with_linear:
            object.__setattr__(self, "ffn_mult", None)
        elif not self.no_op:
            if self.ffn_mult is None:
                raise ValueError("ffn_mult must be specified for active FFN blocks")
            object.__setattr__(self, "ffn_mult", round(self.ffn_mult, 6))


@dataclass(frozen=True)
class BlockConfig:
    attention: AttentionConfig
    ffn: FFNConfig

    @classmethod
    def from_dict(cls, data: dict):
        attn_conf = AttentionConfig(**data.get("attention", {}))
        ffn_conf = FFNConfig(**data.get("ffn", {}))
        return cls(attention=attn_conf, ffn=ffn_conf)


def _find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    intermediate_size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(intermediate_size, 256)


_ACT2FN = {
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": F.gelu,
    "gelu_fast": F.gelu,
}


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "nemotron-nas"
    hidden_size: int = 8192
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128256
    block_configs: list = field(default_factory=list)
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.block_configs and isinstance(self.block_configs[0], dict):
            self.block_configs = [
                BlockConfig.from_dict(conf) for conf in self.block_configs
            ]

        if len(self.block_configs) != self.num_hidden_layers:
            raise ValueError(
                f"Number of block_configs ({len(self.block_configs)}) must match "
                f"num_hidden_layers ({self.num_hidden_layers})"
            )

        if self.rope_scaling:
            if "factor" not in self.rope_scaling:
                raise ValueError("rope_scaling must contain 'factor'")
            rope_type = self.rope_scaling.get("rope_type")
            if rope_type is None:
                raise ValueError("rope_scaling must contain 'rope_type'")

        for i, block_conf in enumerate(self.block_configs):
            attn_conf = block_conf.attention
            if not attn_conf.no_op and not attn_conf.replace_with_linear:
                if self.num_attention_heads % attn_conf.n_heads_in_group != 0:
                    raise ValueError(
                        f"Layer {i}: num_attention_heads ({self.num_attention_heads}) "
                        f"must be divisible by n_heads_in_group ({attn_conf.n_heads_in_group})"
                    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, attention_config: AttentionConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = n_heads // attention_config.n_heads_in_group

        self.head_dim = head_dim = args.hidden_size // n_heads
        if (self.head_dim * n_heads) != dim:
            raise ValueError(
                f"hidden_size ({dim}) must be divisible by num_attention_heads ({n_heads})"
            )

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

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


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, ffn_config: FFNConfig):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = _ffn_mult_to_intermediate_size(ffn_config.ffn_mult, dim)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)

        self.act_fn = args.hidden_act
        if self.act_fn not in _ACT2FN:
            raise ValueError(f"Unknown activation function: {args.hidden_act}")

    def forward(self, x) -> torch.Tensor:
        act_fn = _ACT2FN[self.act_fn]
        return self.down_proj(act_fn(self.gate_proj(x)) * self.up_proj(x))


class LinearSubblockReplacement(nn.Module):
    def __init__(self, hidden_size: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        block_config = args.block_configs[layer_idx]
        self.attention_config = block_config.attention
        self.ffn_config = block_config.ffn

        if not self.attention_config.no_op:
            self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        else:
            self.input_layernorm = None

        if self.attention_config.no_op:
            self.self_attn = None
        elif self.attention_config.replace_with_linear:
            self.self_attn = LinearSubblockReplacement(
                args.hidden_size, args.attention_bias
            )
        else:
            self.self_attn = Attention(args, self.attention_config)

        if not self.ffn_config.no_op:
            self.post_attention_layernorm = RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
        else:
            self.post_attention_layernorm = None

        if self.ffn_config.no_op:
            self.mlp = None
        elif self.ffn_config.replace_with_linear:
            self.mlp = LinearSubblockReplacement(args.hidden_size, args.mlp_bias)
        else:
            self.mlp = MLP(args, self.ffn_config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:

        if self.self_attn is not None:
            residual = x
            h = self.input_layernorm(x)
            attn_out = self.self_attn(h, mask=mask, cache=cache)
            x = residual + attn_out

        if self.mlp is not None:
            residual = x
            h = self.post_attention_layernorm(x)
            mlp_out = self.mlp(h)
            x = residual + mlp_out

        return x


class NemotronNASModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(args=args, layer_idx=i)
                for i in range(args.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.num_attn_layers = sum(
            1 for layer in self.layers if layer.self_attn is not None
        )

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[List[Any]] = None,
    ):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * self.num_attn_layers

        mask = create_attention_mask(h, cache[0])

        cache_idx = 0
        for layer in self.layers:
            if layer.self_attn is not None:
                c = cache[cache_idx]
                cache_idx += 1
            else:
                c = None
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NemotronNASModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = None

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        out = self.model(inputs, cache=cache)
        if self.args.tie_word_embeddings:
            out = F.linear(out, self.model.embed_tokens.weight)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for layer in self.layers if layer.self_attn is not None]
