# Copyright © 2025 Apple Inc.
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


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
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    use_expert_bias: bool
    num_dense_layers: int
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    rope_theta: float = 1000000.0
    rope_parameters: Optional[dict] = None
    full_attn_idxs: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.full_attn_idxs is None:
            self.full_attn_idxs = [
                i
                for i, layer_type in enumerate(self.layer_types)
                if layer_type == "full_attention"
            ]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = head_dim**-0.5

        self.q_layernorm = RMSNorm(head_dim, eps=args.norm_eps)
        self.k_layernorm = RMSNorm(head_dim, eps=args.norm_eps)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            1, 2
        )
        keys = self.k_layernorm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )
        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(output)


class ShortConv(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_idx: int,
    ):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.L_cache = args.conv_L_cache
        self.bias = args.conv_bias

        self.conv = nn.Conv1d(
            in_channels=args.hidden_size,
            out_channels=args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=self.bias,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ):
        BCx = self.in_proj(x)
        B, C, x = BCx.chunk(3, dim=-1)
        Bx = B * x
        if mask is not None:
            Bx = torch.where(mask[..., None], Bx, torch.zeros_like(Bx))

        if cache is not None:
            if cache[0] is None:
                state = torch.zeros(
                    (Bx.shape[0], self.L_cache - 1, self.args.hidden_size),
                    dtype=Bx.dtype,
                    device=Bx.device,
                )
            else:
                state = cache[0]
            Bx = torch.cat([state, Bx], dim=1)
            n_keep = self.L_cache - 1
            t = x.shape[1]
            if cache.lengths is not None:
                ends = torch.clamp(cache.lengths, 0, t)
                positions = (ends[:, None] + torch.arange(n_keep, device=Bx.device))[
                    ..., None
                ]
                cache[0] = torch.gather(Bx, 1, positions.expand(-1, -1, Bx.shape[-1]))
            else:
                cache[0] = Bx[:, -n_keep:, :]
            cache.advance(t)
        else:
            Bx = F.pad(Bx, (0, 0, self.L_cache - 1, 0))

        conv_out = self.conv(Bx)

        y = C * conv_out
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_expert_bias = args.use_expert_bias

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)
        if self.use_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))

    def forward(
        self,
        x: torch.Tensor,
    ):
        gates = self.gate(x).to(torch.float32)
        gates = F.softmax(gates, dim=-1)

        if self.use_expert_bias:
            gates = gates + self.expert_bias

        k = self.top_k
        inds = torch.topk(gates, k, dim=-1).indices

        scores = torch.gather(gates, -1, inds)
        if self.norm_topk_prob:
            scores = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-20)
        scores = scores.to(x.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(dim=-2)

        return y


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_attention_layer = layer_idx in args.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args, layer_idx)
        self.feed_forward = (
            MLP(
                config=args,
                intermediate_size=args.intermediate_size,
            )
            if layer_idx < args.num_dense_layers
            else Lfm2MoeSparseMoeBlock(args)
        )

        self.operator_norm = RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:

        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            r = self.conv(
                self.operator_norm(x),
                mask=mask,
                cache=cache,
            )
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Lfm2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [Lfm2DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)]
        )

        self.embedding_norm = RMSNorm(args.hidden_size, eps=args.norm_eps)

        self.fa_idx = args.full_attn_idxs[0]
        self.conv_idx = 0
        for i in range(args.num_hidden_layers):
            if i in args.full_attn_idxs:
                self.conv_idx += 1
            else:
                break

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
        input_embeddings: Optional[torch.Tensor] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self.fa_idx])
        conv_mask = create_ssm_mask(h, cache[self.conv_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask, cache=c)

        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm2Model(args)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
        input_embeddings: Optional[torch.Tensor] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        return F.linear(out, self.model.embed_tokens.weight)

    def sanitize(self, weights):
        sanitized_weights = {}
        for name, param in weights.items():
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)
            replacements = {
                "w1.weight": "gate_proj.weight",
                "w2.weight": "down_proj.weight",
                "w3.weight": "up_proj.weight",
            }
            for old, new in replacements.items():
                if old in name:
                    name = name.replace(old, new)
            sanitized_weights[name] = param

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n in ["gate_proj", "down_proj", "up_proj"]:
                if f"{prefix}.feed_forward.experts.0.{n}.weight" in sanitized_weights:
                    to_join = [
                        sanitized_weights.pop(
                            f"{prefix}.feed_forward.experts.{e}.{n}.weight"
                        )
                        for e in range(self.args.num_experts)
                    ]
                    sanitized_weights[
                        f"{prefix}.feed_forward.switch_mlp.{n}.weight"
                    ] = torch.stack(to_join)
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if l.is_attention_layer else ArraysCache(size=1)
            for l in self.layers
        ]

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("feed_forward.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate
