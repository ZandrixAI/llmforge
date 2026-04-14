# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
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
    layer_types: List[str]
    vocab_size: int = 200192
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    num_dense_layers: int = 2
    route_norm: bool = True
    route_scale: float = 2.826
    score_func: str = "sigmoid"
    n_group: int = 1
    topk_group: int = 1
    sliding_window: int = 2048
    mup_enabled: bool = True


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_local_attention: bool = False):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.is_local_attention = is_local_attention

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.gate_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=False
        )

        if is_local_attention:
            self.rope = initialize_rope(
                self.head_dim,
                args.rope_theta,
                False,
                args.rope_scaling,
                args.max_position_embeddings,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if self.is_local_attention and self.rope is not None:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(1, 2).reshape(B, L, -1)

        gate = torch.sigmoid(self.gate_proj(x))
        output = output * gate

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = (
            intermediate_size
            if intermediate_size is not None
            else args.intermediate_size
        )

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoERouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


class AfmoeMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.route_norm = args.route_norm
        self.route_scale = args.route_scale
        self.score_func = args.score_func
        self.n_group = args.n_group
        self.topk_group = args.topk_group

        self.router = MoERouter(args)

        self.expert_bias = nn.Parameter(
            torch.zeros(args.num_experts), requires_grad=False
        )

        self.experts = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )

        if args.num_shared_experts > 0:
            shared_intermediate_size = (
                args.moe_intermediate_size * args.num_shared_experts
            )
            self.shared_experts = MLP(args, intermediate_size=shared_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.router(x)

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(gates.to(torch.float32))
        else:
            scores = torch.softmax(gates.to(torch.float32), dim=-1)

        selection_scores = scores + self.expert_bias

        if self.n_group > 1:
            selection_scores = selection_scores.unflatten(-1, (self.n_group, -1))
            group_scores = torch.topk(selection_scores, 2, dim=-1).values.sum(
                dim=-1, keepdim=True
            )
            k = self.n_group - self.topk_group
            group_idx = torch.topk(group_scores, k, dim=-2).indices
            selection_scores = torch.scatter(
                selection_scores,
                -2,
                group_idx.expand_as(selection_scores),
                torch.zeros_like(selection_scores),
            )
            selection_scores = selection_scores.flatten(-2, -1)

        k = self.num_experts_per_tok
        inds = torch.topk(selection_scores, k=k, dim=-1).indices

        selected_scores = torch.gather(scores, -1, inds)

        if self.route_norm and self.num_experts_per_tok > 1:
            denominator = selected_scores.sum(dim=-1, keepdim=True)
            selected_scores = selected_scores / denominator

        selected_scores = selected_scores * self.route_scale

        y = self.experts(x, inds)
        y = (y * selected_scores[..., None]).sum(dim=-2).to(y.dtype)

        if self.args.num_shared_experts > 0:
            y = y + self.shared_experts(x)

        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, use_sliding: bool = False):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding
        self.layer_idx = layer_idx

        self.self_attn = Attention(args, is_local_attention=use_sliding)

        if layer_idx < args.num_dense_layers:
            self.mlp = MLP(args)
        else:
            self.mlp = AfmoeMoE(args)

        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_mlp_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_mlp_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        r = self.post_attention_layernorm(r)
        h = x + r

        r = self.mlp(self.pre_mlp_layernorm(h))
        r = self.post_mlp_layernorm(r)
        return h + r


class AfmoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.mup_enabled = args.mup_enabled
        self.hidden_size = args.hidden_size

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    args=args,
                    layer_idx=idx,
                    use_sliding=layer_type == "sliding_attention",
                )
                for idx, layer_type in enumerate(self.layer_types)
            ]
        )
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for idx, layer in enumerate(self.layers):
            if layer.use_sliding:
                self.swa_idx = idx
                break

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if self.mup_enabled:
            h = h * math.sqrt(self.hidden_size)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = None
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AfmoeModel(args)

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

    def sanitize(self, weights):
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.args.num_hidden_layers):
            if l < self.args.num_dense_layers:
                continue
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.experts.{n}.{k}"] = torch.stack(to_join)

        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "router.gate" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
