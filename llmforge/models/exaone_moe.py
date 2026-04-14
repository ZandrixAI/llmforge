# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, List, Optional

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
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    rms_norm_eps: float
    max_position_embeddings: int
    sliding_window: int
    layer_types: List[str]
    is_moe_layer: List[bool]
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    rope_parameters: Optional[dict] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]


def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):
    scores = torch.sigmoid(gates.to(torch.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias

    if n_group > 1:
        scores = scores.unflatten(-1, (n_group, -1))
        group_scores = torch.topk(scores, 2, dim=-1).values.sum(dim=-1, keepdim=True)
        k = n_group - topk_group
        group_idx = torch.topk(group_scores, k, dim=-2, largest=False).indices
        scores.scatter_(-2, group_idx, 0.0)
        scores = scores.flatten(-2, -1)

    k = top_k
    inds = torch.topk(scores, k, dim=-1).indices
    scores = orig_scores.gather(-1, inds)

    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(dim=-1, keepdim=True)
        scores = scores / (denominator + 1e-20)

    scores = scores * routed_scaling_factor
    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.n_routed_experts = args.num_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.weight = nn.Parameter(torch.zeros(self.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(
                self.n_routed_experts,
            )
        )
        assert args.topk_method == "noaux_tc", "Unsupported topk method."

    def forward(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        hidden_size = args.hidden_size
        intermediate_size = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )

        self.gate = MoEGate(args)

        self.shared_experts = (
            MLP(
                args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
            )
            if args.num_shared_experts is not None and args.num_shared_experts > 0
            else None
        )

    def forward(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(dim=-2).to(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
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

        self.is_sliding_window = args.layer_types[layer_idx] == "sliding_attention"
        self.apply_rope_all_layers = "sliding_attention" not in args.layer_types
        self.use_rope = self.is_sliding_window or self.apply_rope_all_layers

        if self.use_rope:
            self.rope = initialize_rope(
                self.head_dim,
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(1, 2)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)

        if cache is not None:
            if self.use_rope:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        elif self.use_rope:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.self_attn = Attention(args, layer_idx)
        self.mlp = MoE(args) if args.is_moe_layer[layer_idx] else MLP(args)
        self.is_sliding_window = self.self_attn.is_sliding_window

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
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class ExaoneMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        )
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.swa_idx = None
        self.ga_idx = None
        for i, layer in enumerate(self.layers):
            if layer.is_sliding_window and self.swa_idx is None:
                self.swa_idx = i
            if not layer.is_sliding_window and self.ga_idx is None:
                self.ga_idx = i

        self.window_size = args.sliding_window

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(
            h, cache[self.ga_idx] if self.ga_idx is not None else cache[0]
        )
        swa_mask = create_attention_mask(
            h,
            cache[self.swa_idx] if self.swa_idx is not None else cache[0],
            window_size=self.window_size,
        )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.is_sliding_window else global_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ExaoneMoEModel(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = F.linear(out, self.model.embed_tokens.weight)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        new_weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
        weights = new_weights

        for l in range(self.args.num_hidden_layers):
            if not self.args.is_moe_layer[l]:
                continue

            prefix = f"model.layers.{l}"

            bias_key = f"{prefix}.mlp.e_score_correction_bias"
            if bias_key in weights:
                weights[f"{prefix}.mlp.gate.e_score_correction_bias"] = weights.pop(
                    bias_key
                )

            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    first_key = f"{prefix}.mlp.experts.0.{m}.{k}"
                    last_key = (
                        f"{prefix}.mlp.experts.{self.args.num_experts - 1}.{m}.{k}"
                    )
                    if first_key in weights and last_key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = torch.stack(
                            to_join
                        )

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_sliding_window:
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches
