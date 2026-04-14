import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache
from .mla import MultiLinear
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
    attention_method: str
    zero_expert_type: str
    hidden_size: int
    ffn_hidden_size: int
    moe_topk: int
    expert_ffn_hidden_size: int
    n_routed_experts: int
    zero_expert_num: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    mla_scale_q_lora: bool
    mla_scale_kv_lora: bool
    attention_bias: bool
    norm_topk_prob: bool = False
    router_bias: bool = False
    rope_scaling: Optional[Dict] = None


class LongcatFlashMLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.v_head_dim = args.v_head_dim

        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.scale = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size,
                self.num_attention_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                args.hidden_size, self.q_lora_rank, bias=args.attention_bias
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_attention_heads * self.qk_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_attention_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_attention_heads
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * args.v_head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        if args.mla_scale_q_lora:
            self.mla_scale_q_lora = (args.hidden_size / self.q_lora_rank) ** 0.5
        if args.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (args.hidden_size / self.kv_lora_rank) ** 0.5

        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = args.rope_scaling["factor"]
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_attention_heads, self.qk_head_dim).transpose(1, 2)

        if self.mla_scale_q_lora is not None:
            q = q * self.mla_scale_q_lora

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        if self.mla_scale_kv_lora is not None:
            kv_latent = kv_latent * self.mla_scale_kv_lora

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = kv_latent.unsqueeze(1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.transpose(-1, -2)
        if mask is not None:
            pe_scores = torch.where(
                mask,
                pe_scores,
                torch.tensor(
                    torch.finfo(pe_scores.dtype).min,
                    dtype=pe_scores.dtype,
                    device=pe_scores.device,
                ),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(output)


class LongcatFlashMLP(nn.Module):
    def __init__(self, args: ModelArgs, is_expert: bool = False):
        super().__init__()
        hidden_size = args.expert_ffn_hidden_size if is_expert else args.ffn_hidden_size

        self.gate_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, args.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LongcatFlashTopkRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.top_k = args.moe_topk
        self.n_routed_experts = args.n_routed_experts + args.zero_expert_num
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.router_bias = args.router_bias

        self.classifier = nn.Linear(
            args.hidden_size, self.n_routed_experts, bias=self.router_bias
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.n_routed_experts), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        dtype = hidden_states.dtype
        router_logits = self.classifier(hidden_states)
        scores = torch.softmax(router_logits, dim=-1)

        corrected_scores = scores + self.e_score_correction_bias
        topk_indices = torch.topk(
            corrected_scores, k=self.top_k, dim=-1, largest=True
        ).indices
        topk_weights = torch.gather(scores, -1, topk_indices)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights.to(dtype)


class LongcatFlashMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.num_experts_per_tok = args.moe_topk
        self.n_routed_experts = args.n_routed_experts
        self.zero_expert_num = args.zero_expert_num
        self.zero_expert_type = args.zero_expert_type

        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.expert_ffn_hidden_size,
            args.n_routed_experts,
        )

        self.router = LongcatFlashTopkRouter(args)

    def forward(self, hidden_states):
        topk_indices, topk_weights = self.router(hidden_states)

        # Process all regular experts at once
        mask = topk_indices >= self.n_routed_experts
        topk_indices = torch.where(mask, torch.zeros_like(topk_indices), topk_indices)
        regular_weights = torch.where(
            mask, torch.zeros_like(topk_weights), topk_weights
        )

        regular_outputs = self.switch_mlp(hidden_states, topk_indices)

        weighted_outputs = regular_outputs * regular_weights[..., None]
        final_output = weighted_outputs.sum(dim=-2)

        # Add identity expert contribution
        assert self.zero_expert_type == "identity"
        identity_weights_sum = torch.where(
            mask, topk_weights, torch.zeros_like(topk_weights)
        ).sum(dim=-1, keepdim=True)
        final_output = final_output + hidden_states * identity_weights_sum

        return final_output


class LongcatFlashDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.mlp = LongcatFlashMoE(args)

        self.self_attn = nn.ModuleList([LongcatFlashMLA(args) for _ in range(2)])
        self.mlps = nn.ModuleList([LongcatFlashMLP(args, False) for _ in range(2)])
        self.input_layernorm = nn.ModuleList(
            [RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        hidden_states = x
        shortcut_mlp_output = None

        if cache is None:
            cache = (None, None)

        for i in range(2):
            residual = hidden_states

            hidden_states = self.input_layernorm[i](hidden_states)
            hidden_states = self.self_attn[i](hidden_states, mask=mask, cache=cache[i])
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm[i](hidden_states)

            if i == 0:
                shortcut_mlp_output = self.mlp(hidden_states)

            hidden_states = self.mlps[i](hidden_states)
            hidden_states = residual + hidden_states

            if i == 1:
                hidden_states = hidden_states + shortcut_mlp_output

        return hidden_states


class LongcatFlashModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_layers = args.num_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [LongcatFlashDecoderLayer(args) for idx in range(args.num_layers)]
        )
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [(None, None)] * self.num_layers

        mask = create_attention_mask(h, cache[0][0], return_array=True)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LongcatFlashModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("classifier"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def sanitize(self, weights):
        for l in range(self.args.num_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = torch.stack(
                            to_join
                        )

        for l in range(self.args.num_layers):
            for i in range(2):
                prefix = f"model.layers.{l}.self_attn.{i}"
                kv_b_key = f"{prefix}.kv_b_proj.weight"
                if kv_b_key in weights:
                    num_heads = self.args.num_attention_heads
                    head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
                    quantized = f"{prefix}.kv_b_proj.scales" in weights
                    v = weights.pop(kv_b_key)

                    if quantized:
                        dims = self.args.kv_lora_rank
                        scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                        biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                        bits = (v.shape[-1] * 32) // dims
                        group_size = dims // scales.shape[-1]
                        v = torch.ops.aten._dequantize_4bit(
                            v, scales, biases, group_size=group_size, bits=bits
                        )

                    v = v.reshape(num_heads, head_dim, -1)
                    wk = (
                        v[:, : self.args.qk_nope_head_dim, :]
                        .transpose(-1, -2)
                        .contiguous()
                    )
                    wv = v[:, self.args.qk_nope_head_dim :, :].contiguous()

                    weights[f"{prefix}.embed_q.weight"] = wk
                    weights[f"{prefix}.unembed_out.weight"] = wv

        new_weights = {}
        for k, v in weights.items():
            if k.startswith("model.mtp"):
                continue
            new_weights[k] = v
        return new_weights

    def make_cache(self):
        return [CacheList(KVCache(), KVCache()) for _ in self.model.layers]
