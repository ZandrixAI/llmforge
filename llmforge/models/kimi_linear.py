# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    linear_attn_config: Dict[str, Any]
    model_max_length: int
    num_experts: int
    moe_intermediate_size: int
    kv_lora_rank: int
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = False
    num_experts_per_token: int = 1
    num_shared_experts: int = 0
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1


class KimiMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        dim = hidden_size or args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def _group_expert_select(
    gates: torch.Tensor,
    bias: Optional[torch.Tensor],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    score_function: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if score_function == "sigmoid":
        scores = torch.sigmoid(gates)
    elif score_function == "softmax":
        scores = F.softmax(gates, dim=-1)
    else:
        raise ValueError(f"Unsupported MoE router activation '{score_function}'")

    orig_scores = scores
    if bias is not None:
        scores = scores + bias.to(scores.dtype)

    if n_group > 1:
        scores = scores.unflatten(-1, (n_group, -1))
        group_scores = torch.topk(scores, 2, dim=-1).values.sum(dim=-1, keepdim=True)
        k = n_group - topk_group
        group_idx = torch.topk(group_scores, k, dim=-2).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-2, group_idx, True)
        scores = scores.masked_fill(~mask, 0.0)
        scores = scores.flatten(-2, -1)

    inds = torch.topk(scores, top_k, dim=-1).indices
    scores = torch.gather(orig_scores, -1, inds)

    if top_k > 1 and renormalize:
        denominator = scores.sum(dim=-1, keepdim=True) + 1e-20
        scores = scores / denominator

    return inds, scores * routed_scaling_factor


class KimiSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts
        if experts is None:
            raise ValueError("num_experts must be specified for MoE layers")

        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, args.moe_intermediate_size, experts)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(experts, dtype=torch.float32)
        )

        if args.num_shared_experts:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = KimiMLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.gate(x)
        inds, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )
        out = self.switch_mlp(x, inds)
        out = (out * weights[..., None]).sum(dim=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class KimiMLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim or args.head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or args.head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            args.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.q_head_dim)
        q = q.transpose(1, 2)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = compressed_kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv_latent = self.kv_a_layernorm(compressed_kv)

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


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            x = torch.where(mask[..., None], x, torch.zeros_like(x))

        if state is None:
            state = torch.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]),
                dtype=x.dtype,
                device=x.device,
            )
        conv_input = torch.cat([state, x], dim=1)
        out = F.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = torch.clamp(lengths, 0, x.shape[1])
            positions = (ends[:, None] + torch.arange(n_keep, device=x.device))[
                ..., None
            ]
            new_state = torch.gather(
                conv_input, 1, positions.expand(-1, -1, conv_input.shape[-1])
            )
        else:
            new_state = conv_input[:, -n_keep:, :]

        return out, new_state


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)

        self.projection_dim = self.num_heads * self.head_dim
        hidden = args.hidden_size

        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads).uniform_(1.0, 16.0))
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.projection_dim))

        self.o_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = None
            k_state = None
            v_state = None
            ssm_state = None
            lengths = None

        if q_state is None:
            s = torch.zeros(
                (B, self.conv_kernel - 1, self.projection_dim),
                dtype=dtype,
                device=x.device,
            )
            q_state = s
            k_state = s
            v_state = s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)

        inv_scale = self.scale
        q_norm = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
        k_norm = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
        q = (inv_scale**2) * q_norm
        k = inv_scale * k_norm

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        b_logits = self.b_proj(x).reshape(B, T, self.num_heads)

        from .gated_delta import gated_delta_update

        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        gate = self.g_b_proj(self.g_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        out = (
            self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim))
            * torch.sigmoid(gate)
        ).reshape(B, T, -1)
        return self.o_proj(out)


class KimiDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiMLAAttention(args)

        if (
            args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = KimiSparseMoE(args)
        else:
            self.mlp = KimiMLP(args)

        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        attn_cache = None if cache is None else cache
        y = self.self_attn(self.input_layernorm(x), mask, attn_cache)
        h = x + y
        z = self.mlp(self.post_attention_layernorm(h))
        return h + z


class KimiLinearModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [KimiDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        )
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        kda_layers = args.linear_attn_config["kda_layers"]
        self.ssm_idx = kda_layers[0] - 1
        for i in range(len(self.layers)):
            if (i + 1) not in kda_layers:
                self.attn_idx = i
                break

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[List[Any]] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(h, cache[self.attn_idx], return_array=True)

        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            h = layer(h, mask=mask, cache=layer_cache)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiLinearModel(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: Optional[List[Any]] = None,
    ) -> torch.Tensor:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return F.linear(out, self.model.embed_tokens.weight)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        weights = {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, KimiSparseMoE):
                src_prefix = f"{prefix}.block_sparse_moe"
                dst_prefix = f"{prefix}.mlp"
                for src, dst in [
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ]:
                    key = f"{src_prefix}.experts.0.{src}.weight"
                    if key in weights:
                        stacked = [
                            weights.pop(f"{src_prefix}.experts.{i}.{src}.weight")
                            for i in range(self.args.num_experts)
                        ]
                        weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = torch.stack(
                            stacked
                        )

                for name in ("gate_proj", "up_proj", "down_proj"):
                    src_key = f"{src_prefix}.shared_experts.{name}.weight"
                    if src_key in weights:
                        weights[f"{dst_prefix}.shared_experts.{name}.weight"] = (
                            weights.pop(src_key)
                        )

                gate_key = f"{src_prefix}.gate.weight"
                if gate_key in weights:
                    weights[f"{dst_prefix}.gate.weight"] = weights.pop(gate_key)

                bias_key = f"{src_prefix}.gate.e_score_correction_bias"
                if bias_key in weights:
                    weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(
                        bias_key
                    )

            attn = getattr(layer, "self_attn", None)
            if isinstance(attn, KimiDeltaAttention):
                attn_prefix = f"{prefix}.self_attn"
                for src_name, dst_name in (
                    ("q_conv1d", "q_conv"),
                    ("k_conv1d", "k_conv"),
                    ("v_conv1d", "v_conv"),
                ):
                    src_key = f"{attn_prefix}.{src_name}.weight"
                    if src_key in weights:
                        w = weights.pop(src_key)
                        if w.ndim == 3:
                            w = w.transpose(1, 2)
                        weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
                dt_key = f"{attn_prefix}.dt_bias"
                if dt_key in weights:
                    if weights[dt_key].ndim > 1:
                        weights[dt_key] = weights[dt_key].reshape(-1)

            attn_prefix = f"{prefix}.self_attn"
            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if kv_b_key in weights:
                qk_nope = self.args.qk_nope_head_dim or self.args.head_dim
                v_head = self.args.v_head_dim or self.args.head_dim
                head_dim = qk_nope + v_head
                num_heads = self.args.num_attention_heads

                v = weights.pop(kv_b_key)

                v = v.reshape(num_heads, head_dim, -1)
                wk = v[:, :qk_nope, :].transpose(-1, -2).contiguous()
                wv = v[:, qk_nope:, :].contiguous()

                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
