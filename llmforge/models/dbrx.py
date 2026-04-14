# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    vocab_size: int
    d_model: int
    ffn_config: dict
    attn_config: dict
    n_layers: int
    n_heads: int


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.n_heads
        self.d_model = args.d_model
        self.head_dim = args.d_model // args.n_heads
        self.num_key_value_heads = args.attn_config["kv_n_heads"]
        self.clip_qkv = args.attn_config["clip_qkv"]
        self.rope_theta = args.attn_config["rope_theta"]

        self.scale = self.head_dim**-0.5

        self.Wqkv = nn.Linear(
            args.d_model,
            (self.num_key_value_heads * 2 + self.num_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        self.rope = initialize_rope(self.head_dim, self.rope_theta, False, None, None)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:

        qkv = self.Wqkv(x)
        qkv = torch.clamp(qkv, min=-self.clip_qkv, max=self.clip_qkv)
        splits = [self.d_model, self.d_model + self.head_dim * self.num_key_value_heads]
        queries, keys, values = torch.split(qkv, splits, dim=-1)

        B, L, D = x.shape

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(1, 2)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(1, 2)

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


class NormAttnNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_1 = nn.LayerNorm(args.d_model, bias=False)
        self.norm_2 = nn.LayerNorm(args.d_model, bias=False)
        self.attn = Attention(args)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        h = self.attn(self.norm_1(x), mask=mask, cache=cache)
        x = h + x
        return x, self.norm_2(x)


class MLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.v1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_hidden_states = swiglu(self.w1(x), self.v1(x))
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.layer = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.ffn_dim = args.ffn_config["ffn_hidden_size"]
        self.num_experts = args.ffn_config["moe_num_experts"]
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]

        self.router = Router(self.d_model, self.num_experts)
        self.experts = nn.ModuleList(
            [MLP(self.d_model, self.ffn_dim) for _ in range(self.num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)
        gates = torch.softmax(gates.to(torch.float32), dim=-1)

        inds = torch.topk(-gates, k=ne, dim=-1).indices.detach()
        scores = torch.gather(gates, -1, inds)
        scores = scores / torch.linalg.norm(scores, ord=1, dim=-1, keepdim=True)
        scores = scores.to(x.dtype)

        if self.training:
            inds_np = inds.cpu().numpy()
            y = torch.zeros(x.shape[0], ne, x.shape[-1], dtype=x.dtype, device=x.device)
            for e, expert in enumerate(self.experts):
                idx1, idx2 = np.where(inds_np == e)
                if len(idx1) == 0:
                    continue
                y[idx1, idx2] = expert(x[idx1])

            y = (y * scores[:, :, None]).sum(dim=1)
        else:
            y = []
            for xt, st, it in zip(x, scores, inds.tolist()):
                yt = torch.stack([self.experts[e](xt) for e in it], dim=-1)
                yt = (yt * st).sum(dim=-1)
                y.append(yt)
            y = torch.stack(y, dim=0)

        return y.reshape(orig_shape)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn = SparseMoeBlock(args)
        self.norm_attn_norm = NormAttnNorm(args)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        r, h = self.norm_attn_norm(x, mask, cache)
        out = self.ffn(h) + r
        return out


class DBRX(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = nn.ModuleList(
            [DecoderLayer(args=args) for _ in range(args.n_layers)]
        )
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        h = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.blocks)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.blocks, cache):
            h = layer(h, mask, c)

        return self.norm_f(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.transformer = DBRX(args)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.args = args

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.transformer.blocks

    def sanitize(self, weights):
        num_experts = self.args.ffn_config["moe_num_experts"]

        pattern = "experts.mlp"
        new_weights = {k: v for k, v in weights.items() if pattern not in k}
        for k, v in weights.items():
            if pattern in k:
                experts = [
                    (k.replace(".mlp", f".{e}") + ".weight", sv)
                    for e, sv in enumerate(torch.split(v, 1, dim=0))
                ]
                if k.endswith("w2"):
                    experts = [(s, sv.squeeze(0).T) for s, sv in experts]
                else:
                    experts = [(s, sv.squeeze(0)) for s, sv in experts]
                new_weights.update(experts)
        return new_weights
