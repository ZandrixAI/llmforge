# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModelArgs
from .cache import ArraysCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    norm_eps: float
    head_dim: int
    num_hidden_layers: int
    a_low_rank_dim: int
    v_low_rank_dim: int
    gate_low_rank_dim: int
    decay_low_rank_dim: int
    tie_word_embeddings: bool = False


def addcmul(x, y, z):
    return x + y * z


def l2_norm(x):
    return x / torch.maximum(
        torch.linalg.norm(x, axis=-1, keepdim=True), torch.tensor(1e-7)
    )


def _wkv7_step_ops(r, w, k, v, a, b, state):
    sab = (state @ a[..., None]) @ b[..., None, :]
    state = state * w[:, :, None, :] + v[..., None] @ k[..., None, :] + sab
    y = state @ r[..., None]
    return y, state


class LayerNormPerHead(nn.Module):
    def __init__(self, head_dim, num_heads, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((num_heads, head_dim)))
        self.bias = nn.Parameter(torch.zeros((num_heads, head_dim)))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class LoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: Optional[bool] = True,
        activation: Optional[str] = "tanh",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        if activation is None:
            self.activation = nn.Identity()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation}.")

        self.lora = nn.ModuleList(
            [
                nn.Linear(self.input_dim, self.low_rank_dim, bias=False),
                self.activation,
                nn.Linear(self.low_rank_dim, self.output_dim, bias=self.bias),
            ]
        )

    def forward(self, x) -> torch.Tensor:
        return self.lora[2](self.lora[1](self.lora[0](x)))


class TokenShift(nn.Module):
    def forward(self, x, state):
        B, L, D = x.shape
        if state is None:
            state = torch.zeros((B, 1, D), dtype=x.dtype, device=x.device)
        if L == 1:
            return state
        else:
            return torch.cat([state, x[:, :-1, :]], dim=1)


class Rwkv7ChannelMixing(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = args.hidden_size
        intermediate_size = args.intermediate_size

        self.key = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_dim, bias=False)

        self.x_k = nn.Parameter(torch.zeros((hidden_dim,)))

        self.token_shift = TokenShift()

    def forward(self, x, cache) -> torch.Tensor:
        state = cache[2] if cache is not None else None
        x_prev = self.token_shift(x, state)
        xx = addcmul(x, x_prev - x, self.x_k)
        if cache is not None:
            cache[2] = x[:, -1:, :]
        return self.value(F.relu(self.key(xx)) ** 2)


class Rwkv7TimeMixing(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.args = args
        self.hidden_size = args.hidden_size
        self.head_dim = args.head_dim
        self.num_heads = self.hidden_size // self.head_dim
        self.a_low_rank_dim = args.a_low_rank_dim
        self.v_low_rank_dim = args.v_low_rank_dim
        self.gate_low_rank_dim = args.gate_low_rank_dim
        self.decay_low_rank_dim = args.decay_low_rank_dim

        self.token_shift = TokenShift()

        self.x_r = nn.Parameter(torch.zeros((1, 1, self.hidden_size)))
        self.x_w = nn.Parameter(torch.zeros((1, 1, self.hidden_size)))
        self.x_k = nn.Parameter(torch.zeros((1, 1, self.hidden_size)))
        self.x_v = nn.Parameter(torch.zeros((1, 1, self.hidden_size)))
        self.x_a = nn.Parameter(torch.zeros((1, 1, self.hidden_size)))
        self.x_g = nn.Parameter(torch.zeros((1, 1, self.hidden_size)))

        self.k_k = nn.Parameter(torch.zeros((self.num_heads, self.head_dim)))
        self.k_a = nn.Parameter(torch.zeros((self.num_heads, self.head_dim)))
        self.r_k = nn.Parameter(torch.zeros((self.num_heads, self.head_dim)))

        self.r_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.g_norm = LayerNormPerHead(self.head_dim, self.num_heads, eps=64e-5)

        self.w_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.decay_low_rank_dim,
            activation="tanh",
        )

        if self.layer_idx > 0:
            self.v_lora = LoRA(
                self.hidden_size,
                self.hidden_size,
                low_rank_dim=self.v_low_rank_dim,
                activation=None,
            )

        self.a_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.a_low_rank_dim,
            activation=None,
        )

        self.g_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.gate_low_rank_dim,
            activation="sigmoid",
            bias=False,
        )

    def _wkv7(self, r, w, k, v, a, b, state):
        B, L, _, _ = r.shape
        if state is None:
            state = torch.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim),
                dtype=r.dtype,
                device=r.device,
            )

        ys = []
        for t in range(L):
            y, state = _wkv7_step_ops(
                r[:, t], w[:, t], k[:, t], v[:, t], a[:, t], b[:, t], state
            )
            ys.append(y)

        y = torch.stack(ys, dim=1).to(r.dtype)
        return y, state

    def forward(self, x, v_first, cache):
        if cache is None:
            token_shift_cache, state_cache = None, None
        else:
            token_shift_cache, state_cache = cache[0], cache[1]

        B, L, D = x.shape
        x_prev = self.token_shift(x, token_shift_cache)
        xx = x_prev - x

        xr = addcmul(x, xx, self.x_r)
        xw = addcmul(x, xx, self.x_w)
        xk = addcmul(x, xx, self.x_k)
        xv = addcmul(x, xx, self.x_v)
        xa = addcmul(x, xx, self.x_a)
        xg = addcmul(x, xx, self.x_g)

        key = self.k_proj(xk).reshape(B, L, self.num_heads, self.head_dim)
        value = self.v_proj(xv).reshape(B, L, self.num_heads, self.head_dim)
        receptance = self.r_proj(xr).reshape(B, L, self.num_heads, self.head_dim)
        iclr = torch.sigmoid(self.a_lora(xa)).reshape(
            B, L, self.num_heads, self.head_dim
        )
        gate = self.g_lora(xg)

        if self.layer_idx == 0:
            v_first = value
        else:
            vv = torch.sigmoid(self.v_lora(xv)).reshape(
                B, L, self.num_heads, self.head_dim
            )
            value = addcmul(value, v_first - value, vv)

        decay = torch.sigmoid(
            self.w_lora(xw).reshape(B, L, self.num_heads, self.head_dim)
        ).to(torch.float32)
        decay = torch.exp(-0.606531 * decay).to(receptance.dtype)
        kk = l2_norm((key * self.k_k))
        key = key * (1 + (iclr - 1) * self.k_a)
        a = -kk
        b = kk * iclr

        out, new_state_cache = self._wkv7(
            receptance, decay, key, value, a, b, state_cache
        )
        out = self.g_norm(out.reshape(B, L, self.num_heads, self.head_dim))
        out = (
            out + (receptance * key * self.r_k).sum(dim=-1, keepdim=True) * value
        ).reshape([B, L, D])

        if cache is not None:
            cache[0] = x[:, -1:, :]
            cache[1] = new_state_cache

        out = self.o_proj(out * gate)
        return out, v_first


class Rwkv7Layer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if self.layer_idx == 0:
            self.pre_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)
        self.attn = Rwkv7TimeMixing(args, layer_idx=self.layer_idx)
        self.ffn = Rwkv7ChannelMixing(args)
        self.attn_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)

    def forward(self, x, v_first, cache):
        if self.layer_idx == 0:
            x = self.pre_norm(x)

        h, v_first = self.attn(self.attn_norm(x), v_first, cache)
        h = x + h
        out = h + self.ffn(self.ffn_norm(h), cache)
        return out, v_first


class Rwkv7Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [Rwkv7Layer(args, layer_idx=i) for i in range(args.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, cache):
        x = self.embeddings(x)
        if cache is None:
            cache = [None] * len(self.layers)

        v_first = None
        for layer, c in zip(self.layers, cache):
            x, v_first = layer(x, v_first, c)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Rwkv7Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(self, inputs: torch.Tensor, cache=None):
        x = self.model(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = F.linear(x, self.model.embeddings.weight)
        else:
            logits = self.lm_head(x)

        return logits

    def make_cache(self):
        return [ArraysCache(size=3) for _ in range(len(self.layers))]

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        for k, v in weights.items():
            if "k_k" in k or "k_a" in k or "g_norm" in k:
                weights[k] = weights[k].reshape(
                    self.args.hidden_size // self.args.head_dim, self.args.head_dim
                )
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "lora.2" in path or "embeddings" in path:
                return {"bits": 8}
            return True

        return predicate
