# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModelArgs, create_attention_mask, create_ssm_mask
from .activations import swiglu
from .cache import ArraysCache, KVCache
from .ssm import ssm_update
from .rope_utils import initialize_rope


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GemmaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        offset: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.offset = offset

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight + self.offset) * hidden_states


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "plamo2"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_size_per_head: int = 128
    max_position_embeddings: int = 2048
    attention_window_size: int = 2048
    full_attention_idx: Optional[list[int]] = None
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_num_heads: int = 64
    mamba_step: int = 2
    mamba_chunk_size: int = 256
    mamba_enabled: bool = True
    intermediate_size: int = 13312
    vocab_size: int = 32000


class Mamba(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.d_state = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.chunk_size = config.mamba_chunk_size
        self.num_heads = config.mamba_num_heads
        self.hidden_size_per_head = config.hidden_size_per_head

        self.intermediate_size = self.num_heads * self.hidden_size_per_head

        self.in_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=0,
        )
        self.dt_dim = max(64, self.hidden_size // 16)
        self.bcdt_proj = nn.Linear(
            self.intermediate_size,
            self.dt_dim + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_dim, self.num_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32))
        )

        self.D = nn.Parameter(torch.ones(self.num_heads))

        self.dt_norm_weight = nn.Parameter(torch.ones(self.dt_dim))
        self.B_norm_weight = nn.Parameter(torch.ones(self.d_state))
        self.C_norm_weight = nn.Parameter(torch.ones(self.d_state))

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def _conv(
        self,
        conv_input: torch.Tensor,
        cache: Optional[ArraysCache],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is not None:
            conv_input = torch.where(
                mask[..., None],
                conv_input,
                torch.tensor(0.0, device=conv_input.device, dtype=conv_input.dtype),
            )

        if cache is not None:
            if cache[0] is None:
                conv_state = torch.zeros(
                    (
                        conv_input.shape[0],
                        self.conv_kernel_size - 1,
                        self.intermediate_size,
                    ),
                    dtype=conv_input.dtype,
                    device=conv_input.device,
                )
            else:
                conv_state = cache[0]
            padded_input = torch.cat([conv_state, conv_input], dim=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                t = padded_input.shape[1]
                ends = torch.clamp(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + torch.arange(n_keep, device=ends.device))[
                    ..., None
                ]
                cache[0] = torch.gather(
                    padded_input, 1, positions.expand(-1, -1, padded_input.shape[-1])
                )
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = F.pad(conv_input, (0, 0, self.conv_kernel_size - 1, 0))

        conv_output = self.conv1d(padded_input)
        return F.silu(conv_output)

    def _ssm(
        self,
        x: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor,
        cache: Optional[Any],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = x.reshape(batch_size, seq_len, self.num_heads, self.hidden_size_per_head)
        B = B.reshape(batch_size, seq_len, 1, self.d_state)
        C = C.reshape(batch_size, seq_len, 1, self.d_state)
        if cache:
            state = cache[1]
            lengths = cache.lengths
        else:
            state, lengths = None, None

        y, state = ssm_update(
            x,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            state,
            mask=mask,
            lengths=lengths,
        )
        if cache:
            cache[1] = state
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache=None,
    ):
        bsize, length, _ = hidden_states.shape

        zx = self.in_proj(hidden_states)
        zx = zx.reshape(bsize, length, self.num_heads, -1)
        z, x = torch.split(
            zx,
            [self.hidden_size_per_head, self.hidden_size_per_head],
            dim=-1,
        )

        x = x.reshape(bsize, -1, self.num_heads * self.hidden_size_per_head)
        x = self._conv(x, cache, mask)

        BCdt = self.bcdt_proj(x)
        B, C, dt = torch.split(BCdt, [self.d_state, self.d_state, self.dt_dim], dim=-1)

        A = -torch.exp(self.A_log.to(torch.float32))
        dt = (
            dt
            * torch.rsqrt(dt.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
            * self.dt_norm_weight
        )
        B = (
            B
            * torch.rsqrt(B.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
            * self.B_norm_weight
        )
        C = (
            C
            * torch.rsqrt(C.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
            * self.C_norm_weight
        )

        dt = self.dt_proj(dt)
        out = self._ssm(
            x,
            B,
            C,
            dt,
            cache,
            mask,
        )
        if cache:
            cache.advance(out.shape[1])

        out = swiglu(z.flatten(-2), out)
        return self.out_proj(out)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        head_dim = config.hidden_size_per_head
        self.max_position_embeddings = config.max_position_embeddings
        self.scale = head_dim**-0.5

        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = config.num_key_value_heads
        assert self.q_num_heads % self.k_num_heads == 0
        self.n_group = self.q_num_heads // self.k_num_heads

        self.q_proj_dim = self.q_num_heads * self.qk_dim
        self.k_proj_dim = self.k_num_heads * self.qk_dim
        self.v_proj_dim = self.k_num_heads * self.v_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_proj_dim + self.k_proj_dim + self.v_proj_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.q_num_heads * self.v_dim, self.hidden_size, bias=False
        )

        self.q_weight = nn.Parameter(torch.ones(self.q_num_heads, self.qk_dim))
        self.k_weight = nn.Parameter(torch.ones(self.k_num_heads, self.qk_dim))

        self.rope = initialize_rope(self.qk_dim, 10000, False, None, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache=None,
    ):
        B, T, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_proj_dim, self.k_proj_dim, self.v_proj_dim], dim=-1
        )
        q = q.reshape(B, T, self.q_num_heads, self.qk_dim).transpose(1, 2)
        k = k.reshape(B, T, self.k_num_heads, self.qk_dim).transpose(1, 2)
        v = v.reshape(B, T, self.v_num_heads, self.v_dim).transpose(1, 2)

        q = torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * self.q_weight[:, None]
        k = torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * self.k_weight[:, None]

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            attn_mask=mask if mask is not None and not isinstance(mask, str) else None,
            is_causal=(mask == "causal") if isinstance(mask, str) else False,
        )
        output = output.transpose(1, 2).reshape(B, T, self.q_num_heads * self.v_dim)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gate_up_proj(x)
        hs = torch.split(h, h.shape[-1] // 2, dim=-1)
        return self.down_proj(swiglu(hs[0], hs[1]))


class PlamoDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, is_mamba: bool) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_mamba = is_mamba
        self.mixer: nn.Module
        if is_mamba:
            self.mixer = Mamba(config)
        else:
            self.mixer = Attention(config)
        self.mlp = MLP(config)
        self.pre_mixer_norm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mixer_norm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / 5
        )
        self.pre_mlp_norm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mlp_norm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / (5**1.5)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache=None,
    ):
        residual = hidden_states
        hidden_states = self.pre_mixer_norm(hidden_states)

        hidden_states_sa = self.mixer(
            hidden_states=hidden_states,
            mask=mask,
            cache=cache,
        )

        hidden_states_sa = self.post_mixer_norm(hidden_states_sa)
        hidden_states = residual + hidden_states_sa

        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)

        hidden_states_mlp = self.mlp(hidden_states)

        hidden_states_mlp = self.post_mlp_norm(hidden_states_mlp)
        return residual + hidden_states_mlp


def is_mamba(config: ModelArgs, i: int) -> bool:
    if not config.mamba_enabled:
        return False
    assert config.mamba_step > 1
    assert i < config.num_hidden_layers

    if config.num_hidden_layers <= (config.mamba_step // 2):
        return i != config.num_hidden_layers - 1
    return (i % config.mamba_step) != (config.mamba_step // 2)


class PlamoDecoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                PlamoDecoderLayer(config, is_mamba=is_mamba(config, i))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ssm_idx = 0 if config.mamba_enabled else None
        self.fa_idx = config.mamba_step // 2

    def forward(self, x: torch.Tensor, cache):
        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(x, cache[self.fa_idx])
        if self.ssm_idx is not None:
            mamba_mask = create_ssm_mask(x, cache[self.ssm_idx])
        else:
            mamba_mask = None

        for l, c in zip(self.layers, cache):
            x = l(
                x,
                mask=mamba_mask if l.is_mamba else attn_mask,
                cache=c,
            )
        return x


class PlamoModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        batch_size, seq_length = inputs.shape

        h = self.embed_tokens(inputs)

        out = self.layers(
            h,
            cache,
        )

        return self.norm(out)


class Model(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = PlamoModel(config)

        self.vocab_size = config.vocab_size

        if not config.tie_word_embeddings:
            self.lm_head: nn.Module = nn.Linear(
                config.hidden_size, self.vocab_size, bias=False
            )

    def sanitize(self, weights: dict) -> dict:
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.transpose(-1, -2).contiguous()
        return weights

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_mamba else KVCache() for l in self.layers]

    def forward(self, inputs: torch.Tensor, cache=None) -> torch.Tensor:
        outputs = self.model(
            inputs=inputs,
            cache=cache,
        )
        if self.config.tie_word_embeddings:
            logits = F.linear(outputs, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(outputs)

        return logits

    @property
    def layers(self):
        return self.model.layers.layers
