# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import swiglu
from .base import BaseModelArgs, create_ssm_mask
from .cache import ArraysCache
from .ssm import ssm_update


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
    num_heads: int
    head_dim: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    state_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    conv_kernel: int
    n_groups: int
    use_bias: bool
    use_conv_bias: bool
    tie_word_embeddings: bool
    time_step_limit: Tuple[float, float]
    time_step_rank: Union[int, str]
    ssm_state_size: Optional[int] = None
    max_position_embeddings: int = 2056

    def __post_init__(self):
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        if self.ssm_state_size is None:
            self.ssm_state_size = self.state_size


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self, hidden_states: torch.Tensor, gate: torch.Tensor = None
    ) -> torch.Tensor:
        if gate is not None:
            hidden_states = swiglu(gate, hidden_states)
        return (
            hidden_states
            * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
            * self.weight
        )


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.num_heads * args.head_dim
        self.use_conv_bias = args.use_conv_bias
        self.n_groups = args.n_groups
        self.head_dim = args.head_dim
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups
        self.use_bias = args.use_bias

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            padding=0,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=args.use_bias)

        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))

        self.norm = MambaRMSNormGated(
            self.intermediate_size, eps=args.layer_norm_epsilon
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.use_bias
        )

    def _conv(
        self,
        conv_input: torch.Tensor,
        cache: Optional[ArraysCache],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is not None:
            conv_input = torch.where(
                mask[..., None], conv_input, torch.tensor(0.0, dtype=conv_input.dtype)
            )

        if cache is not None:
            if cache[0] is None:
                conv_state = torch.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
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
                positions = (ends[:, None] + torch.arange(n_keep))[..., None]
                cache[0] = torch.gather(padded_input, 1, positions)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = F.pad(conv_input, (0, 0, self.conv_kernel_size - 1, 0))

        conv_output = self.conv1d(padded_input.transpose(1, 2)).transpose(1, 2)
        return F.silu(conv_output)

    def _ssm(
        self,
        hidden_states: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor,
        cache: Optional[ArraysCache],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        if cache:
            state = cache[1]
            lengths = cache.lengths
        else:
            state, lengths = None, None
        y, state = ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask,
            lengths,
        )
        if cache:
            cache[1] = state
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache: Optional[ArraysCache] = None,
    ) -> torch.Tensor:
        projected = self.in_proj(hidden_states)
        gate, conv_input, dt = torch.split(
            projected,
            [
                self.intermediate_size,
                self.intermediate_size + self.conv_dim,
                projected.shape[-1] - self.intermediate_size - self.conv_dim,
            ],
            dim=-1,
        )
        conv_output = self._conv(conv_input, cache, mask)
        hidden_states, B, C = torch.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )
        y = self._ssm(hidden_states, B, C, dt, cache, mask=mask)
        if cache:
            cache.advance(y.shape[1])
        y = self.norm(y, gate)
        return self.out_proj(y)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.mixer = Mamba2Block(args, layer_idx)
        self.norm = RMSNorm(args.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache: Optional[ArraysCache] = None,
    ) -> torch.Tensor:
        output = self.mixer(self.norm(x), mask, cache)
        return output + x


class Mamba2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList(
            [ResidualBlock(args, i) for i in range(args.num_hidden_layers)]
        )
        self.norm_f = RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def forward(self, x: torch.Tensor, cache: Optional[list] = None) -> torch.Tensor:
        hidden = self.embeddings(x)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_ssm_mask(hidden, cache[0])
        for layer, c in zip(self.layers, cache):
            hidden = layer(hidden, mask, c)

        return self.norm_f(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba2(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(
        self, inputs: torch.Tensor, cache: Optional[list] = None
    ) -> torch.Tensor:
        hidden = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = F.linear(hidden, self.backbone.embeddings.weight)
        else:
            logits = self.lm_head(hidden)
        return logits

    def make_cache(self, batch_size: int = 1) -> list:
        return [ArraysCache(size=2) for _ in range(self.args.num_hidden_layers)]

    @property
    def layers(self):
        return self.backbone.layers

    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights
