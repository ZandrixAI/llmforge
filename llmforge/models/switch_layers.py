# Copyright © 2023-2024 Apple Inc.

import math

import torch
import torch.nn as nn

from .activations import swiglu


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = torch.argsort(indices)
    inv_order = torch.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = x.unflatten(0, shape)
    return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = nn.Parameter(
            torch.empty(num_experts, output_dims, input_dims).uniform_(-scale, scale)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, output_dims))
        else:
            self.bias = None

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def forward(self, x, indices, sorted_indices=False):
        # indices selects which expert to use for each token
        # x shape: (..., 1, input_dims) or (..., input_dims)
        w = self.weight[indices]  # (..., output_dims, input_dims)
        if x.dim() == w.dim():
            x = torch.matmul(x.unsqueeze(-2), w.transpose(-1, -2)).squeeze(-2)
        else:
            x = torch.matmul(x, w.transpose(-1, -2))
        if self.bias is not None:
            x = x + self.bias[indices].unsqueeze(-2)
        return x


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=None,
        bias: bool = False,
    ):
        super().__init__()

        if activation is None:
            activation = SwiGLU()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def forward(self, x, indices) -> torch.Tensor:
        x = x.unsqueeze(-2).unsqueeze(-3)

        # When we have many tokens, sort them to make sure expert access is in order
        do_sort = indices.numel() >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = idx.detach()
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=None,
        bias: bool = False,
    ):
        super().__init__()

        if activation is None:
            activation = nn.GELU()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def forward(self, x, indices) -> torch.Tensor:
        x = x.unsqueeze(-2).unsqueeze(-3)

        do_sort = indices.numel() >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = idx.detach()
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
