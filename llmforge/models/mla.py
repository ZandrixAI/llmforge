# Copyright © 2026 Apple Inc.

import math

import torch
import torch.nn as nn


class MultiLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = nn.Parameter(
            torch.empty(num_heads, output_dims, input_dims).uniform_(-scale, scale)
        )

    def forward(self, x, transpose=True):
        if transpose:
            return x @ self.weight.transpose(-1, -2)
        else:
            return x @ self.weight
