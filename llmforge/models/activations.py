# Copyright © 2023-2026 Apple Inc.

import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu(gate, x):
    return F.silu(gate) * x


def xielu(x, alpha_p, alpha_n, beta, eps):
    alpha_p = F.softplus(alpha_p)
    alpha_n = beta + F.softplus(alpha_n)
    return torch.where(
        x > 0,
        alpha_p * torch.square(x) + beta * x,
        (torch.expm1(torch.minimum(x, eps)) - x) * alpha_n + beta * x,
    )


class XieLU(nn.Module):
    def __init__(
        self,
        alpha_p_init=0.8,
        alpha_n_init=0.8,
        beta=0.5,
        eps=-1e-6,
    ):
        super().__init__()
        alpha_p_tensor = torch.tensor(alpha_p_init)
        alpha_n_tensor = torch.tensor(alpha_n_init - beta)
        self.alpha_p = nn.Parameter(torch.log(torch.exp(alpha_p_tensor) - 1))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(alpha_n_tensor) - 1))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.eps = nn.Parameter(torch.tensor(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return xielu(x, self.alpha_p, self.alpha_n, self.beta, self.eps)
