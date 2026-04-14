# Copyright © 2024 Apple Inc.

import math

import torch
import torch.nn as nn


class DoRALinear(nn.Module):
    @staticmethod
    def from_base(
        linear: nn.Linear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if hasattr(linear, "bits"):
            input_dims *= 32 // linear.bits
        dora_lin = DoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        dora_lin.set_linear(linear)
        return dora_lin

    def fuse(self, dequantize: bool = False):
        linear = self.linear
        bias = linear.bias is not None
        weight = self._dequantized_weight()

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=False)

        lora_b = self.scale * self.lora_b.T
        lora_a = self.lora_a.T
        weight = weight + (lora_b @ lora_a).to(weight.dtype)
        norm_scale = self.m / torch.linalg.norm(weight, dim=1)
        fused_linear.weight = nn.Parameter(norm_scale[:, None] * weight)

        if bias:
            fused_linear.bias = linear.bias

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        self.set_linear(nn.Linear(input_dims, output_dims, bias=bias))
        self.dropout = nn.Dropout(p=dropout)

        self.scale = scale

        scale_val = 1 / math.sqrt(input_dims)
        self.lora_a = nn.Parameter(
            torch.empty(input_dims, r).uniform_(-scale_val, scale_val)
        )
        self.lora_b = nn.Parameter(torch.zeros(r, output_dims))

    def set_linear(self, linear):
        self.linear = linear
        self.m = torch.linalg.norm(self._dequantized_weight().to(torch.float32), dim=1)

    def _dequantized_weight(self):
        return self.linear.weight

    def forward(self, x):
        w = self._dequantized_weight()
        y = x @ w.T

        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        out = y + (self.scale * z).to(x.dtype)

        adapted = w + (self.scale * self.lora_b.T) @ self.lora_a.T
        denom = torch.linalg.norm(adapted.detach(), dim=1)

        out = (self.m / denom).to(x.dtype) * out

        if self.linear.bias is not None:
            out = out + self.linear.bias
        return out


class DoRAEmbedding(nn.Module):
    def from_base(
        embedding: nn.Embedding,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        num_embeddings, dims = embedding.weight.shape

        dora_embedding = DoRAEmbedding(
            num_embeddings=num_embeddings,
            dims=dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        dora_embedding.set_embedding(embedding)
        return dora_embedding

    def fuse(self, dequantize: bool = False):
        embedding = self.embedding
        weight = embedding.weight

        num_embeddings, dims = weight.shape
        fused_embedding = nn.Embedding(num_embeddings, dims)

        lora_a = self.scale * self.lora_a
        lora_b = self.lora_b
        weight = weight + (lora_a @ lora_b).to(weight.dtype)
        norm_scale = self.m / torch.linalg.norm(weight, dim=1)
        fused_embedding.weight = nn.Parameter(norm_scale[:, None] * weight)

        return fused_embedding

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        super().__init__()

        self.set_embedding(nn.Embedding(num_embeddings, dims))
        self.dropout = nn.Dropout(p=dropout)

        self.scale = scale

        scale_val = 1 / math.sqrt(num_embeddings)
        self.lora_a = nn.Parameter(
            torch.empty(num_embeddings, r).uniform_(-scale_val, scale_val)
        )
        self.lora_b = nn.Parameter(torch.zeros(r, dims))

    def set_embedding(self, embedding: nn.Module):
        self.embedding = embedding
        self.m = torch.linalg.norm(embedding.weight, dim=1)

    def forward(self, x):
        y = self.embedding(x)
        z = self.scale * self.lora_a[x] @ self.lora_b
        out = y + self.dropout(z).to(y.dtype)

        adapted = y + z
        denom = torch.linalg.norm(adapted.detach(), dim=-1)

        out = (self.m[x] / denom)[..., None] * out

        return out

    def as_linear(self, x):
        y = nn.functional.linear(x, self.embedding.weight)
        z = (self.dropout(x) @ self.lora_b.T) @ self.lora_a.T
        out = y + (self.scale * z).to(x.dtype)

        adapted = self.embedding.weight + (self.scale * self.lora_a) @ self.lora_b
        denom = torch.linalg.norm(adapted.detach(), dim=1)

        out = (self.m / denom) * out

        return out
