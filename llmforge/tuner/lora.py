# Copyright © 2024 Apple Inc.

import math

import torch
import torch.nn as nn

from ..models.switch_layers import QuantizedSwitchLinear, SwitchLinear


class LoRALinear(nn.Module):
    @staticmethod
    def from_base(
        linear: nn.Linear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if hasattr(linear, "bits"):
            input_dims = input_dims * 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def fuse(self, dequantize: bool = False):
        linear = self.linear
        bias = linear.bias is not None
        weight = linear.weight

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        delta = (self.scale * self.lora_b.T @ self.lora_a.T).to(weight.dtype)
        fused_linear.weight = nn.Parameter(weight + delta)
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

        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        self.scale = scale

        scale_val = 1 / math.sqrt(input_dims)
        self.lora_a = nn.Parameter(
            torch.empty(input_dims, r).uniform_(-scale_val, scale_val)
        )
        self.lora_b = nn.Parameter(torch.zeros(r, output_dims))

    def forward(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).to(x.dtype)


class LoRASwitchLinear(nn.Module):
    @staticmethod
    def from_base(
        linear: nn.Module,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        lora_lin = LoRASwitchLinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            num_experts=linear.num_experts,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def fuse(self, dequantize: bool = False):
        linear = self.linear
        bias = linear.bias is not None
        weight = linear.weight

        num_experts, output_dims, input_dims = weight.shape
        fused_linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        lora_b = self.scale * self.lora_b
        lora_a = self.lora_a.reshape(num_experts, -1, input_dims)
        fused_linear.weight = nn.Parameter(weight + (lora_b @ lora_a).to(weight.dtype))
        if bias:
            fused_linear.bias = linear.bias

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        self.linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        self.scale = scale

        scale_val = 1 / math.sqrt(input_dims)
        self.lora_a = nn.Parameter(
            torch.empty(num_experts, r, input_dims).uniform_(-scale_val, scale_val)
        )
        self.lora_b = nn.Parameter(torch.zeros(num_experts, output_dims, r))
        self.num_experts = num_experts

    def forward(self, x, indices, sorted_indices=False):
        y = self.linear(x, indices, sorted_indices=sorted_indices)
        dropout_x = self.dropout(x)
        # Gather matmul: select lora_a by expert indices
        # indices shape: (..., num_selected_experts)
        z = torch.einsum(
            "bi,eir->ber" if dropout_x.dim() == 2 else "bse,eir->bsr",
            dropout_x.flatten(0, -2) if dropout_x.dim() > 2 else dropout_x,
            self.lora_a.transpose(-1, -2),
        )
        z = torch.einsum(
            "bsr,eoi->bso" if dropout_x.dim() > 2 else "ber,eoi->beo",
            z,
            self.lora_b.transpose(-1, -2),
        )
        return y + (self.scale * z).to(x.dtype)


class LoRAEmbedding(nn.Module):
    @staticmethod
    def from_base(
        embedding: nn.Embedding,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        num_embeddings, dims = embedding.weight.shape
        lora_embedding = LoRAEmbedding(
            num_embeddings=num_embeddings,
            dims=dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_embedding.embedding = embedding
        return lora_embedding

    def fuse(self, dequantize: bool = False):
        embedding = self.embedding
        weight = embedding.weight

        num_embeddings, dims = weight.shape
        fused_embedding = nn.Embedding(num_embeddings, dims)

        lora_a = self.scale * self.lora_a
        lora_b = self.lora_b
        fused_embedding.weight = nn.Parameter(
            weight + (lora_a @ lora_b).to(weight.dtype)
        )

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

        self.embedding = nn.Embedding(num_embeddings, dims)
        self.dropout = nn.Dropout(p=dropout)

        self.scale = scale

        scale_val = 1 / math.sqrt(num_embeddings)
        self.lora_a = nn.Parameter(
            torch.empty(num_embeddings, r).uniform_(-scale_val, scale_val)
        )
        self.lora_b = nn.Parameter(torch.zeros(r, dims))

    def forward(self, x):
        y = self.embedding(x)
        z = self.dropout(self.lora_a[x] @ self.lora_b)
        out = y + (self.scale * z).to(y.dtype)
        return out

    def as_linear(self, x):
        y = nn.functional.linear(x, self.embedding.weight)
        z = (self.dropout(x) @ self.lora_b.T) @ self.lora_a.T
        return y + (self.scale * z).to(x.dtype)
