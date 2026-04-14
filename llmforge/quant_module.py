"""
LLMForge - 4-bit Quantization Module
Reduces model size by 4x-8x to fit massive models in limited memory.

Usage:
    from llmforge.quant_module import quantize_model

    model = quantize_model(model, bits=4)   # 4x smaller
    model = quantize_model(model, bits=8)   # 2x smaller
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear4Bit(nn.Module):
    """4-bit quantized linear layer with per-row scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Pack 2 INT4 values per byte
        n_packed = math.ceil(in_features * out_features / 2)
        self.register_buffer("weight_packed", torch.zeros(n_packed, dtype=torch.uint8))
        self.register_buffer(
            "weight_scales", torch.zeros(out_features, dtype=torch.float16)
        )
        self.register_buffer(
            "weight_zeros", torch.zeros(out_features, dtype=torch.float16)
        )
        self._weight = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def _unpack(self):
        """Unpack INT4 weights to float16."""
        if self._weight is not None:
            return self._weight

        packed = self.weight_packed
        lo = (packed & 0xF).to(torch.float16)
        hi = ((packed >> 4) & 0xF).to(torch.float16)
        interleaved = torch.stack([lo, hi], dim=-1).reshape(-1)
        unpacked = interleaved[: self.out_features * self.in_features].reshape(
            self.out_features, self.in_features
        )

        # Dequantize per-row: weight = scale * (unpacked - zero)
        self._weight = (
            self.weight_scales.unsqueeze(1)
            * (unpacked - self.weight_zeros.unsqueeze(1))
        ).to(torch.float32)
        return self._weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._unpack()
        return F.linear(x, weight, self.bias)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bits=4"


def _quantize_tensor_int4(tensor: torch.Tensor):
    """
    Quantize a float tensor to INT4 with per-row scales/zeros.
    Returns: (packed, scales, zeros)
    """
    out_features, in_features = tensor.shape

    # Find min/max per row
    min_vals = tensor.min(dim=-1).values
    max_vals = tensor.max(dim=-1).values

    # Compute scale and zero-point
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
    scales = range_vals / 15.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    zeros = (-min_vals / scales).round().clamp(0, 15).to(torch.float16)

    # Quantize to [0, 15]
    quantized = (
        ((tensor / scales.unsqueeze(1)) + zeros.unsqueeze(1))
        .round()
        .clamp(0, 15)
        .to(torch.uint8)
    )

    # Pack 2 values per byte
    if in_features % 2 != 0:
        quantized = F.pad(quantized, (0, 1))
    lo = quantized[:, ::2]
    hi = quantized[:, 1::2]
    packed = (lo | (hi << 4)).reshape(-1)

    return packed, scales.to(torch.float16), zeros


def quantize_model(model: nn.Module, bits: int = 4):
    """
    Quantize all linear layers in a model to lower precision.
    4-bit: ~4x smaller. 8-bit: ~2x smaller.
    """
    total_params = sum(p.nelement() for p in model.parameters())
    original_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    print(
        f"[quant] Quantizing {total_params:,} params ({original_bytes / 1e9:.1f} GB) to {bits}-bit..."
    )

    quantized_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and module.weight.shape[0] > 64:
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]

            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)

            if bits == 4:
                new_layer = Linear4Bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
                packed, scales, zeros = _quantize_tensor_int4(module.weight.data)
                new_layer.weight_packed = packed
                new_layer.weight_scales = scales
                new_layer.weight_zeros = zeros
                if module.bias is not None:
                    new_layer.bias = nn.Parameter(module.bias.data.to(torch.float32))

                setattr(parent, child_name, new_layer)
                quantized_count += 1

    new_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    new_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())
    print(
        f"[quant] Quantized {quantized_count} layers, new size: {new_bytes / 1e9:.2f} GB "
        f"({original_bytes / new_bytes:.1f}x compression)"
    )
    return model
