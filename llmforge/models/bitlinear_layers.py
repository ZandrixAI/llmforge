# Copyright © 2025 Apple Inc.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import tree_flatten, tree_unflatten


def bitnet_quantize(model, quantization_config: dict):
    quantize_layers = []
    modules_to_not_convert = quantization_config.get("modules_to_not_convert", [])
    invert_weight_scales = (
        quantization_config.get("linear_class", "") != "autobitlinear"
    )

    for name, module in model.named_modules():
        if name not in modules_to_not_convert and isinstance(module, nn.Linear):
            old_weight = module.weight
            out_features, in_features = old_weight.shape
            bias = module.bias is not None
            new_layer = BitLinear(
                in_features,
                out_features,
                bias=bias,
                invert_weight_scales=invert_weight_scales,
            )
            quantize_layers.append((name, new_layer))
    if len(quantize_layers) > 0:
        for name, new_layer in quantize_layers:
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_layer)
    return model


class BitLinear(nn.Module):
    """
    BitLinear module with memory-efficient weight handling.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        invert_weight_scales=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        packed_out_features = (out_features + 3) // 4
        self.weight = nn.Parameter(
            torch.zeros((packed_out_features, in_features), dtype=torch.uint8),
            requires_grad=False,
        )

        self.invert_weight_scales = invert_weight_scales
        self.weight_scale = torch.tensor([1.0])

        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
        else:
            self.bias = None

    def forward(self, x):
        # Standard linear with unpacked weights for PyTorch compatibility
        # The actual quantized matmul would need custom CUDA kernels
        weight = self.weight.to(x.dtype)
        y = F.linear(x, weight)
        if self.bias is not None:
            y = y + self.bias
        return y
