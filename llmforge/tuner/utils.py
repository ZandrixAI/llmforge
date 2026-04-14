# Copyright © 2024 Apple Inc.
import json
import types
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import torch
import torch.nn as nn

from ..models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from ..utils import get_total_parameters
from .dora import DoRAEmbedding, DoRALinear
from .lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear


def tree_flatten(tree):
    """Flatten a nested dictionary/list into a flat list of (key, value) tuples."""
    flat = []

    def _flatten(prefix, subtree):
        if isinstance(subtree, nn.Parameter):
            flat.append((prefix, subtree))
        elif isinstance(subtree, torch.Tensor):
            flat.append((prefix, subtree))
        elif isinstance(subtree, dict):
            for k, v in subtree.items():
                key = f"{prefix}.{k}" if prefix else k
                _flatten(key, v)
        elif isinstance(subtree, (list, tuple)):
            for i, v in enumerate(subtree):
                key = f"{prefix}.{i}" if prefix else str(i)
                _flatten(key, v)
        elif hasattr(subtree, "state_dict"):
            for k, v in subtree.state_dict().items():
                key = f"{prefix}.{k}" if prefix else k
                flat.append((key, v))
        elif hasattr(subtree, "__dict__"):
            for k, v in subtree.__dict__.items():
                if k.startswith("_"):
                    continue
                key = f"{prefix}.{k}" if prefix else k
                _flatten(key, v)

    _flatten("", tree)
    return flat


def tree_unflatten(tree):
    """Unflatten a list of (key, value) tuples into a nested dictionary."""
    result = {}
    for key, value in tree:
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def tree_map(fn, tree, *rest):
    """Map a function over a nested structure."""
    if isinstance(tree, dict):
        return {k: tree_map(fn, v, *(r[k] for r in rest)) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(
            tree_map(fn, v, *(r[i] for r in rest)) for i, v in enumerate(tree)
        )
    elif isinstance(tree, torch.Tensor):
        return fn(tree, *rest)
    else:
        return fn(tree, *rest)


def build_schedule(schedule_config: Dict):
    """
    Build a learning rate schedule from the given config.
    """
    name = schedule_config["name"]
    arguments = schedule_config["arguments"]
    initial_lr = arguments[0]

    if name == "cosine_decay":
        warmup_steps = schedule_config.get("warmup", 0)
        warmup_init = schedule_config.get("warmup_init", 0.0)

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                alpha = step / warmup_steps
                return warmup_init + alpha * (initial_lr - warmup_init)
            return initial_lr

        return lr_lambda
    else:
        return initial_lr


def linear_to_lora_layers(
    model: nn.Module,
    num_layers: int,
    config: Dict,
    use_dora: bool = False,
):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        num_layers (int): The number of blocks to convert to lora layers
        starting from the last layer.
        config (dict): More configuration parameters for LoRA, including the
          rank, scale, and optional layer keys.
        use_dora (bool): If True, uses DoRA instead of LoRA.
          Default: ``False``
    """

    def to_lora(layer):
        if not use_dora and hasattr(layer, "to_lora"):
            return layer.to_lora(
                r=config["rank"],
                scale=config["scale"],
                dropout=config["dropout"],
            )

        if isinstance(layer, nn.Linear):
            LoRALayer = DoRALinear if use_dora else LoRALinear
        elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
            if use_dora:
                raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
            LoRALayer = LoRASwitchLinear
        elif isinstance(layer, nn.Embedding):
            LoRALayer = DoRAEmbedding if use_dora else LoRAEmbedding
        else:
            raise ValueError(
                f"Can't convert layer of type {type(layer).__name__} to LoRA"
            )

        return LoRALayer.from_base(
            layer,
            r=config["rank"],
            scale=config["scale"],
            dropout=config["dropout"],
        )

    if (keys := config.get("keys", None)) is None:
        keys = set()

        def get_keys_for_lora(p, m):
            types = (
                nn.Linear,
                SwitchLinear,
                QuantizedSwitchLinear,
                nn.Embedding,
            )
            if hasattr(m, "to_lora") or isinstance(m, types):
                keys.add(p)

        for l in model.layers:
            l.apply_to_modules(get_keys_for_lora)

    for l in model.layers[-max(num_layers, 0) :]:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
        if lora_layers:
            l.update_modules(tree_unflatten(lora_layers))

    lora_modules = [(k, to_lora(m)) for k, m in model.named_modules() if k in keys]
    if lora_modules:
        model.update_modules(tree_unflatten(lora_modules))


def load_adapters(model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Load any fine-tuned adapters / layers.

    Args:
        model (nn.Module): The neural network model.
        adapter_path (str): Path to the adapter configuration file.

    Returns:
        nn.Module: The updated model with LoRA layers applied.
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"The adapter path does not exist: {adapter_path}")
    with open(adapter_path / "adapter_config.json", "r") as fid:
        config = types.SimpleNamespace(**json.load(fid))
    fine_tune_type = getattr(config, "fine_tune_type", "lora")
    if fine_tune_type != "full":
        linear_to_lora_layers(
            model,
            config.num_layers,
            config.lora_parameters,
            use_dora=(fine_tune_type == "dora"),
        )
    model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)
    return model


def remove_lora_layers(model: nn.Module) -> nn.Module:
    """
    Remove the LoRA layers from the model.

    Args:
        model (nn.Module): The model with LoRA layers.

    Returns:
        nn.Module: The model without LoRA layers.
    """
    reset_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            reset_layers.append((name, module.linear))
    if len(reset_layers) > 0:
        model.update_modules(tree_unflatten(reset_layers))
    return model


def print_trainable_parameters(model):
    total_p = get_total_parameters(model) / 1e6
    trainable_p = (
        sum(v.numel() for _, v in tree_flatten(model.trainable_parameters())) / 1e6
    )
    print(
        f"Trainable parameters: {(trainable_p * 100 / total_p):.3f}% "
        f"({trainable_p:.3f}M/{total_p:.3f}M)"
    )
