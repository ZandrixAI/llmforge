"""
LLMForge - Tensor Parallelism Module
Automatically shards large models across CPU and GPU based on available memory.

Usage:
    from llmforge.tensor_parallel import shard_model, auto_device

    # Auto-shard model across available devices
    model = shard_model(model, devices={"cuda": 3e9, "cpu": 8e9})
    # or
    model = auto_device(model)  # fully automatic
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


def get_available_memory() -> Dict[str, float]:
    """Return available memory per device in bytes."""
    mem = {"cpu": 0}
    try:
        import psutil

        mem["cpu"] = psutil.virtual_memory().available
    except ImportError:
        mem["cpu"] = 8e9  # assume 8GB if psutil not installed

    if torch.cuda.is_available():
        mem["cuda"] = torch.cuda.get_device_properties(0).total_memory
        mem["cuda"] -= torch.cuda.memory_allocated()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mem["mps"] = 8e9  # MPS shares system memory, estimate

    return mem


def get_module_memory(module: nn.Module) -> float:
    """Return total memory of a module in bytes."""
    total = 0
    for p in module.parameters():
        total += p.element_size() * p.nelement()
    return total


def get_model_layers(model: nn.Module) -> list:
    """Get all submodules that contain parameters (leaf modules)."""
    layers = []
    for name, module in model.named_modules():
        # Skip if it has children with parameters (it's a container)
        has_children_with_params = any(
            sum(p.nelement() for p in child.parameters()) > 0
            for child in module.children()
        )
        if (
            not has_children_with_params
            and sum(p.nelement() for p in module.parameters()) > 0
        ):
            layers.append((name, module))
    return layers


def shard_model(
    model: nn.Module,
    devices: Optional[Dict[str, float]] = None,
    min_gpu_memory: float = 100e6,
) -> nn.Module:
    """
    Shard model across devices based on parameter size.

    Strategy:
    - Place all layers on GPU if the entire model fits
    - Otherwise, place layers greedily: GPU first (filling budget), then CPU
    - Large embedding/output layers go to CPU to save GPU memory
    - Attention and MLP layers go to GPU for speed

    Args:
        model: PyTorch model
        devices: Dict of device -> available memory in bytes.
                 If None, auto-detect.
        min_gpu_memory: Minimum free GPU memory to use GPU at all.

    Returns:
        Model with layers placed on appropriate devices.
    """
    if devices is None:
        devices = get_available_memory()

    gpu_device = "cuda" if "cuda" in devices else ("mps" if "mps" in devices else None)
    gpu_mem = devices.get(gpu_device, 0) if gpu_device else 0

    # Get all leaf layers
    layers = get_model_layers(model)

    if gpu_mem < min_gpu_memory:
        # Not enough GPU memory, keep everything on CPU
        print(
            f"[tensor-parallel] Not enough GPU memory ({gpu_mem / 1e6:.0f}MB), using CPU"
        )
        return model

    # Calculate total model size
    total_params = get_module_memory(model)

    # If model fits entirely on GPU, just move it
    if total_params < gpu_mem * 0.8:
        model = model.to(gpu_device)
        print(
            f"[tensor-parallel] Model fits on {gpu_device} ({total_params / 1e6:.0f}MB)"
        )
        return model

    # Otherwise, shard across GPU and CPU
    gpu_budget = gpu_mem * 0.7  # leave 30% headroom
    gpu_used = 0

    # Sort layers: prioritize attention/MLP for GPU, embeddings for CPU
    gpu_priority = []
    cpu_layers = []

    for name, module in layers:
        mem = get_module_memory(module)
        is_embedding = "embed" in name.lower() or "lm_head" in name.lower()
        is_norm = "norm" in name.lower()

        if is_embedding or is_norm:
            # Embeddings and norms are memory-heavy but compute-light
            cpu_layers.append((name, module, mem))
        else:
            gpu_priority.append((name, module, mem))

    # Sort GPU-priority layers by size (largest first for efficiency)
    gpu_priority.sort(key=lambda x: -x[2])

    placement = {}

    # Place on GPU until budget is exhausted
    for name, module, mem in gpu_priority:
        if gpu_used + mem < gpu_budget:
            placement[name] = gpu_device
            gpu_used += mem
        else:
            placement[name] = "cpu"

    # Place CPU layers on CPU
    for name, module, mem in cpu_layers:
        placement[name] = "cpu"

    # Apply placement
    gpu_layers = sum(1 for d in placement.values() if d == gpu_device)
    cpu_layers_count = sum(1 for d in placement.values() if d == "cpu")

    for name, module in model.named_modules():
        if name in placement:
            module.to(placement[name])

    print(
        f"[tensor-parallel] Sharded across {gpu_device} ({gpu_layers} layers, "
        f"{gpu_used / 1e6:.0f}MB) and CPU ({cpu_layers_count} layers)"
    )

    return model


class DeviceManager:
    """
    Manages layer movement between devices during forward pass.
    Enables pipeline-style execution where layers are moved to GPU
    only when needed.
    """

    def __init__(self, model: nn.Module, gpu_device: str = "cuda"):
        self.model = model
        self.gpu_device = gpu_device
        self._layer_devices = {}
        self._record_devices()

    def _record_devices(self):
        for name, module in self.model.named_modules():
            params = list(module.parameters())
            if params:
                self._layer_devices[name] = str(params[0].device)

    def move_to_gpu(self, module: nn.Module):
        module.to(self.gpu_device)

    def move_to_cpu(self, module: nn.Module):
        module.to("cpu")
        if self.gpu_device == "cuda":
            torch.cuda.empty_cache()

    def execute_forward(self, input_ids: torch.Tensor, cache=None):
        """
        Execute forward pass with automatic device management.
        Layers on GPU stay there, layers on CPU are temporarily moved
        to GPU for computation then moved back.
        """
        x = input_ids
        if x.device.type == "cpu" and self.gpu_device == "cuda":
            x = x.to(self.gpu_device)

        # This is a simplified version - the actual implementation
        # would need to handle the model's specific forward signature
        with torch.no_grad():
            return self.model(x, cache=cache)
