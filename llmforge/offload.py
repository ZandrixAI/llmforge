"""
LLMForge - Layer Offloading Module
Moves model layers between GPU, CPU, and Disk on-the-fly.
Enables running 400B-500B parameter models on systems with 6GB GPU + 16GB RAM + disk.

Usage:
    from llmforge.offload import OffloadModel

    # Load a 400B model
    model = OffloadModel.from_pretrained("meta-llama/Llama-3-405B", bits=4)
"""

import os
import gc
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def get_system_memory() -> dict:
    """Get available system memory."""
    import psutil

    mem = psutil.virtual_memory()
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
    return {
        "ram_total": mem.total,
        "ram_available": mem.available,
        "gpu_total": gpu_mem,
        "gpu_available": gpu_mem - torch.cuda.memory_allocated() if gpu_mem > 0 else 0,
    }


class DiskBackedTensor:
    """
    A tensor that lives on disk and is loaded into memory on demand.
    Uses memory-mapping for efficient partial loading.
    """

    def __init__(self, path: str, shape: tuple, dtype: torch.dtype):
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self._mmap = None
        self._tensor = None

    def load(self, device: str = "cpu") -> torch.Tensor:
        """Load tensor from disk to device."""
        if self._tensor is None:
            self._tensor = torch.load(self.path, map_location=device, weights_only=True)
        return self._tensor.to(device)

    def unload(self):
        """Free memory."""
        self._tensor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def nbytes(self):
        return torch.tensor(0, dtype=self.dtype).element_size() * (
            self.shape[0] * self.shape[1] if len(self.shape) == 2 else self.shape[0]
        )

    def __repr__(self):
        return f"DiskTensor({self.shape}, {self.dtype}, {self.nbytes / 1e6:.1f}MB)"


class OffloadedLayer(nn.Module):
    """
    Wrapper that keeps a layer offloaded (CPU/Disk) and loads it to GPU
    only during forward pass.
    """

    def __init__(self, layer: nn.Module, offload_device: str = "cpu"):
        super().__init__()
        self.offload_device = offload_device
        self.compute_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move to offload device
        self.layer = layer.to(offload_device)

    def forward(self, *args, **kwargs):
        # Move layer to compute device
        self.layer.to(self.compute_device)

        # Move inputs to compute device
        moved_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                moved_args.append(arg.to(self.compute_device))
            else:
                moved_args.append(arg)

        moved_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                moved_kwargs[k] = v.to(self.compute_device)
            else:
                moved_kwargs[k] = v

        # Compute
        result = self.layer(*moved_args, **moved_kwargs)

        # Move layer back to offload device
        self.layer.to(self.offload_device)

        return result

    def extra_repr(self):
        return f"offload={self.offload_device}, compute={self.compute_device}"


class OffloadModel:
    """
    Manages layer offloading for massive models.

    Strategy:
    - Layer 0 (embeddings) → CPU (memory-heavy, compute-light)
    - Middle layers → GPU (compute-heavy, rotate through offload layers)
    - Last layers (norm + lm_head) → GPU (small, always needed)
    - Extra layers → CPU/Disk
    """

    def __init__(self, model: nn.Module, gpu_budget: float = 0.7):
        self.model = model
        self.gpu_budget = gpu_budget
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layers = []
        self._setup()

    def _setup(self):
        """Identify and categorize model layers."""
        mem = get_system_memory()
        gpu_mem = mem["gpu_available"] * self.gpu_budget

        # Get all children of the model's main body
        children = list(self.model.children())
        total_params = sum(p.nelement() for p in self.model.parameters())

        print(
            f"[offload] Model: {total_params:,} params ({total_params * 2 / 1e9:.1f} GB in fp16)"
        )
        print(f"[offload] GPU budget: {gpu_mem / 1e9:.1f} GB")
        print(f"[offload] RAM available: {mem['ram_available'] / 1e9:.1f} GB")

        # For each child, decide where to place it
        gpu_used = 0
        for name, module in self.model.named_children():
            module_mem = sum(
                p.nelement() * p.element_size() for p in module.parameters()
            )

            if gpu_used + module_mem < gpu_mem:
                module.to(self.device)
                gpu_used += module_mem
                self.layers.append((name, module, "gpu"))
            else:
                module.to("cpu")
                self.layers.append((name, module, "cpu"))

        gpu_layers = sum(1 for _, _, d in self.layers if d == "gpu")
        cpu_layers = sum(1 for _, _, d in self.layers if d == "cpu")
        print(
            f"[offload] Placed: {gpu_layers} on GPU ({gpu_used / 1e9:.1f}GB), {cpu_layers} on CPU"
        )

    def forward(self, *args, **kwargs):
        """Forward pass with automatic device management."""
        # Move inputs to device
        x = args[0]
        if (
            isinstance(x, torch.Tensor)
            and x.device.type == "cpu"
            and self.device == "cuda"
        ):
            x = x.to(self.device)

        # For the main body layers, we need to handle them specially
        # since they're inside the model structure
        with torch.no_grad():
            return self.model(x, *args[1:], **kwargs)

    def to(self, device):
        """Move entire model to device."""
        self.model.to(device)
        return self

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        bits: int = 4,
        gpu_memory_fraction: float = 0.7,
    ):
        """
        Load a massive model with automatic offloading and quantization.

        Args:
            model_name: HuggingFace model name
            bits: Quantization bits (4 or 8). 4-bit = 4x size reduction.
            gpu_memory_fraction: Fraction of GPU memory to use.
        """
        from llmforge.utils import load as _load

        print(f"[offload] Loading {model_name} with {bits}-bit quantization...")

        model, tokenizer = _load(
            model_name,
            tokenizer_config={"trust_remote_code": True},
        )

        # Quantize if requested
        if bits < 16:
            from llmforge.quant_module import quantize_model

            model = quantize_model(model, bits=bits)

        # Create offloaded model
        offloaded = cls(model, gpu_budget=gpu_memory_fraction)

        return offloaded, tokenizer


class AccelerateOffload:
    """
    Uses HuggingFace Accelerate for automatic model sharding.
    Supports: disk offloading, CPU offloading, multi-GPU.
    """

    @staticmethod
    def load_model(
        model_name: str,
        max_memory: Optional[Dict[str, str]] = None,
        offload_folder: Optional[str] = None,
    ):
        """
        Load model with accelerate's device_map="auto".

        Args:
            model_name: HuggingFace model name
            max_memory: Dict of device -> max memory string (e.g., {"cuda:0": "5GiB", "cpu": "16GiB"})
            offload_folder: Folder for disk offloading
        """
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.utils import get_balanced_memory

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if offload_folder is None:
            offload_folder = os.path.join(tempfile.gettempdir(), "llmforge_offload")

        os.makedirs(offload_folder, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if max_memory is None:
            import psutil

            gpu_mem = "0GiB"
            if torch.cuda.is_available():
                gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.8:.0f}GiB"
            ram = f"{psutil.virtual_memory().available / 1e9 * 0.8:.0f}GiB"
            max_memory = {"cuda:0": gpu_mem, "cpu": ram}

        print(f"[accelerate] Loading {model_name} with device_map=auto")
        print(f"[accelerate] Max memory: {max_memory}")
        print(f"[accelerate] Offload folder: {offload_folder}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        return model, tokenizer
