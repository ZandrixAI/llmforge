"""
LLMForge - Base Engine for Inference and RL
Provides common model loading, device handling, quantization, and forward pass logic.
""" 

import os
from typing import Generator, List, Optional

import torch
import torch.nn as nn

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

_TPU_AVAILABLE = False
try:
    import torch_xla.core.xla_model as xm

    _TPU_AVAILABLE = True
except ImportError:
    pass


def available_devices() -> dict:
    """Return a dict of available devices with info."""
    devices = {"cpu": {"name": "CPU", "available": True}}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        devices["cuda"] = {
            "name": f"CUDA ({props.name})",
            "available": True,
            "memory": f"{props.total_memory / 1e9:.1f} GB",
        }
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices["mps"] = {"name": "MPS (Apple Silicon)", "available": True}
    if _TPU_AVAILABLE:
        devices["tpu"] = {"name": f"TPU ({xm.xla_device()})", "available": True}
    return devices


def _detect_device() -> str:
    """Auto-detect best device: CUDA > TPU > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if _TPU_AVAILABLE:
        return "tpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _validate_device(device: str) -> str:
    """Validate and return device string."""
    device = device.lower().strip()
    if device == "auto":
        return _detect_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use device='cpu' or device='mps'.")
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                "MPS is not available. Use device='cpu' or device='cuda'."
            )
    if device == "tpu":
        if not _TPU_AVAILABLE:
            raise RuntimeError(
                "TPU is not available. Install torch_xla or use device='cpu'."
            )
    if device not in ("cpu", "cuda", "mps", "tpu"):
        raise ValueError(
            f"Unknown device '{device}'. Choose from: cpu, cuda, mps, tpu, auto"
        )
    return device


class BaseEngine:
    """
    Base engine providing common functionality for model loading, device handling,
    quantization, tensor parallelism, and core forward pass.

    Subclasses should implement generation-specific logic.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        tensor_parallel: bool = True,
        bits: int = 16,
        offload: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model name or local path
            device: "cuda", "cpu", "mps", "tpu", "auto", or None (auto-detect)
            tensor_parallel: Auto-shard across GPU + CPU if model too large
            bits: Quantization bits (16=full, 8=INT8, 4=INT4).
                  4-bit reduces size by 4x (400B model → ~100GB, fits in RAM+disk)
            offload: Enable layer offloading (layers move GPU↔CPU on-the-fly)
        """
        self.model_name = model_name
        self.device = _validate_device(device) if device else _detect_device()

        from llmforge.utils import load

        self.model, self.tokenizer = load(
            model_name, tokenizer_config={"trust_remote_code": True}
        )
        self.model.eval()

        if bits < 16:
            from llmforge.quant_module import quantize_model

            self.model = quantize_model(self.model, bits=bits)

        model_bytes = sum(
            p.element_size() * p.nelement() for p in self.model.parameters()
        )
        model_bytes += sum(
            b.element_size() * b.nelement() for b in self.model.buffers()
        )

        if offload:
            from llmforge.offload import OffloadedLayer
            import psutil

            ram_avail = psutil.virtual_memory().available
            gpu_mem = 0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory

            if model_bytes > gpu_mem * 0.8:
                gpu_budget = gpu_mem * 0.6
                gpu_used = 0
                for name, module in list(self.model.named_modules()):
                    if isinstance(module, nn.Linear):
                        mod_mem = sum(
                            p.element_size() * p.nelement() for p in module.parameters()
                        )
                        if gpu_used + mod_mem > gpu_budget:
                            parent_name = ".".join(name.split(".")[:-1])
                            parent = (
                                self.model.get_submodule(parent_name)
                                if parent_name
                                else self.model
                            )
                            submodule_name = name.split(".")[-1]
                            setattr(parent, submodule_name, OffloadedLayer(module))
                            gpu_used = 0
                        else:
                            gpu_used += mod_mem

        if tensor_parallel:
            try:
                from llmforge.tensor_parallel import auto_tensor_parallel

                self.model = auto_tensor_parallel(self.model)
                print(f"[llmforge] Auto tensor parallelism enabled")
            except Exception:
                pass

        self.model = self.model.to(self.device)
        self.model_params = sum(p.numel() for p in self.model.parameters())
        self._use_rust = False
        self._tensor_parallel = tensor_parallel

        print(
            f"[llmforge] Model: {model_name} | "
            f"Params: {self.model_params / 1e9:.1f}B | "
            f"Device: {self.device} | "
            f"Bits: {bits}"
        )

    def _make_cache(self):
        """Create a fresh KV cache for generation."""
        cache = {"offset": 0}
        return cache

    def _get_eos_ids(self) -> set:
        """Get end-of-sequence token IDs."""
        eos_ids = set()
        if (
            hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            eos_ids.add(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, "eos_token_ids"):
            for eos_id in self.tokenizer.eos_token_ids:
                eos_ids.add(eos_id)
        return eos_ids

    def _forward(self, input_ids: torch.Tensor, cache: dict = None):
        """Core model forward pass. Subclasses may override."""
        with torch.no_grad():
            # Ensure input is 2D [batch, seq]
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
            
            #print(f"DEBUG _forward: input_ids shape: {input_ids.shape}", flush=True)

            outputs = self.model(input_ids)

            # Handle different model output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                # Raw tensor output
                logits = outputs

            # Get last token logits
            if len(logits.shape) == 3:
                # [batch, seq, vocab] -> [batch, vocab]
                logits = logits[0, -1, :]
            elif len(logits.shape) == 2:
                # [seq, vocab] -> [vocab]
                logits = logits[-1, :]
            else:
                logits = logits.squeeze()

            return logits.unsqueeze(0)  # Return [1, vocab]

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        bits: int = 4,
        device: Optional[str] = None,
        offload: bool = True,
        use_accelerate: bool = False,
    ):
        """
        Load any model (including 400B-500B param models) with automatic
        quantization and offloading.

        Examples:
            # 400B model on 6GB GPU + 16GB RAM + disk
            engine = BaseEngine.from_pretrained("meta-llama/Llama-3-405B", bits=4)

            # Use Accelerate for 100B+ models needing disk offloading
            engine = BaseEngine.from_pretrained("meta-llama/Llama-3-70B", bits=4, use_accelerate=True)
        """
        if use_accelerate:
            from llmforge.offload import AccelerateOffload

            model, tokenizer = AccelerateOffload.load_model(model_name)
            engine = cls.__new__(cls)
            engine.model_name = model_name
            engine.device = device or _detect_device()
            engine.model = model
            engine.tokenizer = tokenizer
            engine.model_params = sum(p.numel() for p in model.parameters())
            engine._use_rust = False
            engine._tensor_parallel = False
            print(f"[llmforge] Model: {model_name} (Accelerate auto-sharded)")
            return engine

        return cls(model_name, device=device, bits=bits, offload=offload)
