# BaseEngine API

`BaseEngine` provides common functionality for model loading, device handling, quantization, and forward pass.

## Class Definition

```python
class BaseEngine:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        tensor_parallel: bool = True,
        bits: int = 16,
        offload: bool = False,
    )
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | required | HuggingFace model name or local path |
| `device` | str | "auto" | Device: "auto", "cuda", "cpu", "mps", "tpu" |
| `tensor_parallel` | bool | True | Auto-shard across devices |
| `bits` | int | 16 | Quantization bits: 16 (full), 8 (INT8), 4 (INT4) |
| `offload` | bool | False | Enable layer offloading |

## Methods

### from_pretrained()

```python
@classmethod
def from_pretrained(
    cls,
    model_name: str,
    bits: int = 4,
    device: Optional[str] = None,
    offload: bool = True,
    use_accelerate: bool = False,
) -> "BaseEngine"
```

Load any model with automatic quantization and offloading.

### _forward()

```python
def _forward(
    self,
    input_ids: torch.Tensor,
    cache: dict = None
) -> torch.Tensor
```

Core model forward pass. Returns last-token logits.

### _get_eos_ids()

```python
def _get_eos_ids(self) -> set
```

Get end-of-sequence token IDs.

### _make_cache()

```python
def _make_cache(self) -> dict
```

Create a fresh KV cache for generation.

## Module Functions

### available_devices()

```python
def available_devices() -> dict
```

Return a dict of available devices with info:

```python
{
    "cuda": {"name": "CUDA (RTX 3090)", "available": True, "memory": "24.0 GB"},
    "cpu": {"name": "CPU", "available": True},
    "mps": {"name": "MPS (Apple Silicon)", "available": True},
}
```

### _detect_device()

```python
def _detect_device() -> str
```

Auto-detect best device: CUDA > TPU > MPS > CPU.

### _validate_device()

```python
def _validate_device(device: str) -> str
```

Validate and return device string.