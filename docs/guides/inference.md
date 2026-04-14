# Inference Guide

The `InferenceEngine` provides a clean interface for text generation and chat interactions. It inherits from `BaseEngine` which handles device detection, model loading, and quantization.

## Overview

```
InferenceEngine (inherits from BaseEngine)
├── Model Loading & Tokenization
├── Device Management (CPU/GPU/MPS/TPU)
├── Quantization (FP16/INT8/INT4)
├── Text Generation
└── Chat Interface
```

## Basic Usage

### Plain Text Generation

```python
from llmforge import InferenceEngine

# Initialize engine with default settings
engine = InferenceEngine("Qwen/Qwen3-0.6B")

# Generate text
prompt = "Write a short story about a robot"
result = engine.generate(prompt, max_tokens=128, temperature=0.7)
print(result)
```

### Chat Mode

```python
from llmforge import InferenceEngine

engine = InferenceEngine("Qwen/Qwen3-0.6B")

# Define messages
messages = [
    {"role": "system", "content": "You are a helpful Python coding assistant."},
    {"role": "user", "content": "How do I read a file in Python?"},
    {"role": "assistant", "content": "You can use the built-in `open()` function."},
    {"role": "user", "content": "How about writing?"},
]

# Generate response
response = engine.chat(messages, max_tokens=256, temperature=0.7)
print(response)
```

---

## Device Configuration

### Auto-Detect (Default)

```python
# Automatically selects best device: CUDA > TPU > MPS > CPU
engine = InferenceEngine("Qwen/Qwen3-0.6B")
```

### Explicit Device

```python
from llmforge import InferenceEngine

# Use specific device
engine = InferenceEngine(
    "Qwen/Qwen3-0.6B",
    device="cuda"   # or "cpu", "mps", "tpu", "auto"
)
```

### Check Available Devices

```python
from llmforge.base_engine import available_devices

devices = available_devices()
for name, info in devices.items():
    print(f"{name}: {info}")
```

---

## Quantization

Reduce memory usage with quantization:

```python
from llmforge import InferenceEngine

# 4-bit quantization (75% size reduction)
engine = InferenceEngine(
    "Qwen/Qwen3-0.6B",
    bits=4
)

# 8-bit quantization (50% size reduction)
engine = InferenceEngine(
    "Qwen/Qwen3-0.6B", 
    bits=8
)

# Full precision (no quantization)
engine = InferenceEngine(
    "Qwen/Qwen3-0.6B", 
    bits=16
)
```

---

## Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Input prompt |
| `max_tokens` | int | 128 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0=greedy, 1=random) |

### Temperature Guide

| Temperature | Use Case |
|------------|----------|
| 0.0 | Code generation, factual answers |
| 0.1-0.3 | Precise, focused responses |
| 0.4-0.7 | Balanced (default) |
| 0.8-1.0 | Creative, diverse outputs |

---

## Advanced Options

### Tensor Parallelism

Automatically shard large models across multiple devices:

```python
from llmforge import InferenceEngine

engine = InferenceEngine(
    "meta-llama/Llama-3.2-70B-Instruct",
    tensor_parallel=True  # Auto-shard across devices
)
```

### Layer Offloading

For very large models, enable layer offloading:

```python
from llmforge import InferenceEngine

engine = InferenceEngine(
    "meta-llama/Llama-3.1-70B",
    offload=True,  # Move layers to CPU as needed
    bits=4
)
```

---

## Error Handling

```python
from llmforge import InferenceEngine

try:
    engine = InferenceEngine("Qwen/Qwen3-0.6B", device="cuda")
except RuntimeError as e:
    print(f"Device error: {e}")
    # Fall back to CPU
    engine = InferenceEngine("Qwen/Qwen3-0.6B", device="cpu")
```

---

## Next Steps

- [RL Engine Guide](rl_engine.md) - Self-improving responses
- [Chat API](chat.md) - Interactive chat interface
- [Quantization](../advanced/quantization.md) - Memory optimization
