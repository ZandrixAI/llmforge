# Installation

This guide covers installing LLMForge on various platforms and configurations.

## Prerequisites

- **Python**: 3.10 or later
- **PyTorch**: 2.0 or later
- **Optional**: CUDA Toolkit (for GPU support), MLX (for Apple Silicon)

!!! tip "Quick Installation"
    For most users, simply run:
    ```bash
    pip install llmforge
    ```

---

## Supported Devices

| Device | Support | Notes |
|--------|---------|-------|
| CPU | ✅ Full | Works out of the box |
| NVIDIA GPU | ✅ Full | Requires CUDA Toolkit |
| Apple Silicon (M-series) | ✅ Full | Requires PyTorch with MPS support |
| Google TPU | ⚠️ Beta | Requires torch-xla |

---

## Installation Methods

### 1. PyPI (Recommended)

```bash
pip install llmforge
```

### 2. From Source

```bash
git clone https://github.com/ZandrixAI/llmforge.git
cd llmforge
pip install -e .
```

### 3. With GPU Support

```bash
# NVIDIA CUDA
pip install llmforge
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon
pip install llmforge
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. With All Dependencies

```bash
pip install llmforge[all]
```

---

## Verify Installation

```python
import llmforge

# Check version
print(llmforge.__version__)

# Check available devices
from llmforge.base_engine import available_devices
devices = available_devices()
print(f"Available devices: {list(devices.keys())}")
```

---

## Optional Dependencies

| Package | Install | Purpose |
|---------|---------|---------|
| accelerate | `pip install accelerate` | Large model loading |
| bitsandbytes | `pip install bitsandbytes` | INT8/INT4 quantization |
| peft | `pip install peft` | LoRA fine-tuning |
| transformers | `pip install transformers` | Additional model support |
| sentencepiece | `pip install sentencepiece` | Tokenizer support |

---

## Troubleshooting

### CUDA Not Found

```bash
# Check CUDA installation
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### MPS Not Available (Apple Silicon)

```bash
# Ensure you're using the latest PyTorch
pip install torch --upgrade
```

### Out of Memory

See [Quantization Guide](advanced/quantization.md) for reducing memory usage.

---

## Next Steps

- [Quick Start](quickstart.md) - Run your first generation
- [Inference Guide](guides/inference.md) - Deep dive into generation
- [RL Engine](guides/rl_engine.md) - Self-improving responses
