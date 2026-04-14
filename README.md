# LLMForge

**LLMForge** is a powerful Python package for text generation and reinforcement learning self-improvement with large language models. Built on PyTorch, it supports CPU, GPU (CUDA), MPS (Apple Silicon), and TPU devices.

## Key Features

- **Multi-Device Support**: Run models on CPU, CUDA GPUs, Apple MPS, or TPU with automatic device detection
- **Massive Model Support**: Handle 400B+ parameter models with 4-bit quantization and intelligent offloading
- **Tensor Parallelism**: Auto-shard large models across multiple devices
- **RL Self-Improvement**: Improve outputs without fine-tuning using:
  - Self-critique (model critiques its own output)
  - Best-of-N sampling (generate N candidates, pick the best)
  - Iterative refinement (critique → improve → repeat)
  - Chain-of-thought verification
- **Streaming Generation**: Real-time token streaming with stop word support
- **Hugging Face Integration**: Easily use thousands of LLMs from the Hugging Face Hub
- **Thinking Filter**: Filter model's internal reasoning (`<think>` / `</think>` tags)

## Installation

**With pip**:

```sh
pip install llmforge
```

**With uv**:

```sh
uv pip install llmforge
```

**From source**:

```sh
git clone https://github.com/llmforge/llmforge.git
cd llmforge
pip install -e .
```

## Quick Start

### Python API

```python
from llmforge import InferenceEngine

# Create inference engine
engine = InferenceEngine("Qwen/Qwen3-0.6B")

# Streaming generation
print("Model: ", end="", flush=True)
for text in engine.generate("Explain quantum computing.", max_tokens=200):
    print(text, end="", flush=True)
print()

# Non-streaming
response = engine.generate(
    "What is Python?",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
    stream=False
)
print(response)
```

### RL Self-Improvement

```python
from llmforge import RLEngine

# Create RL engine directly
rl = RLEngine("Qwen/Qwen3-0.6B")

# Generate with self-improvement
print("Model: ", end="", flush=True)
for text in rl.generate(
    "Explain gravity.",
    strategy="self_critique",
    max_tokens=200
):
    print(text, end="", flush=True)
print()

# Score text quality
scores = rl.score("Your generated text here", reference="Original prompt")
print(f"Quality scores: {scores}")
```

### Chat API

```python
from llmforge import InferenceEngine

engine = InferenceEngine("Qwen/Qwen3-0.6B")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]

for text in engine.chat(messages, max_tokens=50):
    print(text, end="", flush=True)
```

### Command Line

```bash
# Generate text
llmforge generate --model Qwen/Qwen3-0.6B --prompt "Hello, how are you?"

# Chat REPL
llmforge chat --model Qwen/Qwen3-0.6B

# Quantize and run large models
llmforge generate --model meta-llama/Llama-3-70B --bits 4
```

## Configuration Options

### Device Selection

```python
from llmforge import InferenceEngine

# Auto-detect (default)
engine = InferenceEngine("Qwen/Qwen3-0.6B")

# Force specific device
engine = InferenceEngine("Qwen/Qwen3-0.6B", device="cuda")   # NVIDIA GPU
engine = InferenceEngine("Qwen/Qwen3-0.6B", device="mps")     # Apple Silicon
engine = InferenceEngine("Qwen/Qwen3-0.6B", device="cpu")    # CPU
```

### Quantization & Offloading

```python
from llmforge import InferenceEngine

# 4-bit quantization (reduces size by 4x)
engine = InferenceEngine("meta-llama/Llama-3-8B", bits=4)

# Layer offloading for massive models (GPU ↔ CPU)
engine = InferenceEngine("meta-llama/Llama-3-70B", bits=4, offload=True)

# Use Accelerate for disk offloading
engine = InferenceEngine.from_pretrained(
    "meta-llama/Llama-3-405B",
    bits=4,
    use_accelerate=True
)
```

### Generation Parameters

```python
response = engine.generate(
    prompt="Your prompt here",
    max_tokens=512,           # Maximum tokens to generate
    temperature=0.7,          # Sampling temperature (0 = greedy)
    top_p=0.9,                # Nucleus sampling threshold
    top_k=50,                 # Top-k sampling
    repetition_penalty=1.1,   # Penalty for repeated tokens
    stop_words=["\n\n"],      # Stop on these strings
    enable_thinking=True,     # Show model's reasoning
    stream=True,              # Stream tokens
)
```

## Architecture

```
BaseEngine
├── Device detection, model loading
├── Quantization (4-bit, 8-bit)
├── Tensor parallelism
├── KV cache management
│
└── InferenceEngine
    ├── Sampling (top-k, top-p, temperature)
    ├── Stop words, thinking filter
    ├── Streaming generation
    └── Chat API
        │
        └── RLEngine
            ├── QualityScorer (NLTK-based)
            ├── Best-of-N strategy
            ├── Self-critique strategy
            ├── Iterative refinement
            └── Chain-of-thought verification
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers 5.0+
- Additional dependencies listed in `pyproject.toml`

## Examples

See the `tests/` directory for example scripts:

- `quick_run.py` - Basic inference
- `test_engine.py` - Full engine API test
- `test_rl.py` - RL self-improvement demo
- `test_devices.py` - Multi-device support
- `test_quant_offload.py` - Quantization and offloading

## Acknowledgements

LLMForge is built upon the pioneering work of the **MLX Community** and draws significant inspiration from [mlx-lm](https://github.com/ml-explore/mlx-lm). We are grateful for their contributions to democratizing LLM inference on Apple Silicon.

Specifically, we acknowledge:

- **Apple MLX Team**: For creating the MLX framework that inspired this PyTorch port
- **Hugging Face**: For the Transformers library and model hub integration
- **PyTorch Team**: For the underlying deep learning framework
- **MLX Community**: For the extensive collection of optimized models in the [MLX Community](https://huggingface.co/mlx-community) Hub
- **NLTK Project**: For natural language processing tools used in quality scoring
- All contributors to open-source LLM research and tooling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use LLMForge in your research, please cite:

```bibtex
@software{llmforge,
  title = {LLMForge: LLM Inference and RL Self-Improvement Engine},
  author = {LLMForge Contributors},
  year = {2025},
  url = {https://github.com/llmforge/llmforge}
}
```
