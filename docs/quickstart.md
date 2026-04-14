# Quick Start

Get up and running with LLMForge in 5 minutes.

## Your First Generation

### Step 1: Install LLMForge

```bash
pip install llmforge
```

### Step 2: Run Inference

```python
from llmforge import InferenceEngine

# Create engine - auto-detects best device (CPU/GPU/MPS)
engine = InferenceEngine("Qwen/Qwen3-0.6B")

# Generate text
response = engine.generate("Write a short poem about AI")
print(response)
```

### Step 3: Chat with the Model

```python
from llmforge import InferenceEngine

engine = InferenceEngine("Qwen/Qwen3-0.6B")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

response = engine.chat(messages, max_tokens=128)
print(response)
```

---

## RL Engine with Self-Improvement

```python
from llmforge import RLEngine

# Create RL engine with memory
rl = RLEngine("Qwen/Qwen3-0.6B")

# Generate with auto-improvement strategies
for i in range(3):
    response = rl.generate(f"Explain concept {i+1}", max_tokens=128)
    print(f"Response {i+1}: {response[:100]}...")
    print("-" * 50)

# Check stats
stats = rl.get_stats()
print(f"Total generations: {stats['generations']}")
print(f"Memory hits: {stats['memory_hits']}")
```

---

## Command Line Usage

### Chat Interface

```bash
llmforge chat --model Qwen/Qwen3-0.6B
```

Options:
- `--temp 0.8` - Sampling temperature
- `--max-tokens 512` - Maximum tokens
- `--system-prompt "You are helpful"` - System prompt

### Generate from File

```bash
llmforge generate --model Qwen/Qwen3-0.6B --prompt-file prompts.txt
```

---

## Common Models

| Model | Size | Memory (FP16) | Memory (INT4) |
|-------|------|---------------|---------------|
| Qwen3-0.6B | 0.6B | 1.2GB | 0.3GB |
| Llama-3.2-1B | 1B | 2GB | 0.5GB |
| Llama-3.2-3B | 3B | 6GB | 1.5GB |
| Llama-3.2-8B | 8B | 16GB | 4GB |
| Llama-3.1-70B | 70B | 140GB | 35GB |

---

## What's Next?

- [Inference Guide](guides/inference.md) - Deep dive into text generation
- [RL Engine](guides/rl_engine.md) - Self-improving AI responses
- [Quantization](advanced/quantization.md) - Reduce memory usage
- [Server](advanced/server.md) - Run as API server
