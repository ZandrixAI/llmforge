# Chat API Guide

The chat module provides an interactive chat interface and programmatic chat generation.

## Command Line Interface

```bash
# Basic chat
llmforge chat --model meta-llama/Llama-3.2-1B-Instruct

# With custom settings
llmforge chat --model meta-llama/Llama-3.2-1B-Instruct \
    --temp 0.8 \
    --max-tokens 512 \
    --system-prompt "You are a helpful assistant"
```

## Chat Commands

| Command | Description |
|---------|-------------|
| `q` | Exit chat |
| `r` | Reset conversation |
| `h` | Show help |

## Programmatic Chat

```python
from llmforge.chat import stream_generate
from llmforge import load, make_sampler, make_prompt_cache

# Load model and tokenizer
model, tokenizer = load("meta-llama/Llama-3.2-1B-Instruct")

# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is PyTorch?"},
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True
)

# Create sampler
sampler = make_sampler(
    temperature=0.7,
    top_p=0.9
)

# Create prompt cache
prompt_cache = make_prompt_cache(model)

# Stream generated text
for response in stream_generate(
    model,
    tokenizer,
    prompt,
    max_tokens=256,
    sampler=sampler,
    prompt_cache=prompt_cache
):
    print(response.text, end="")
```

## Parameters

| Parameter | CLI Flag | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| model | `--model` | str | Llama-3.2-3B-Instruct-4bit | Model path or HF repo |
| temperature | `--temp` | float | 0.0 | Sampling temperature |
| top_p | `--top-p` | float | 1.0 | Nucleus sampling threshold |
| max_tokens | `-m` | int | 256 | Max tokens to generate |
| system_prompt | `--system-prompt` | str | None | System prompt |
| adapter_path | `--adapter-path` | str | None | LoRA adapter path |