# Examples

End-to-end examples for common use cases.

## Basic Generation

```python
from llmforge import InferenceEngine

engine = InferenceEngine("meta-llama/Llama-3.2-1B-Instruct")
result = engine.generate("Hello, how are you?", max_tokens=64)
print(result)
```

## Chat with System Prompt

```python
from llmforge import InferenceEngine

engine = InferenceEngine("meta-llama/Llama-3.2-1B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful Python coding assistant."},
    {"role": "user", "content": "How do I read a file in Python?"},
]

response = engine.chat(messages, max_tokens=256)
print(response)
```

## RL Engine with Memory

```python
from llmforge import RLEngine

engine = RLEngine("meta-llama/Llama-3.2-1B-Instruct")

for i in range(5):
    result = engine.generate(f"What is {i+1} + {i+1}?", max_tokens=64)
    print(f"Q: What is {i+1} + {i+1}?")
    print(f"A: {result}")
    print()

stats = engine.get_stats()
print(f"Memory hits: {stats['memory_hits']}")
```

## Batch Generation

```python
from llmforge import InferenceEngine

engine = InferenceEngine("meta-llama/Llama-3.2-1B-Instruct")

prompts = [
    "Write a haiku",
    "Explain AI",
    "What is Python?",
]

for prompt in prompts:
    result = engine.generate(prompt, max_tokens=64)
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

## Streaming Generation

```python
from llmforge.generate import stream_generate
from llmforge import load, make_sampler, make_prompt_cache

model, tokenizer = load("meta-llama/Llama-3.2-1B-Instruct")
prompt = tokenizer.encode("Tell me a story", add_special_tokens=True)

sampler = make_sampler(temperature=0.8)
cache = make_prompt_cache(model)

for output in stream_generate(
    model, tokenizer, prompt,
    max_tokens=128,
    sampler=sampler,
    prompt_cache=cache
):
    print(output.text, end="")
```

## Custom Sampler

```python
from llmforge import InferenceEngine
from llmforge.sample_utils import make_sampler

engine = InferenceEngine("meta-llama/Llama-3.2-1B-Instruct")

sampler = make_sampler(
    temperature=0.9,
    top_p=0.95,
    top_k=50,
)
```