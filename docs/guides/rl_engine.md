# RL Engine Guide

The `RLEngine` extends `InferenceEngine` with self-improvement strategies and persistent memory.

## Key Features

- **Self-Improvement Strategies**: Best-of-N, refine, expand, consensus
- **SQLite Memory**: Stores good responses for recall
- **Quality Scoring**: Automatic quality assessment

## Basic Usage

```python
from llmforge import RLEngine

# Initialize RL engine with memory
engine = RLEngine("meta-llama/Llama-3.2-1B-Instruct")

# Generate with auto strategy selection
result = engine.generate(
    "Explain quantum computing",
    max_tokens=256,
    strategy="auto"  # or "best_of_n", "refine", "expand", "consensus"
)
print(result)
```

## Strategies

### Best-of-N

Generates multiple candidates and selects the best:

```python
result = engine.generate(
    "Write a haiku about spring",
    max_tokens=64,
    strategy="best_of_n"
)
```

### Refine

Generates, scores, and improves if needed:

```python
result = engine.generate(
    "What is machine learning?",
    max_tokens=128,
    strategy="refine"
)
```

### Expand

Forces detailed, expanded responses:

```python
result = engine.generate(
    "List the planets",
    max_tokens=128,
    strategy="expand"
)
```

### Consensus

Generates multiple and picks via consensus:

```python
result = engine.generate(
    "Explain gravity",
    max_tokens=128,
    strategy="consensus"
)
```

## Memory System

The RL engine automatically stores good responses and recalls them:

```python
# First generation is stored
result = engine.generate("Hello world", max_tokens=64)

# Later, similar prompts trigger memory recall
result = engine.generate("Hi there", max_tokens=64)
# Output: [Memory hit!] ...
```

### Search Memory

```python
# Search for similar past responses
memories = engine.search_memory("python tutorial", limit=5)

for mem in memories:
    print(f"Strategy: {mem['strategy']}")
    print(f"Score: {mem['quality_score']}")
    print(f"Response: {mem['response'][:100]}...")
```

### Get Stats

```python
stats = engine.get_stats()
print(f"Total generations: {stats['generations']}")
print(f"Memory hits: {stats['memory_hits']}")
```

## Custom Memory DB

```python
from llmforge import RLEngine

# Custom SQLite database path
engine = RLEngine(
    "meta-llama/Llama-3.2-1B-Instruct",
    memory_db="/path/to/custom_memory.db"
)
```

## Auto Strategy Selection

When `strategy="auto"`, the engine chooses based on prompt:

| Prompt Type | Strategy |
|------------|----------|
| Short (<20 chars) | expand |
| Contains "explain", "describe", "what is" | refine |
| Contains "create", "write", "list", "make" | expand |
| Default | refine |