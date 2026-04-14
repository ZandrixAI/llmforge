# RLEngine API

`RLEngine` extends `InferenceEngine` with self-improvement strategies and SQLite memory.

## Class Definition

```python
class RLEngine(InferenceEngine):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        bits: int = 16,
        offload: bool = False,
        memory_db: str = None,
    )
```

## Methods

### generate()

```python
def generate(
    self,
    prompt: str,
    max_tokens: int = 128,
    strategy: str = "auto"
) -> str
```

Generate with RL self-improvement and memory.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Input prompt |
| `max_tokens` | int | 128 | Maximum tokens to generate |
| `strategy` | str | "auto" | Strategy: "auto", "best_of_n", "refine", "expand", "consensus" |

**Returns:** Generated text string.

### get_stats()

```python
def get_stats(self) -> Dict
```

Get generation statistics.

**Returns:**
```python
{
    "generations": 100,
    "memory_hits": 25
}
```

### search_memory()

```python
def search_memory(
    self,
    prompt: str,
    limit: int = 3
) -> List[Dict]
```

Search memory for similar prompts.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Search query |
| `limit` | int | 3 | Maximum results |

**Returns:**
```python
[
    {
        "prompt": "...",
        "response": "...",
        "strategy": "refine",
        "quality_score": 0.8
    }
]
```

## Inner Classes

### QualityScorer

```python
class QualityScorer:
    def score(self, text: str) -> float
```

Score a response (0.0 - 1.0).

### MemoryDB

```python
class MemoryDB:
    def __init__(self, db_path: str = "llmforge_memory.db")
    def store(self, prompt: str, response: str, strategy: str, quality_score: float)
    def recall(self, prompt: str, limit: int = 3) -> List[Dict]
    def get_best(self, prompt: str) -> Optional[str]
    def close(self)
```

SQLite-based memory storage.