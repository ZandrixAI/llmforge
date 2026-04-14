# InferenceEngine API

`InferenceEngine` extends `BaseEngine` with generation and chat support.

## Class Definition

```python
class InferenceEngine(BaseEngine):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        tensor_parallel: bool = True,
        bits: int = 16,
        offload: bool = False,
    )
```

## Methods

### generate()

```python
def generate(
    self,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7
) -> str
```

Plain text generation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Input prompt |
| `max_tokens` | int | 128 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0=greedy) |

**Returns:** Generated text string.

### chat()

```python
def chat(
    self,
    messages: List[dict],
    max_tokens: int = 128,
    temperature: float = 0.7,
    enable_thinking: bool = False
) -> str
```

Chat format generation using tokenizer's chat template.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | List[dict] | required | Chat messages [{"role": "user", "content": "..."}] |
| `max_tokens` | int | 128 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `enable_thinking` | bool | False | Enable reasoning mode |

**Returns:** Response text string.

### _sample_token()

```python
def _sample_token(
    self,
    logits: torch.Tensor,
    temperature: float = 1.0
) -> int
```

Sample next token from logits.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logits` | torch.Tensor | required | Model output logits |
| `temperature` | float | 1.0 | Sampling temperature |

**Returns:** Sampled token ID.