# Utils API

Utility functions for model loading and generation.

## load()

```python
from llmforge.utils import load

model, tokenizer = load(
    model_name: str,
    adapter_path: Optional[str] = None,
    tokenizer_config: Optional[dict] = None,
)
```

Load a model and tokenizer.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | required | HuggingFace model name or local path |
| `adapter_path` | str | None | LoRA adapter path |
| `tokenizer_config` | dict | None | Tokenizer configuration |

**Returns:** Tuple of (model, tokenizer)

## stream_generate()

```python
from llmforge.generate import stream_generate

for text in stream_generate(
    model,
    tokenizer,
    prompt,
    max_tokens: int = 256,
    sampler = None,
    prompt_cache = None,
):
    print(text.text, end="")
```

Stream generated text token by token.

## make_sampler()

```python
from llmforge.sample_utils import make_sampler

sampler = make_sampler(
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
)
```

Create a sampler for generation.

## make_prompt_cache()

```python
from llmforge.models.cache import make_prompt_cache

prompt_cache = make_prompt_cache(
    model,
    max_kv_size: Optional[int] = None,
)
```

Create a prompt cache for efficient generation.

## quantize_model()

```python
from llmforge.quant_module import quantize_model

quantized_model = quantize_model(
    model,
    bits: int = 4,
)
```

Quantize a model to reduce memory usage.

## auto_tensor_parallel()

```python
from llmforge.tensor_parallel import auto_tensor_parallel

sharded_model = auto_tensor_parallel(model)
```

Automatically shard model across devices.