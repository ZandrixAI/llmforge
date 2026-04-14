# Quantization Guide

Reduce model size and memory usage with quantization.

## Quantization Levels

| Bits | Size Reduction | Quality Loss | Use Case |
|------|---------------|--------------|----------|
| 16 | 0% | None | Original precision |
| 8 | 50% | Minimal | Desktop inference |
| 4 | 75% | Low | Mobile/embedded |
| 2 | 87.5% | High | Extreme compression |

## Basic Usage

```python
from llmforge import InferenceEngine

# 4-bit quantization (default for large models)
engine = InferenceEngine(
    "meta-llama/Llama-3.2-1B-Instruct",
    bits=4
)

# 8-bit quantization
engine = InferenceEngine(
    "meta-llama/Llama-3.2-1B-Instruct",
    bits=8
)
```

## Manual Quantization

```python
from llmforge.quant_module import quantize_model

# Quantize an existing model
model = quantize_model(model, bits=4)
```

## Quantization Methods

### INT8 (Dynamic)

```python
from llmforge.quant.dynamic_quant import apply_dynamic_quant

model = apply_dynamic_quant(model)
```

### INT4 (GPTQ)

```python
from llmforge.quant.gptq import apply_gptq

model = apply_gptq(model, tokenizer, dataset)
```

### INT4 (AWQ)

```python
from llmforge.quant.awq import apply_awq

model = apply_awq(model, tokenizer, dataset)
```

## Model Sizing Example

| Model | Params | FP16 | INT8 | INT4 |
|-------|--------|------|-----|------|
| Llama-3.2-1B | 1B | 2GB | 1GB | 0.5GB |
| Llama-3.2-3B | 3B | 6GB | 3GB | 1.5GB |
| Llama-3.2-8B | 8B | 16GB | 8GB | 4GB |
| Llama-3.1-70B | 70B | 140GB | 70GB | 35GB |

## Trade-offs

- **INT8**: Great for batch processing, minimal quality loss
- **INT4**: Best for deployment, slight quality loss
- **AWQ**: Better than GPTQ for instruction tuning
- **GPTQ**: Better for base models