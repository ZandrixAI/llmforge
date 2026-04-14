# Fine-tuning Guide

Fine-tune models using LoRA or full parameter training.

## LoRA (Recommended)

### Training Loop

```python
from llmforge.tuner import LoRA Trainer

trainer = LoRATrainer(
    model=model,
    tokenizer=tokenizer,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

trainer.train(train_dataset, epochs=3)
trainer.save_adapter("adapter/")
```

### Training Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling |
| `lora_dropout` | 0.05 | Dropout probability |
| `target_modules` | ["q_proj", "v_proj"] | Modules to apply LoRA |
| `epochs` | 3 | Training epochs |
| `batch_size` | 1 | Batch size |
| `learning_rate` | 3e-4 | Learning rate |

### Merge Adapter

```python
from llmforge.tuner import merge_adapter

model_with_adapter = merge_adapter(model, "adapter/")
```

## Full Fine-tuning

```python
from llmforge.tuner import load_for_training

model = load_for_training(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

for epoch in range(epochs):
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## DORA (DoRA)

Better than standard LoRA for multimodal models:

```python
from llmforge.tuner import DoRATrainer

trainer = DoRATrainer(
    model=model,
    tokenizer=tokenizer,
    dora_r=16,
)
```

## Datasets

### CSV Dataset

```python
from llmforge.tuner import CSVDataset

dataset = CSVDataset(
    "data.csv",
    text_column="prompt",
    label_column="completion",
)
```

### JSONL Dataset

```python
from llmforge.tuner import JSONLDataset

dataset = JSONLDataset("data.jsonl")
```

### HuggingFace Dataset

```python
from llmforge.tuner import HFDataset
from datasets import load_dataset

hf_ds = load_dataset("openai/webgpt_completions", split="train")
dataset = HFDataset(hf_ds, "question", "best_answer")
```