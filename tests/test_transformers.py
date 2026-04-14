"""Test with transformers directly"""

import os, sys
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Chat format that transformers uses
messages = [{"role": "user", "content": "Hello"}]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(f"Chat template output:\n{repr(text)}\n")

# Generate
inputs = tokenizer(text, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=5000, do_sample=False)
result = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Transformers output:\n{result}")