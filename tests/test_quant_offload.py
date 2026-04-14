"""Test quantization and offloading"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
import torch.nn as nn

print("=" * 50)
print("Test 1: 4-bit quantization")
print("=" * 50)
from llmforge.quant_module import quantize_model


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


model = TestModel()
original_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
print(f"Original size: {original_bytes:,} bytes")

model_q = quantize_model(model, bits=4)
quant_bytes = sum(p.nelement() * p.element_size() for p in model_q.parameters())
quant_bytes += sum(b.nelement() * b.element_size() for b in model_q.buffers())
print(f"Quantized size: {quant_bytes:,} bytes")
print(f"Compression: {original_bytes / quant_bytes:.1f}x")

x = torch.randn(1, 1024)
out = model_q(x)
print(f"Forward pass OK: {out.shape}")
print()

print("=" * 50)
print("Test 2: System memory")
print("=" * 50)
from llmforge.offload import get_system_memory

mem = get_system_memory()
for k, v in mem.items():
    print(f"  {k}: {v / 1e9:.1f} GB")
print()

print("=" * 50)
print("Test 3: OffloadedLayer")
print("=" * 50)
from llmforge.offload import OffloadedLayer

layer = nn.Linear(512, 512)
offloaded = OffloadedLayer(layer, offload_device="cpu")
print(f"Offloaded: {offloaded}")

x = torch.randn(1, 512)
if torch.cuda.is_available():
    x = x.cuda()
    out = offloaded(x)
    print(f"Forward OK on GPU: {out.shape}, device={out.device}")
else:
    out = offloaded(x)
    print(f"Forward OK on CPU: {out.shape}")
print()

print("=" * 50)
print("Test 4: InferenceEngine with quantization")
print("=" * 50)
from llmforge.inference import InferenceEngine

engine = InferenceEngine("Qwen/Qwen3-0.6B", bits=8)
print()
print("User: What is AI?")
print("Model: ", end="", flush=True)
for text in engine.generate("What is AI?", max_tokens=40):
    print(text, end="", flush=True)
print()
