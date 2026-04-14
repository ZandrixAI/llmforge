"""Test device parameter"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from llmforge.inference import InferenceEngine
from llmforge.base_engine import available_devices

print("Available devices:")
for dev, info in available_devices().items():
    mem = f" ({info['memory']})" if "memory" in info else ""
    print(f"  {dev}: {info['name']}{mem}")
print()

print("=" * 50)
print("Test 1: device=None (auto-detect)")
print("=" * 50)
engine = InferenceEngine("Qwen/Qwen3-0.6B")
print()

print("=" * 50)
print("Test 2: device='cuda'")
print("=" * 50)
engine_cuda = InferenceEngine("Qwen/Qwen3-0.6B", device="cuda")
print()

print("=" * 50)
print("Test 3: device='cpu'")
print("=" * 50)
engine_cpu = InferenceEngine("Qwen/Qwen3-0.6B", device="cpu")
print()

print("=" * 50)
print("Test 4: device='tpu' (should error)")
print("=" * 50)
try:
    engine_tpu = InferenceEngine("Qwen/Qwen3-0.6B", device="tpu")
except RuntimeError as e:
    print(f"Expected error: {e}")
print()

print("=" * 50)
print("Generation on auto-detected device")
print("=" * 50)
print(f"User: What is Python?")
print(f"Model ({engine.device}): ", end="", flush=True)
for text in engine.generate("What is Python?", max_tokens=50):
    print(text, end="", flush=True)
print()
