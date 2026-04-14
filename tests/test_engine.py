"""Test the InferenceEngine"""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from llmforge.inference import InferenceEngine

engine = InferenceEngine("Qwen/Qwen3-0.6B")
print(f"Model: {engine.model_name} ({engine.model_params:,} params)")
print(f"Device: {engine.device}")
print(f"EOS tokens: {engine.eos_token_ids}")
print()

print("=" * 50)
print("Test 1: Streaming + stop_words")
print("=" * 50)
print("User: Hello.")
print("Model: ", end="", flush=True)
t0 = time.time()
for text in engine.generate("Hello.", max_tokens=200, stop_words=["\n\n"]):
    print(text, end="", flush=True)
print(f"\n[Time: {time.time() - t0:.1f}s]\n")

print("=" * 50)
print("Test 2: Non-streaming (returns full string)")
print("=" * 50)
print("User: What is Python?")
t0 = time.time()
response = engine.generate("What is Python?", max_tokens=100, stream=False)
print(f"Model: {response}")
print(f"[Time: {time.time() - t0:.1f}s]\n")

print("=" * 50)
print("Test 3: Chat API with conversation")
print("=" * 50)
messages = [
    {"role": "system", "content": "You are a helpful assistant. Give short answers."},
    {"role": "user", "content": "What is 2+2?"},
]
print("User: What is 2+2?")
print("Model: ", end="", flush=True)
t0 = time.time()
for text in engine.chat(messages, max_tokens=50):
    print(text, end="", flush=True)
print(f"\n[Time: {time.time() - t0:.1f}s]")
