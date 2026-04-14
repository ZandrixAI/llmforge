"""LLMForge - Quick Inference Demo"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from llmforge.inference import InferenceEngine

engine = InferenceEngine("Qwen/Qwen3-0.6B")

# Use chat with thinking disabled
messages = [{"role": "user", "content": "Hello"}]
response = engine.chat(messages, max_tokens=32000, enable_thinking=False)
print(f"Response: {response}")