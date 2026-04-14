"""Test RL Engine with SQLite Memory"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from llmforge.rl import RLEngine

print("Loading RL engine with SQLite memory...")
rl = RLEngine("Qwen/Qwen3-0.6B")

# Test generation
tests = [
    "What is Python?",
    "Explain gravity",
]

for prompt in tests:
    print(f"\n=== {prompt} ===")
    response = rl.generate(prompt, max_tokens=50)
    print(f"Response: {response[:80]}...")

# Check stats
stats = rl.get_stats()
print(f"\n=== Stats: {stats} ===")

# Search memory
print("\nSearching memory for 'python':")
results = rl.search_memory("python", limit=2)
for r in results:
    print(f"  - {r['response'][:50]}... (score: {r['quality_score']})")

print("\nDone!")
rl.close()