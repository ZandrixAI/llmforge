# Copyright © 2025 Apple Inc.

"""
Run with:

```
python sharded_generate.py --prompt 'Hello world'
```

This example demonstrates loading a model and generating text with PyTorch.
"""

import argparse

from llmforge import stream_generate
from llmforge.utils import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM inference example")
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.3-70B-Instruct-4bit",
        help="HF repo or path to local model.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Write a quicksort in C++.",
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    model, tokenizer = load(args.model)

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens
    ):
        print(response.text, end="", flush=True)

    print()
    print("=" * 10)
    print(
        f"Prompt: {response.prompt_tokens} tokens, "
        f"{response.prompt_tps:.3f} tokens-per-sec"
    )
    print(
        f"Generation: {response.generation_tokens} tokens, "
        f"{response.generation_tps:.3f} tokens-per-sec"
    )
    print(f"Peak memory: {response.peak_memory:.3f} GB")
