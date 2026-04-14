# LLMForge - Setup Configuration

import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "llmforge"
sys.path.append(str(package_dir))

from _version import __version__

setup(
    name="llmforge",
    version=__version__,
    description="LLM Inference and RL Self-Improvement Engine - Built on PyTorch",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author="LLMForge Contributors",
    url="https://github.com/llmforge/llmforge",
    license="MIT",
    install_requires=[
        "torch>=2.0.0",
        "safetensors",
        "numpy",
        "transformers>=5.0.0",
        "sentencepiece",
        "protobuf",
        "pyyaml",
        "jinja2",
        "psutil>=5.9.0",
        "nltk>=3.8.0",
    ],
    packages=[
        "llmforge",
        "llmforge.models",
        "llmforge.quant",
        "llmforge.tuner",
        "llmforge.tool_parsers",
        "llmforge.chat_templates",
    ],
    python_requires=">=3.8",
    extras_require={
        "test": ["datasets", "lm-eval", "pytest"],
        "train": ["datasets", "tqdm", "accelerate"],
        "evaluate": ["lm-eval", "tqdm"],
    },
    entry_points={
        "console_scripts": [
            "llmforge = llmforge.cli:main",
            "llmforge.awq = llmforge.quant.awq:main",
            "llmforge.dwq = llmforge.quant.dwq:main",
            "llmforge.dynamic_quant = llmforge.quant.dynamic_quant:main",
            "llmforge.gptq = llmforge.quant.gptq:main",
            "llmforge.benchmark = llmforge.benchmark:main",
            "llmforge.cache_prompt = llmforge.cache_prompt:main",
            "llmforge.chat = llmforge.chat:main",
            "llmforge.convert = llmforge.convert:main",
            "llmforge.evaluate = llmforge.evaluate:main",
            "llmforge.fuse = llmforge.fuse:main",
            "llmforge.generate = llmforge.generate:main",
            "llmforge.lora = llmforge.lora:main",
            "llmforge.perplexity = llmforge.perplexity:main",
            "llmforge.server = llmforge.server:main",
            "llmforge.share = llmforge.share:main",
            "llmforge.manage = llmforge.manage:main",
            "llmforge.upload = llmforge.upload:main",
        ]
    },
)
