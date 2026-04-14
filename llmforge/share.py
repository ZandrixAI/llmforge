# Copyright © 2026 Apple Inc.

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def error(*args, **kwargs):
    import sys

    kwargs["file"] = sys.stderr
    print("\033[31m[ERROR]", *args, "\033[0m", **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Distribute a model to other nodes (PyTorch version - not supported)."
    )
    parser.add_argument("--path", type=str, help="Path to a file or folder to share.")
    parser.add_argument(
        "--model", type=str, help="The path to a local model or Hugging Face repo"
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        help="The file containing the hosts and connection information",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="The destination path in other nodes (defaults to --path or --model)",
    )
    parser.add_argument(
        "--tmpdir",
        type=str,
        help="Intermediate temporary directory to ensure successful transfer",
    )

    args = parser.parse_args()

    print(
        "[ERROR] Distributed model sharing is not supported in the PyTorch version. "
        "This feature was MLX-specific. Use torch.distributed or other PyTorch "
        "distributed tools instead.",
        file=sys.stderr,
    )
    sys.exit(1)
