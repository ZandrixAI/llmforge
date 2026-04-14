"""
LLMForge - Distributed communication utilities for PyTorch.
Provides an API-compatible wrapper around torch.distributed.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class Group:
    """Distributed communication group (mirrors mx.distributed.Group)."""

    rank_id: int = 0
    world_size: int = 1
    _initialized: bool = True

    def rank(self) -> int:
        """Return the rank of this process."""
        return self.rank_id

    def size(self) -> int:
        """Return the total number of processes."""
        return self.world_size

    def is_distributed(self) -> bool:
        """Return True if this group has more than one process."""
        return self.world_size > 1

    def all_sum(self, x, stream=None):
        """Sum arrays across all processes."""
        if self.world_size == 1:
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(x)

        if isinstance(x, (int, float)):
            x = torch.tensor(x)

        if torch.cuda.is_available():
            x = x.cuda()

        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        return x

    def all_max(self, x, stream=None):
        """Max across all processes."""
        if self.world_size == 1:
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(x)

        if isinstance(x, (int, float)):
            x = torch.tensor(x)

        if torch.cuda.is_available():
            x = x.cuda()

        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
        return x

    def all_gather(self, x, stream=None):
        """Gather arrays from all processes."""
        if self.world_size == 1:
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(x)

        if isinstance(x, (int, float)):
            x = torch.tensor(x)

        if torch.cuda.is_available():
            x = x.cuda()

        output = [torch.zeros_like(x) for _ in range(self.world_size)]
        torch.distributed.all_gather(output, x)
        return torch.stack(output)

    def send(self, x, dst: int):
        """Send array to destination rank."""
        if self.world_size == 1:
            return
        torch.distributed.send(x, dst=dst)

    def recv(self, src: int, shape=None, dtype=None):
        """Receive array from source rank."""
        if self.world_size == 1:
            if shape is not None:
                return torch.zeros(shape, dtype=dtype or torch.float32)
            return None
        if shape is not None:
            x = torch.zeros(shape, dtype=dtype or torch.float32)
            if torch.cuda.is_available():
                x = x.cuda()
            torch.distributed.recv(x, src=src)
            return x
        return None


# Singleton group for single-process usage
_singleton = Group()


def init(backend: Optional[str] = None) -> Group:
    """
    Initialize distributed communication (mirrors mx.distributed.init()).

    Returns a Group object. In single-process mode, returns a trivial group
    with rank=0 and size=1.
    """
    global _singleton

    # Check environment variables for distributed setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        if not torch.distributed.is_initialized():
            if backend is None:
                backend = "nccl" if torch.cuda.is_available() else "gloo"

            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

        _singleton = Group(rank_id=rank, world_size=world_size)
    else:
        _singleton = Group(rank_id=0, world_size=1)

    return _singleton


def is_distributed() -> bool:
    """Return True if distributed training is active."""
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def get_rank() -> int:
    """Return current process rank."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Return total number of processes."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1
