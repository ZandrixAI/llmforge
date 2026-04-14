"""
LLMForge - Tensor parallel sharding utilities for PyTorch.
Provides sharded linear layers for distributed inference and training.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn


class ColumnParallelLinear(nn.Linear):
    """
    Split output dim across ranks (each rank gets a slice of columns).
    Mirrors MLX's shard_linear with mode="all-to-sharded".

    Each rank computes its local portion of the output. The caller must
    all-gather to get the full output when needed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group=None,
    ):
        self.group = group
        self.full_out_features = out_features

        if group is not None and group.size() > 1:
            rank = group.rank()
            world_size = group.size()
            # Ensure out_features is divisible
            local_out = out_features // world_size
            assert out_features % world_size == 0, (
                f"out_features ({out_features}) must be divisible by "
                f"world_size ({world_size})"
            )
        else:
            local_out = out_features

        super().__init__(in_features, local_out, bias=bias)

        if group is not None and group.size() > 1:
            rank = group.rank()
            with torch.no_grad():
                # Slice the weight to get this rank's portion
                full_weight = torch.empty(out_features, in_features)
                nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
                self.weight = nn.Parameter(
                    full_weight[rank * local_out : (rank + 1) * local_out]
                )
                if bias:
                    fan_in = in_features
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    full_bias = torch.empty(out_features).uniform_(-bound, bound)
                    self.bias = nn.Parameter(
                        full_bias[rank * local_out : (rank + 1) * local_out]
                    )


class RowParallelLinear(nn.Linear):
    """
    Split input dim across ranks (each rank has a slice of input dim).
    Mirrors MLX's shard_linear with mode="sharded-to-all".

    Each rank computes partial results; a distributed all-reduce sums them.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group=None,
    ):
        self.group = group
        self.full_in_features = in_features

        if group is not None and group.size() > 1:
            rank = group.rank()
            world_size = group.size()
            assert in_features % world_size == 0, (
                f"in_features ({in_features}) must be divisible by "
                f"world_size ({world_size})"
            )
            local_in = in_features // world_size
        else:
            local_in = in_features

        # Only rank 0 gets bias in sharded case (bias added after all-reduce)
        use_bias = bias and (group is None or group.rank() == 0 or group.size() == 1)
        super().__init__(local_in, out_features, bias=use_bias)

        if group is not None and group.size() > 1:
            rank = group.rank()
            with torch.no_grad():
                full_weight = torch.empty(out_features, in_features)
                nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
                self.weight = nn.Parameter(
                    full_weight[:, rank * local_in : (rank + 1) * local_in]
                )

    def forward(self, x):
        out = super().forward(x)
        if self.group is not None and self.group.size() > 1:
            dist.all_reduce(out)
        return out


class ShardedEmbedding(nn.Embedding):
    """
    Split embedding table across ranks (vocabulary parallel).
    Each rank holds a slice of the vocabulary.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        group=None,
        padding_idx=None,
    ):
        self.group = group
        self.full_num_embeddings = num_embeddings

        if group is not None and group.size() > 1:
            rank = group.rank()
            world_size = group.size()
            assert num_embeddings % world_size == 0, (
                f"num_embeddings ({num_embeddings}) must be divisible by "
                f"world_size ({world_size})"
            )
            local_num = num_embeddings // world_size
        else:
            local_num = num_embeddings

        super().__init__(local_num, embedding_dim, padding_idx=padding_idx)

        if group is not None and group.size() > 1:
            rank = group.rank()
            self._vocab_start = rank * local_num
            self._vocab_end = (rank + 1) * local_num

    def forward(self, x):
        if self.group is None or self.group.size() == 1:
            return super().forward(x)

        # Mask tokens that belong to other ranks
        local_mask = (x >= self._vocab_start) & (x < self._vocab_end)
        local_indices = x - self._vocab_start
        local_indices = torch.where(
            local_mask, local_indices, torch.zeros_like(local_indices)
        )

        # Local embedding lookup
        embedded = super().forward(local_indices)
        # Zero out embeddings for tokens not belonging to this rank
        embedded = embedded * local_mask.unsqueeze(-1).float()

        # All-reduce to sum partial embeddings
        dist.all_reduce(embedded)
        return embedded


def shard_linear(linear, mode: str, group=None):
    """
    Shard a linear layer for distributed inference.

    Args:
        linear: nn.Linear layer to shard
        mode: "all-to-sharded" (column parallel) or "sharded-to-all" (row parallel)
        group: distributed Group object

    Returns:
        Sharded linear layer
    """
    if group is None or group.size() == 1:
        return linear

    if mode == "all-to-sharded":
        sharded = ColumnParallelLinear(
            linear.in_features,
            linear.full_out_features
            if hasattr(linear, "full_out_features")
            else linear.out_features * group.size(),
            bias=linear.bias is not None,
            group=group,
        )
    elif mode == "sharded-to-all":
        sharded = RowParallelLinear(
            linear.full_in_features
            if hasattr(linear, "full_in_features")
            else linear.in_features * group.size(),
            linear.out_features,
            bias=linear.bias is not None,
            group=group,
        )
    else:
        raise ValueError(f"Unknown shard mode: {mode}")

    return sharded


def shard_embedding(embedding, group=None):
    """Shard an embedding layer for distributed inference."""
    if group is None or group.size() == 1:
        return embedding

    return ShardedEmbedding(
        embedding.num_embeddings,
        embedding.embedding_dim,
        group=group,
        padding_idx=embedding.padding_idx,
    )
