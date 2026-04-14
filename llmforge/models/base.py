# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .utils import tree_map


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[torch.Tensor] = None,
    left_padding: Optional[torch.Tensor] = None,
):
    rinds = torch.arange(offset + N)
    linds = torch.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < (offset + N) - right_padding.view(1, 1, 1, -1))
    if left_padding is not None:
        mask = mask & (left_padding.view(1, 1, 1, -1) <= rinds)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, n_heads, L_q, D = queries.shape
    n_kv_heads = keys.shape[1]
    L_k = keys.shape[2]

    # Handle GQA: expand KV heads to match Q heads
    if n_kv_heads != n_heads and n_heads % n_kv_heads == 0:
        n_rep = n_heads // n_kv_heads
        keys = (
            keys.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, n_heads, L_k, D)
        )
        values = (
            values.unsqueeze(2)
            .expand(-1, -1, n_rep, -1, -1)
            .reshape(B, n_heads, L_k, D)
        )

    if sinks is not None:
        sink_k = sinks.unsqueeze(0).expand(B, -1, -1, -1)
        sink_v = sinks.unsqueeze(0).expand(B, -1, -1, -1)
        keys = torch.cat([sink_k, keys], dim=2)
        values = torch.cat([sink_v, values], dim=2)

    if isinstance(mask, str):
        out = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, is_causal=True, scale=scale
        )
        if sinks is not None:
            out = out[:, :, -L_q:, :]
        return out
    elif mask is not None:
        if mask.dtype == torch.bool:
            mask = torch.where(
                mask,
                torch.tensor(0.0, dtype=queries.dtype, device=queries.device),
                torch.tensor(float("-inf"), dtype=queries.dtype, device=queries.device),
            )
        if sinks is not None:
            n_sink = sinks.shape[2]
            sink_mask = torch.zeros(
                B, 1, 1, n_sink, dtype=mask.dtype, device=mask.device
            )
            mask = torch.cat([sink_mask, mask], dim=-1)
        out = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=mask, scale=scale
        )
        if sinks is not None:
            out = out[:, :, -L_q:, :]
        return out
    else:
        out = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, scale=scale
        )
        if sinks is not None:
            out = out[:, :, -L_q:, :]
        return out
