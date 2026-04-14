from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_g(A_log, a, dt_bias):
    return torch.exp(-torch.exp(A_log.to(torch.float32)) * F.softplus(a + dt_bias))


def _gated_delta_step_ops(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(dim=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(dim=-1)  # [B, H, Dv]

    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        state = torch.where(mask, state, old_state)
    return y.to(q.dtype), state


def gated_delta_ops(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = torch.zeros((B, Hv, Dv, Dk), dtype=torch.float32, device=q.device)

    if (repeat_factor := Hv // Hk) > 1:
        q = q.repeat_interleave(repeat_factor, dim=-2)
        k = k.repeat_interleave(repeat_factor, dim=-2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    y = torch.stack(ys, dim=1)
    return y, state


def gated_delta_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    use_kernel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    beta = torch.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = torch.zeros((B, Hv, Dv, Dk), dtype=torch.float32, device=q.device)

    return gated_delta_ops(q, k, v, g, beta, state, mask)
