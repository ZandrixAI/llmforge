from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dt(dt, dt_bias, time_step_limit):
    dt = dt.to(torch.float32)
    dt = F.softplus(dt + dt_bias)
    return torch.clamp(dt, time_step_limit[0], time_step_limit[1])


def segsum(x, mask=None):
    l = x.shape[-1]
    if mask is not None:
        mask = mask.unsqueeze(1)
        x = x * mask
    x = x[..., None].repeat(1, 1, 1, 1, l)
    x = torch.tril(x, -1)
    x_segsum = torch.cumsum(x, dim=-2)
    if mask is not None:
        x_segsum = torch.where(
            mask[..., None, :] * mask[..., None], x_segsum, float("-inf")
        )
    return x_segsum


def ssm_attn(
    x: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
    mask: Optional[torch.Tensor] = None,
    lengths: Optional[torch.Tensor] = None,
    step: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, l, h, dh = x.shape
    _, _, g, d = B.shape

    dt = compute_dt(dt, dt_bias, time_step_limit)
    repeats = h // g
    A = -torch.exp(A_log).to(dt.dtype)
    dtA = dt * A.reshape(1, 1, -1)
    dtx = dt.reshape(b, l, h, 1) * x

    def _step(dtx, dtA, B, C, state, mask):
        s = dtx.shape[1]
        B = B.permute(0, 2, 3, 1)

        CB = C.transpose(1, 2) @ B
        CB = CB.repeat(1, repeats, 1, 1)

        decay = torch.exp(segsum(dtA.transpose(1, 2), mask=mask))

        surrogate_attention_matrix = torch.tril(CB * decay, 0)

        y = surrogate_attention_matrix @ dtx.transpose(1, 2)
        y = y.transpose(1, 2)

        if lengths is not None:
            pos = torch.maximum(
                torch.minimum(lengths, torch.tensor(step)) - 1, torch.tensor(0)
            )
            pos = pos.reshape(-1, 1, 1, 1)
            decay = torch.gather(decay, 2, pos.expand(-1, -1, -1, decay.shape[-1]))
        else:
            decay = decay[:, :, -1:, :]

        decay = decay.permute(0, 3, 1, 2)
        B = B.repeat(1, h // g, 1, 1).transpose(2, 3)
        dtxdecay = dtx * decay
        dtxdecay = dtxdecay.transpose(1, 2).transpose(2, 3)

        next_state = dtxdecay @ B

        if state is not None:
            exp_dtA_cumsum = torch.exp(torch.cumsum(dtA, dim=-2))
            next_state = next_state + exp_dtA_cumsum[:, -1, :, None, None] * state
            C = C.reshape(b, s, g, 1, d, 1)
            y_prev = (
                (state.reshape((b, 1, g, repeats, dh, d)) @ C).squeeze(-1).flatten(2, 3)
            )
            y = y + exp_dtA_cumsum[..., None] * y_prev
        if lengths is not None and state is not None:
            next_state = torch.where(
                (lengths < 0).reshape(-1, 1, 1, 1), state, next_state
            )

        return y.to(x.dtype), next_state

    ys = []
    for i in range(0, l, step):
        y, state = _step(
            dtx[:, i : i + step],
            dtA[:, i : i + step],
            B[:, i : i + step],
            C[:, i : i + step],
            state,
            None if mask is None else mask[..., i : i + step],
        )
        if lengths is not None:
            lengths = lengths - step
        ys.append(y)
    y = torch.cat(ys, dim=1) + x * D.reshape(1, 1, h, 1)
    return y, state


def ssm_update(
    hidden_states: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
    mask: Optional[torch.Tensor] = None,
    lengths: Optional[torch.Tensor] = None,
):
    return ssm_attn(
        hidden_states,
        A_log,
        B,
        C,
        D,
        dt,
        dt_bias,
        state,
        time_step_limit,
        mask=mask,
        lengths=lengths,
    )
