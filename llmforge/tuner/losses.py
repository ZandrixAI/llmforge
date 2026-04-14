# Copyright © 2025 Apple Inc.

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_div_loss(logits_q, logits_p):
    logprobs_q = logits_q - torch.logsumexp(logits_q, dim=-1, keepdim=True)
    logprobs_p = logits_p - torch.logsumexp(logits_p, dim=-1, keepdim=True)
    return F.kl_div(
        logprobs_q,
        logprobs_p.exp(),
        reduction="none",
    ).sum(dim=-1)


def js_div_loss(logits_q, logits_p):
    logprobs_q = logits_q - torch.logsumexp(logits_q, dim=-1, keepdim=True)
    logprobs_p = logits_p - torch.logsumexp(logits_p, dim=-1, keepdim=True)
    logprobs_m = (
        logprobs_p
        + torch.log(1 + torch.exp(logprobs_q - logprobs_p))
        - torch.log(torch.tensor(2.0, dtype=logits_q.dtype, device=logits_q.device))
    )
    kl_p = F.kl_div(logprobs_m, logprobs_p.exp(), reduction="none").sum(dim=-1)
    kl_q = F.kl_div(logprobs_m, logprobs_q.exp(), reduction="none").sum(dim=-1)
    return 0.5 * (kl_p + kl_q)
