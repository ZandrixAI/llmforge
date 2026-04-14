# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable, Dict, List, Optional

import torch


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Make a sampler function for use with ``generate_step``.

    Args:
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        top_k (int, optional): The top k tokens ranked by probability to constrain
          the sampling to.
        xtc_probability (float, optional): The probability of applying XTC
            sampling.
        xtc_threshold (float, optional): The threshold the probs need to reach
            for being sampled.
        xtc_special_tokens (list(int), optional): List of special tokens IDs to
            be excluded from XTC sampling.


    Returns:
        Callable[torch.Tensor, torch.Tensor]:
            A sampler which takes log-probabilities and returns tokens.
    """
    if temp == 0:
        return lambda x: torch.argmax(x, dim=-1)

    # Create sampler chain
    sampling_methods = []
    if top_p > 0 and top_p < 1.0:
        sampling_methods.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        sampling_methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if xtc_probability > 0.0:
        sampling_methods.append(
            lambda x: apply_xtc(x, xtc_probability, xtc_threshold, xtc_special_tokens)
        )
    if top_k > 0:
        sampling_methods.append(lambda x: apply_top_k(x, top_k))

    # Apply the sampling methods
    def sampler(logprobs):
        for method in sampling_methods:
            logprobs = method(logprobs)
        # Return the sampled token
        return categorical_sampling(logprobs, temp)

    return sampler


def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    presence_penalty: Optional[float] = None,
    presence_context_size: Optional[int] = 20,
    frequency_penalty: Optional[float] = None,
    frequency_context_size: Optional[int] = 20,
):
    """
    Make logits processors for use with ``generate_step``.

    Args:
        repetition_penalty (float, optional): A (sign-aware) multiplicative
          penalty for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        presence_penalty (float, optional): An additive penalty to reduce
          repeating tokens.
        presence_context_size (int, optional): The number of tokens to consider
          for the presence penalty. Default: ``20``.
        frequency_penalty (float, optional): An additive penalty to reduce
          repeating tokens. The tokens are penalized proportionally to their
          frequency.
        frequency_context_size (int, optional): The number of tokens to consider
          for the frequency penalty. Default: ``20``.
        logit_bias (dictionary, optional): Additive logit bias.

    Returns:
        List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
            A list of logits processors. Each processor in the list is a
            callable which takes an array of tokens and an array of logits
            and returns the updated logits.
    """
    logits_processors = []
    if logit_bias:
        indices = list(logit_bias.keys())
        values = torch.tensor(list(logit_bias.values()))

        def logit_bias_processor(_, logits):
            logits[:, indices] += values
            return logits

        logits_processors.append(logit_bias_processor)

    repetition_penalties = [
        (make_repetition_penalty, repetition_penalty, repetition_context_size),
        (make_presence_penalty, presence_penalty, presence_context_size),
        (make_frequency_penalty, frequency_penalty, frequency_context_size),
    ]

    for make_penalty, penalty, context_size in repetition_penalties:
        if penalty is not None and penalty != 0:
            logits_processors.append(make_penalty(penalty, context_size))

    return logits_processors


def apply_top_k(
    logprobs: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Sample from only the top K tokens ranked by probability.

    Args:
        logprobs: A vector of log probabilities.
        top_k (int): Top k tokens to sample from.
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    top_k_values, top_k_indices = torch.topk(logprobs, top_k, dim=-1)
    mask = torch.ones_like(logprobs, dtype=torch.bool)
    mask.scatter_(-1, top_k_indices, False)
    return logprobs.masked_fill(mask, float("-inf"))


def apply_min_p(
    logprobs: torch.Tensor,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Apply min-p sampling to the logprobs.

    Min-p keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter is more
    aggressive given a very high-probability token.

    Args:
        logprobs: A vector of log probabilities.
        min_p (float): Minimum token probability. Typical values are in the
            0.01-0.2 range, comparably selective as setting `top_p` in the
            0.99-0.8 range.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
            be filtered. Default: ``1``.

    """
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )

    sorted_indices = torch.argsort(logprobs, dim=-1, descending=True)
    sorted_logprobs = torch.gather(logprobs, -1, sorted_indices)

    top_logprobs = sorted_logprobs[:, 0:1]
    scaled_min_p = top_logprobs + math.log(min_p)

    tokens_to_remove = sorted_logprobs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False

    selected_logprobs = torch.where(tokens_to_remove, float("-inf"), sorted_logprobs)

    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    return torch.gather(selected_logprobs, -1, inverse_indices)


def apply_top_p(logprobs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logprobs: A vector of log probabilities.
        top_p: The cumulative probability threshold for top-p filtering.
    Returns:
        token selected based on the top-p criterion.
    """
    probs = torch.exp(logprobs)
    # sort in ascending order
    sorted_indices = torch.argsort(logprobs, dim=-1)
    sorted_probs = torch.gather(probs, -1, sorted_indices)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Rearrange cumulative probs back to original order
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    cumulative_probs = torch.gather(cumulative_probs, -1, inverse_indices)

    # select tokens with cumulative probs below threshold
    return torch.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        float("-inf"),
    )


def apply_xtc(
    logits: torch.Tensor,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: List[int],
) -> torch.Tensor:
    """
    Apply XTC sampling to the logits.

    Args:
        logits: The logits from the model's output.
        xtc_probability (float): Probability of XTC sampling to happen for each token
        xtc_threshold (float): The threshold the probs need to reach for being sampled.
        special_tokens_ids (list(int)): List of special tokens IDs to be excluded from XTC sampling.
    """
    if not (0 <= xtc_threshold <= 0.5):
        raise ValueError(
            f"`threshold` has to be a float in the [0, 0.5] interval, but is {xtc_threshold}"
        )
    if not (0 <= xtc_probability <= 1.0):
        raise ValueError(
            f"`probability` has to be a float in the [0, 1] interval, but is {xtc_probability}"
        )

    probs = torch.softmax(logits, dim=-1)
    mask = (
        probs
        > torch.where(
            probs > xtc_threshold,
            probs,
            torch.tensor(float("inf"), dtype=probs.dtype, device=probs.device),
        ).min()
    )
    if xtc_special_tokens:
        mask[..., xtc_special_tokens] = False

    return torch.where(
        torch.rand(1, device=logits.device) > xtc_probability,
        logits,
        torch.where(mask, float("-inf"), logits),
    )


def categorical_sampling(logits, temp):
    return torch.multinomial(torch.softmax(logits * (1 / temp), dim=-1), 1).squeeze(-1)


def make_repetition_penalty(penalty: float, context_size: int = 20):
    """
    Make repetition penalty processor.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        penalty (float): The repetition penalty factor to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[torch.Tensor, List[int]], torch.Tensor]:
            The repetition penalty processor.
    """
    if penalty < 0 or not isinstance(penalty, (int, float)):
        raise ValueError(f"penalty must be a non-negative float, got {penalty}")

    def repetition_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = torch.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[:, tokens] = selected_logits
        return logits

    return repetition_penalty_processor


def make_presence_penalty(penalty: float, context_size: int = 20):
    """
    Make a presence penalty processor.

    Corresponds to the OpenAI option with the same name. Namely, subtracts
    ``penalty`` from a logit if the token has occured at least once in the
    ``context_size`` previous tokens.

    Args:
        penalty (float): The presence penalty to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[torch.Tensor, List[int]], torch.Tensor]
    """

    def presence_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            logits[:, tokens] -= penalty
        return logits

    return presence_penalty_processor


def make_frequency_penalty(penalty: float, context_size: int = 20):
    """
    Make a frequency penalty processor.

    Corresponds to the OpenAI option with the same name. Namely, subtracts
    ``penalty`` from a logit for every time that the token has occured in the
    ``context_size`` previous tokens.

    The difference with the presence penalty is that the more often a token
    occurs the more it will be penalized.

    Args:
        penalty (float): The frequency penalty to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[torch.Tensor, List[int]], torch.Tensor]
    """

    def frequency_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            logits[:, tokens] -= penalty
        return logits

    return frequency_penalty_processor
