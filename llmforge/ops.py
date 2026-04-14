"""
LLMForge - Heavy Operations Module
Optimized sampling operations using NumPy (compiled C/Fortran backend).

These functions are faster than pure Python loops for large vocab sizes.
They handle the CPU-bound part while PyTorch handles GPU tensors.
"""

import numpy as np

# Try to import Rust ops (compiled with maturin on systems with a C compiler)
try:
    from llmforge_ops import (
        apply_top_k_py as _rust_top_k,
        apply_top_p_py as _rust_top_p,
        apply_repetition_penalty_py as _rust_rep_penalty,
        sample_with_top_k_top_p_py as _rust_sample,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Apply top-k filtering. Keeps only the top-k highest values."""
    if _HAS_RUST:
        return np.array(_rust_top_k(logits.tolist(), top_k), dtype=np.float32)

    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    # NumPy vectorized approach (compiled C under the hood)
    threshold = np.partition(logits, -top_k)[-top_k]
    logits = np.where(logits < threshold, -np.inf, logits)
    return logits


def apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    """Apply nucleus (top-p) filtering."""
    if _HAS_RUST:
        return np.array(_rust_top_p(logits.tolist(), top_p), dtype=np.float32)

    if top_p >= 1.0:
        return logits

    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    probs = softmax(sorted_logits)
    cumsum = np.cumsum(probs)

    # Find where cumulative probability exceeds top_p
    mask = cumsum - probs >= top_p
    indices_to_remove = sorted_indices[mask]
    logits[indices_to_remove] = -np.inf
    return logits


def apply_repetition_penalty(
    logits: np.ndarray, token_ids: list, penalty: float
) -> np.ndarray:
    """Apply repetition penalty to specific tokens."""
    if _HAS_RUST:
        return np.array(
            _rust_rep_penalty(logits.tolist(), list(token_ids), penalty),
            dtype=np.float32,
        )

    if penalty == 1.0 or not token_ids:
        return logits

    for token_id in set(token_ids):
        if token_id < len(logits):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a single token with top-k and top-p filtering."""
    if _HAS_RUST:
        return _rust_sample(logits.tolist(), temperature, top_k, top_p)

    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probs = softmax(logits)
    return int(np.random.choice(len(probs), p=probs))
