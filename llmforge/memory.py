"""
LLMForge - Memory monitoring and management utilities for PyTorch.
"""

import torch


def is_available():
    """Check if any GPU (CUDA or MPS) is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def get_peak_memory():
    """Returns peak GPU memory usage in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0


def get_cache_memory():
    """Returns current cached GPU memory in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved()
    return 0


def get_total_memory():
    """Returns total GPU memory in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    return 0


def get_device_info():
    """Returns GPU device information dict (mirrors mx.device_info())."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "architecture": props.name,
            "total_memory": props.total_memory,
            "max_recommended_working_set_size": int(props.total_memory * 0.8),
            "device_index": 0,
        }
    elif torch.backends.mps.is_available():
        return {
            "name": "Apple Metal (MPS)",
            "architecture": "Apple Metal",
            "total_memory": 0,
            "max_recommended_working_set_size": 0,
            "device_index": 0,
        }
    return {
        "name": "cpu",
        "architecture": "cpu",
        "total_memory": 0,
        "max_recommended_working_set_size": 0,
        "device_index": 0,
    }


def clear_cache():
    """Clear GPU memory cache (replaces mx.clear_cache())."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def synchronize():
    """Synchronize GPU operations."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def reset_peak_memory_stats():
    """Reset peak memory counters."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def model_memory_bytes(model):
    """Calculate total memory used by model parameters."""
    total = 0
    for p in model.parameters():
        total += p.element_size() * p.nelement()
    return total


def seed(seed_value):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
