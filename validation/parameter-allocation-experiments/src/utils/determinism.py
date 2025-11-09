"""Utilities for ensuring reproducibility and determinism."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def enable_determinism(seed: int = 42) -> None:
    """Enable full determinism for reproducible training.

    This may impact performance but ensures reproducibility.

    Args:
        seed: Random seed value
    """
    set_seed(seed)

    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Use deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)


def disable_determinism() -> None:
    """Disable determinism for better performance."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
