"""Reproducibility utilities for PyTorch training."""

import random
import warnings

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds across all libraries for reproducibility.

    Args:
        seed: The seed value.
        deterministic: If True, enables full determinism (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            warnings.warn(
                f"Could not enable deterministic algorithms: {e}\n"
                "Training will continue with partial determinism."
            )
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker init function for reproducible data loading.

    Usage:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
