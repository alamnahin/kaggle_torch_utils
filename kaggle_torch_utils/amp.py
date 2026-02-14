"""Mixed precision (AMP) utilities with cross-version PyTorch compatibility."""

from contextlib import nullcontext
from typing import ContextManager

import torch


def get_autocast_ctx(use_amp: bool, device_type: str = "cuda") -> ContextManager:
    """
    Returns an autocast context manager compatible across PyTorch versions.

    Args:
        use_amp: Whether to enable automatic mixed precision.
        device_type: Device type for autocast (default: "cuda").

    Returns:
        A context manager for mixed precision or a no-op context.
    """
    enabled = bool(use_amp and torch.cuda.is_available())
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(use_amp: bool) -> torch.cuda.amp.GradScaler:
    """
    Create a GradScaler compatible across PyTorch versions.

    Args:
        use_amp: Whether AMP is enabled.

    Returns:
        A GradScaler instance (enabled or disabled based on use_amp).
    """
    enabled = bool(use_amp and torch.cuda.is_available())
    try:
        return torch.cuda.amp.GradScaler(enabled=enabled)
    except Exception:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(enabled=enabled)
        raise
