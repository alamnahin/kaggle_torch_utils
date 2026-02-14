"""Utilities for robust state dict loading (handles DataParallel prefix mismatches)."""

import warnings
from typing import Dict

import torch
import torch.nn as nn


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove 'module.' prefix from all keys (added by DataParallel/DDP).

    Args:
        state_dict: Model state dict.

    Returns:
        State dict with 'module.' prefixes stripped.
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def add_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Add 'module.' prefix to all keys (for loading into DataParallel models).

    Args:
        state_dict: Model state dict.

    Returns:
        State dict with 'module.' prefixes added.
    """
    out = {}
    for k, v in state_dict.items():
        out[k if k.startswith("module.") else f"module.{k}"] = v
    return out


def load_state_dict_robust(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Robustly load a state dict into a model, automatically handling DataParallel
    'module.' prefix mismatches.

    Tries in order:
    1. strict=True with original / stripped / added prefixes
    2. strict=False fallback with warnings about missing/unexpected keys

    Args:
        model: The target model.
        state_dict: The state dict to load.

    Raises:
        RuntimeError: If no loading strategy succeeds.
    """
    prefix_fns = [lambda x: x, strip_module_prefix, add_module_prefix]

    # Try strict loading first
    for fn in prefix_fns:
        try:
            model.load_state_dict(fn(state_dict), strict=True)
            return
        except Exception:
            pass

    # Fallback to non-strict
    for fn in prefix_fns:
        try:
            result = model.load_state_dict(fn(state_dict), strict=False)
            if getattr(result, "missing_keys", None) or getattr(
                result, "unexpected_keys", None
            ):
                warnings.warn(
                    "Loaded with strict=False:\n"
                    f"  Missing keys: {result.missing_keys}\n"
                    f"  Unexpected keys: {result.unexpected_keys}"
                )
            return
        except Exception:
            pass

    raise RuntimeError("Could not load state dict with any strategy.")
