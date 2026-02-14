"""Safe serialization utilities for JSON/logging with numpy & torch types."""

from typing import Any, Optional

import numpy as np
import torch


def json_safe_scalar(x: Any) -> Optional[Any]:
    """
    Convert a value to a JSON-safe scalar type.

    Handles: int, float, str, bool, np.integer, np.floating,
             0-d torch.Tensor, 0-d np.ndarray.

    Args:
        x: Value to convert.

    Returns:
        A JSON-serializable value, or None for NaN/inf/unsupported types.
    """
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        val = float(x)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return json_safe_scalar(x.item())
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return json_safe_scalar(x.item())
        return x.tolist()
    return str(x)


def safe_json_dict(d: dict) -> dict:
    """
    Recursively convert all values in a dict to JSON-safe types.

    Args:
        d: Input dictionary.

    Returns:
        A new dict with all values converted via json_safe_scalar.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = safe_json_dict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [json_safe_scalar(item) for item in v]
        else:
            out[k] = json_safe_scalar(v)
    return out
