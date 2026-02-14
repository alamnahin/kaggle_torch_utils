"""Model complexity utilities: parameter counts and FLOPs."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def count_params_m(model: nn.Module) -> float:
    """
    Count total model parameters in millions.

    Args:
        model: A PyTorch module.

    Returns:
        Total parameter count / 1e6.
    """
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_params_m(model: nn.Module) -> float:
    """
    Count trainable model parameters in millions.

    Args:
        model: A PyTorch module.

    Returns:
        Trainable parameter count / 1e6.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def try_get_gflops(
    model: nn.Module,
    input_res: Tuple[int, ...],
    device: str = "cpu",
) -> Optional[float]:
    """
    Estimate GFLOPs via ptflops (optional dependency). Returns None if unavailable.

    Args:
        model: A PyTorch module.
        input_res: Input tensor shape *without* batch dim.
                   Examples: (3, 224, 224) for images, (512,) for 1-D,
                   (1, 16000) for audio waveforms.
        device: Device to run estimation on.

    Returns:
        GFLOPs as float, or None if ptflops is not installed or computation fails.
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return None

    if isinstance(model, nn.DataParallel):
        model = model.module

    model = model.to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            macs, _params = get_model_complexity_info(
                model,
                input_res,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
        flops = 2.0 * float(macs)  # FLOPs ≈ 2 * MACs
        return flops / 1e9
    except Exception as e:
        print(f"⚠️ GFLOPs computation failed: {e}")
        return None
    finally:
        if was_training:
            model.train()


def print_model_summary(
    model: nn.Module,
    input_res: Tuple[int, ...] = (3, 224, 224),
    device: str = "cpu",
) -> None:
    """
    Print a summary of model parameters and GFLOPs.

    Args:
        model: A PyTorch module.
        input_res: Input tensor shape without batch dim (e.g. (3, 224, 224)).
        device: Device for GFLOPs estimation.
    """
    total = count_params_m(model)
    trainable = count_trainable_params_m(model)
    gflops = try_get_gflops(model, input_res, device)

    print(f"   Total Params:     {total:.2f}M")
    print(f"   Trainable Params: {trainable:.2f}M")
    if gflops is not None:
        print(f"   GFLOPs:           {gflops:.2f}")
    else:
        print(f"   GFLOPs:           N/A (install ptflops)")
