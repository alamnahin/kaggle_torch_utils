"""Learning rate scheduler factories for common training recipes."""

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 5,
    warmup_lr_init: float = 1e-6,
    base_lr: float = 5e-4,
    min_lr: float = 1e-6,
    gradient_accumulation_steps: int = 1,
) -> _LRScheduler:
    """
    Build a LinearLR warmup → CosineAnnealingLR scheduler (per optimizer step).

    Correctly accounts for gradient accumulation when computing the schedule.

    Args:
        optimizer: The optimizer.
        epochs: Total training epochs.
        steps_per_epoch: Number of batches per epoch (len(dataloader)).
        warmup_epochs: Number of warmup epochs.
        warmup_lr_init: Initial learning rate at start of warmup.
        base_lr: Target learning rate at end of warmup.
        min_lr: Minimum learning rate for cosine decay.
        gradient_accumulation_steps: Number of accumulation steps per optimizer update.

    Returns:
        An LR scheduler (SequentialLR or CosineAnnealingLR).
    """
    accum = max(1, int(gradient_accumulation_steps))
    updates_per_epoch = int(math.ceil(steps_per_epoch / accum))
    total_updates = int(epochs * updates_per_epoch)
    warmup_updates = int(warmup_epochs * updates_per_epoch)
    warmup_updates = min(warmup_updates, max(0, total_updates - 1))

    if warmup_updates > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_lr_init / base_lr,
            end_factor=1.0,
            total_iters=warmup_updates,
        )
        cosine_updates = max(1, total_updates - warmup_updates)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_updates,
            eta_min=min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_updates],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_updates),
            eta_min=min_lr,
        )

    return scheduler


def print_scheduler_info(
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    gradient_accumulation_steps: int = 1,
) -> None:
    """Print scheduler setup info (useful for debugging)."""
    accum = max(1, int(gradient_accumulation_steps))
    updates_per_epoch = int(math.ceil(steps_per_epoch / accum))
    total_updates = int(epochs * updates_per_epoch)
    warmup_updates = int(warmup_epochs * updates_per_epoch)
    warmup_updates = min(warmup_updates, max(0, total_updates - 1))

    print("\n📅 Scheduler Setup (optimizer updates):")
    print(f"   batches/epoch:  {steps_per_epoch}")
    print(f"   accum_steps:    {accum}")
    print(f"   updates/epoch:  {updates_per_epoch}")
    print(f"   warmup_updates: {warmup_updates}")
    print(f"   total_updates:  {total_updates}")
