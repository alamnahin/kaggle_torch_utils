"""Training loop helpers: gradient accumulation, early stopping, benchmarking, dataset validation."""

import csv
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# -------------------------
# Early Stopping
# -------------------------
class EarlyStopping:
    """
    Early stopping tracker.

    Usage:
        es = EarlyStopping(patience=20, mode="max")
        for epoch in ...:
            val_metric = evaluate(...)
            improved = es.step(val_metric)
            if es.should_stop:
                break
    """

    def __init__(self, patience: int = 20, mode: str = "max"):
        """
        Args:
            patience: Number of epochs without improvement before stopping.
            mode: "max" (higher is better) or "min" (lower is better).
        """
        self.patience = patience
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.bad_epochs = 0
        self.best_epoch = -1

    def step(self, value: float, epoch: int = -1) -> bool:
        """
        Update with current metric value.

        Args:
            value: Current metric value.
            epoch: Current epoch number (for tracking).

        Returns:
            True if this is a new best, False otherwise.
        """
        if self.mode == "max":
            improved = value > self.best_value
        else:
            improved = value < self.best_value

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return improved

    @property
    def should_stop(self) -> bool:
        """Whether training should stop."""
        return self.bad_epochs >= self.patience


# -------------------------
# Gradient Accumulation Step
# -------------------------
def gradient_accumulation_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler=None,
    grad_clip_norm: float = 1.0,
    step_index: int = 1,
    accum_steps: int = 1,
    total_steps: int = 1,
    ema=None,
) -> None:
    """
    Perform optimizer step with gradient accumulation, clipping, and AMP scaling.

    Call this after `scaler.scale(loss / accum_steps).backward()`.

    Args:
        model: The model.
        optimizer: The optimizer.
        scaler: GradScaler instance.
        scheduler: LR scheduler (optional, stepped per optimizer update).
        grad_clip_norm: Max gradient norm for clipping.
        step_index: Current batch index (1-based) in the epoch.
        accum_steps: Number of accumulation steps per optimizer update.
        total_steps: Total batches in the epoch (for final-batch flushing).
        ema: Optional EMA instance to update after each optimizer step.
    """
    do_step = (step_index % accum_steps == 0) or (step_index == total_steps)
    if do_step:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update()

        optimizer.zero_grad(set_to_none=True)


# -------------------------
# Inference Benchmarking
# -------------------------
@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    warmup_batches: int = 5,
    use_amp: bool = True,
    prepare_batch: Optional[Callable] = None,
) -> Tuple[float, float]:
    """
    Benchmark model inference throughput.

    Args:
        model: The model.
        loader: DataLoader to iterate.
        device: Device string.
        warmup_batches: Number of warmup batches to skip.
        use_amp: Whether to use mixed precision.
        prepare_batch: Optional callable(batch, device) -> (model_input, batch_size).
                       Default: assumes batch is (tensor, ...) or a single tensor.

    Returns:
        (elapsed_seconds, throughput_samples_per_sec). Returns (-1, -1) on failure.
    """
    from .amp import get_autocast_ctx

    def _default_prepare(batch: Any, dev: str):
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        return x.to(dev, non_blocking=True), x.size(0)

    prep = prepare_batch or _default_prepare
    model.eval()

    total_batches = len(loader)
    if total_batches <= warmup_batches:
        warnings.warn(
            f"Insufficient batches for benchmark: {total_batches} total, "
            f"{warmup_batches} warmup needed. Skipping."
        )
        return -1.0, -1.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = None
    n_samples = 0

    for bi, batch in enumerate(loader):
        x, bs = prep(batch, device)
        with get_autocast_ctx(use_amp):
            if isinstance(x, dict):
                _ = model(**x)
            elif isinstance(x, (tuple, list)):
                _ = model(*x)
            else:
                _ = model(x)

        if bi == warmup_batches:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            n_samples = 0

        if start_time is not None:
            n_samples += bs

    if start_time is None or n_samples == 0:
        warnings.warn("Benchmark failed: no samples processed post-warmup")
        return -1.0, -1.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    throughput = n_samples / max(elapsed, 1e-9)

    print(f"   Benchmark: {n_samples} samples in {elapsed:.2f}s → {throughput:.1f} samples/s")
    return float(elapsed), float(throughput)


# -------------------------
# Dataset Validation
# -------------------------
def validate_dataset_splits(
    train_targets: Union[List[int], np.ndarray],
    val_targets: Union[List[int], np.ndarray],
    test_targets: Optional[Union[List[int], np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Print class distribution across splits and warn about imbalance.

    Works with any labeled dataset — just pass the integer targets.

    Args:
        train_targets: Training split labels (1D int array/list).
        val_targets: Validation split labels.
        test_targets: Test split labels (optional).
        class_names: Optional list of class names (index-aligned).
                     If None, uses integer indices as names.
    """
    print("\n🔍 Dataset Validation:")

    train_targets = np.asarray(train_targets)
    val_targets = np.asarray(val_targets)
    if test_targets is not None:
        test_targets = np.asarray(test_targets)

    all_labels = set(train_targets.tolist())
    all_labels.update(val_targets.tolist())
    if test_targets is not None:
        all_labels.update(test_targets.tolist())
    num_classes = max(all_labels) + 1 if all_labels else 0

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    train_dist = Counter(train_targets.tolist())
    val_dist = Counter(val_targets.tolist())
    test_dist = Counter(test_targets.tolist()) if test_targets is not None else {}

    print("\n  Class Distribution:")
    counts = []
    for ci, cname in enumerate(class_names):
        tr = train_dist.get(ci, 0)
        va = val_dist.get(ci, 0)
        te = test_dist.get(ci, 0) if test_dist else 0
        line = f"    {cname:20s} -> Train: {tr:5d} | Val: {va:5d}"
        if test_targets is not None:
            line += f" | Test: {te:5d}"
        print(line)
        counts.append(tr)
        if tr < 10:
            print(f"    ⚠️ Very low train samples for class '{cname}': {tr}")

    if len(counts) > 0 and min(counts) > 0:
        ratio = max(counts) / min(counts)
        if ratio > 10:
            print(
                f"\n  ⚠️  Severe imbalance ratio: {ratio:.1f}:1 (max/min train class)"
            )
            print(f"      Consider: class weighting, focal loss, or resampling")
        elif ratio > 3:
            print(f"\n  ℹ️  Moderate imbalance ratio: {ratio:.1f}:1")

        median_count = np.median(counts)
        for ci, cname in enumerate(class_names):
            if counts[ci] < median_count * 0.5:
                print(
                    f"      ⚠️  Class '{cname}' underrepresented: "
                    f"{counts[ci]} samples (median: {int(median_count)})"
                )
    print()


def compute_class_weights(
    targets: List[int], num_classes: int, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    Args:
        targets: List of integer class labels.
        num_classes: Number of classes.
        device: Target device for the weight tensor.

    Returns:
        Normalized class weight tensor of shape (num_classes,).
    """
    class_counts = Counter(targets)
    weights = []
    missing = []
    for i in range(num_classes):
        c = int(class_counts.get(i, 0))
        if c == 0:
            missing.append(i)
        weights.append(1.0 / max(c, 1))

    if missing:
        warnings.warn(
            f"Some classes are missing in targets: {missing}. "
            "Using weight=1.0 for them."
        )

    mean_w = sum(weights) / max(1, len(weights))
    weights = [w / max(mean_w, 1e-12) for w in weights]

    return torch.tensor(weights, dtype=torch.float32, device=device)


# -------------------------
# Environment Info
# -------------------------
def save_environment_info(
    save_path: Path, extra_packages: Optional[List[str]] = None
) -> None:
    """
    Save training environment info (Python, PyTorch, CUDA, GPU) to a JSON file.

    Args:
        save_path: Path to save the JSON file.
        extra_packages: Optional list of package names (e.g. ["timm", "transformers"])
                        whose versions will be recorded if installed.
    """
    import sys

    info = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "gpu0": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "num_gpus": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
    }

    for pkg in extra_packages or []:
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            info[pkg] = None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(info, f, indent=2)


def print_environment_info(extra_packages: Optional[List[str]] = None) -> None:
    """
    Print training environment info to console.

    Args:
        extra_packages: Optional list of package names (e.g. ["timm", "transformers"])
                        whose versions will be printed if installed.
    """
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Num GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"PyTorch: {torch.__version__}")
    for pkg in extra_packages or []:
        try:
            mod = __import__(pkg)
            print(f"{pkg}: {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            pass
    print("=" * 60)


# -------------------------
# Training Log
# -------------------------
def save_train_log_csv(train_log: List[Dict], save_path: Path) -> None:
    """
    Save training log to CSV.

    Args:
        train_log: List of dicts (one per epoch).
        save_path: Output CSV path.
    """
    if not train_log:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=train_log[0].keys())
        writer.writeheader()
        writer.writerows(train_log)
