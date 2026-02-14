"""
kaggle_torch_utils - Reusable, experiment-agnostic PyTorch training utilities for Kaggle.

Core modules (no optional deps):
    reproducibility - Seed setting & DataLoader worker init
    amp            - Mixed precision helpers (cross-version compatible)
    ema            - Exponential Moving Average
    state_dict     - Robust state dict loading (DataParallel prefix handling)
    checkpoint     - Checkpoint manager with atomic saves & artifact bundling
    serialization  - JSON-safe serialization for numpy/torch types
    model_info     - Parameter count & GFLOPs estimation
    scheduler      - Warmup + cosine annealing scheduler factory
    training       - Gradient accumulation, early stopping, benchmarking

Optional (requires scikit-learn):
    metrics        - Classification metrics computation
"""

__version__ = "0.1.0"

# Reproducibility
from .reproducibility import set_seed, worker_init_fn

# AMP
from .amp import get_autocast_ctx, make_grad_scaler

# EMA
from .ema import EMA

# State dict utilities
from .state_dict import (
    strip_module_prefix,
    add_module_prefix,
    load_state_dict_robust,
)

# Checkpoint management
from .checkpoint import CheckpointManager

# Metrics (optional — requires scikit-learn)
try:
    from .metrics import compute_classification_metrics, compute_per_class_metrics
except ImportError:
    pass

# Serialization
from .serialization import json_safe_scalar, safe_json_dict

# Model info
from .model_info import count_params_m, count_trainable_params_m, try_get_gflops, print_model_summary

# Scheduler
from .scheduler import build_warmup_cosine_scheduler, print_scheduler_info

# Training helpers
from .training import (
    EarlyStopping,
    gradient_accumulation_step,
    benchmark_inference,
    validate_dataset_splits,
    compute_class_weights,
    save_environment_info,
    print_environment_info,
    save_train_log_csv,
)

__all__ = [
    # Reproducibility
    "set_seed",
    "worker_init_fn",
    # AMP
    "get_autocast_ctx",
    "make_grad_scaler",
    # EMA
    "EMA",
    # State dict
    "strip_module_prefix",
    "add_module_prefix",
    "load_state_dict_robust",
    # Checkpoint
    "CheckpointManager",
    # Metrics (optional, requires scikit-learn)
    "compute_classification_metrics",
    "compute_per_class_metrics",
    # Serialization
    "json_safe_scalar",
    "safe_json_dict",
    # Model info
    "count_params_m",
    "count_trainable_params_m",
    "try_get_gflops",
    "print_model_summary",
    # Scheduler
    "build_warmup_cosine_scheduler",
    "print_scheduler_info",
    # Training
    "EarlyStopping",
    "gradient_accumulation_step",
    "benchmark_inference",
    "validate_dataset_splits",
    "compute_class_weights",
    "save_environment_info",
    "print_environment_info",
    "save_train_log_csv",
]
