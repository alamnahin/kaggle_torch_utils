# kaggle-torch-utils

Reusable PyTorch training utilities extracted for use in Kaggle notebooks and any PyTorch training pipeline.

## Installation

```bash
pip install kaggle-torch-utils

# With optional dependencies (timm, ptflops, matplotlib):
pip install kaggle-torch-utils[full]
```

**In a Kaggle notebook:**
```python
!pip install -q kaggle-torch-utils
```

## Quick Start

```python
from kaggle_torch_utils import (
    set_seed, worker_init_fn,
    get_autocast_ctx, make_grad_scaler,
    EMA, EarlyStopping,
    load_state_dict_robust,
    CheckpointManager,
    build_warmup_cosine_scheduler,
    compute_classification_metrics,
    count_params_m, print_model_summary,
    compute_class_weights,
    print_environment_info,
)
```

## Modules

### `reproducibility` — Deterministic Training
```python
from kaggle_torch_utils import set_seed, worker_init_fn

set_seed(42, deterministic=True)

loader = DataLoader(
    dataset,
    worker_init_fn=worker_init_fn,
    generator=torch.Generator().manual_seed(42),
)
```

### `amp` — Mixed Precision (Cross-Version Compatible)
```python
from kaggle_torch_utils import get_autocast_ctx, make_grad_scaler

scaler = make_grad_scaler(use_amp=True)

with get_autocast_ctx(use_amp=True):
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### `ema` — Exponential Moving Average
```python
from kaggle_torch_utils import EMA

ema = EMA(model, decay=0.9998)

# After each optimizer step:
ema.update()

# For evaluation:
ema.apply_shadow()
val_metrics = evaluate(model, val_loader)
ema.restore()

# Save EMA weights:
torch.save(ema.state_dict(), "ema_weights.pth")
```

### `state_dict` — Robust Checkpoint Loading
```python
from kaggle_torch_utils import load_state_dict_robust

# Automatically handles DataParallel 'module.' prefix mismatches
state_dict = torch.load("model.pth", map_location="cpu")
load_state_dict_robust(model, state_dict)
```

### `checkpoint` — Checkpoint Management
```python
from kaggle_torch_utils import CheckpointManager

ckpt_mgr = CheckpointManager(
    temp_root=Path("/kaggle/temp"),
    final_root=Path("/kaggle/working/experiments"),
    experiment_name="resnet50_run1",
)

# During training:
ckpt_mgr.save_checkpoint(checkpoint_dict, epoch=10, is_best=True)

# After training:
ckpt_mgr.copy_final_artifacts(
    best_ckpt_path=ckpt_mgr.temp_dir / "best_model.pth",
    config_dict=config.__dict__,
    train_log=train_log,
    metrics=test_metrics,
)
```

### `metrics` — Classification Metrics
```python
from kaggle_torch_utils import compute_classification_metrics, compute_per_class_metrics

metrics = compute_classification_metrics(targets, preds, probs=probs)
# -> {'acc1', 'macro_f1', 'micro_f1', 'weighted_f1', 'macro_precision', 'macro_recall', 'macro_auc'}

per_class = compute_per_class_metrics(targets, preds, classes=["cat", "dog", "bird"])
# -> {'cat': {'f1': ..., 'precision': ..., 'recall': ...}, ...}
```

### `model_info` — Parameter Count & GFLOPs
```python
from kaggle_torch_utils import count_params_m, try_get_gflops, print_model_summary

print(f"Params: {count_params_m(model):.2f}M")

gflops = try_get_gflops(model, input_res=(3, 224, 224))  # requires ptflops

print_model_summary(model, input_res=(3, 224, 224))
```

### `scheduler` — Warmup + Cosine Annealing
```python
from kaggle_torch_utils import build_warmup_cosine_scheduler

scheduler = build_warmup_cosine_scheduler(
    optimizer,
    epochs=80,
    steps_per_epoch=len(train_loader),
    warmup_epochs=5,
    warmup_lr_init=1e-6,
    base_lr=5e-4,
    min_lr=1e-6,
    gradient_accumulation_steps=2,
)

# In training loop (call per optimizer step, not per batch):
scheduler.step()
```

### `training` — Training Helpers
```python
from kaggle_torch_utils import (
    EarlyStopping,
    gradient_accumulation_step,
    benchmark_inference,
    validate_dataset_splits,
    compute_class_weights,
    print_environment_info,
    save_train_log_csv,
)

# Early stopping
es = EarlyStopping(patience=20, mode="max")
if es.step(val_f1):
    save_best_model()
if es.should_stop:
    break

# Class weights for imbalanced data
weights = compute_class_weights(train_targets, num_classes=10, device="cuda")
loss = F.cross_entropy(logits, y, weight=weights)

# Dataset validation (prints class distribution & imbalance warnings)
validate_dataset_splits(train_targets, val_targets, class_names=["cat", "dog"])

# Inference benchmarking
elapsed, throughput = benchmark_inference(model, test_loader)

# Environment info
print_environment_info()
```

### `serialization` — Safe JSON Serialization
```python
from kaggle_torch_utils import json_safe_scalar, safe_json_dict

# Safely convert numpy/torch scalars for JSON
json_safe_scalar(np.float32(0.95))  # -> 0.95
json_safe_scalar(torch.tensor(42))  # -> 42

# Convert entire dicts
safe_dict = safe_json_dict({"loss": torch.tensor(0.5), "acc": np.float64(0.9)})
```

## Package Structure

```
kaggle_torch_utils/
    __init__.py          # Public API exports
    reproducibility.py   # set_seed, worker_init_fn
    amp.py               # get_autocast_ctx, make_grad_scaler
    ema.py               # EMA class
    state_dict.py        # strip/add module prefix, load_state_dict_robust
    checkpoint.py        # CheckpointManager
    metrics.py           # compute_classification_metrics, compute_per_class_metrics
    serialization.py     # json_safe_scalar, safe_json_dict
    model_info.py        # count_params_m, try_get_gflops, print_model_summary
    scheduler.py         # build_warmup_cosine_scheduler
    training.py          # EarlyStopping, gradient_accumulation_step, benchmark_inference, etc.
```

## Dependencies

**Required:** `torch>=1.10`, `numpy>=1.20`

**Optional:** `scikit-learn` (for metrics), `timm`, `ptflops` (for GFLOPs), `matplotlib` (for confusion matrix plots), `tqdm`

## License

MIT
