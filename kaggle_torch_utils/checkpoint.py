"""Checkpoint manager with atomic saves, rotation, and artifact bundling.

Kaggle-optimized patterns:
- Atomic saves (temp file + rename) to survive session crashes
- Temp/final directory separation to respect /kaggle/working quota
- Clean artifact bundling for "Save & Run All" commits
"""

import csv
import json
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .serialization import json_safe_scalar, safe_json_dict


class CheckpointManager:
    """
    Manages model checkpoints with:
    - Atomic saves (temp file + rename) to avoid corruption on Kaggle timeouts
    - Best model tracking
    - Epoch checkpoint rotation (keep last N)
    - Generic artifact bundling into a clean directory structure

    Args:
        temp_root: Directory for temporary checkpoint storage (fast I/O, e.g. /kaggle/temp).
        final_root: Directory for final output artifacts (e.g. /kaggle/working).
        experiment_name: Name for this experiment (used as subdirectory).
        subdirs: Optional list of subdirectory names to create under final_root.
                 Defaults to ["model", "config", "logs", "metrics", "raw_outputs", "figures"].
    """

    DEFAULT_SUBDIRS = ["model", "config", "logs", "metrics", "raw_outputs", "figures"]

    def __init__(
        self,
        temp_root: Union[str, Path],
        final_root: Union[str, Path],
        experiment_name: str,
        subdirs: Optional[List[str]] = None,
    ):
        self.temp_dir = Path(temp_root) / experiment_name
        self.final_dir = Path(final_root) / experiment_name
        self.pid = os.getpid()

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        for subdir in (subdirs or self.DEFAULT_SUBDIRS):
            (self.final_dir / subdir).mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Directories:")
        print(f"   Temp:  {self.temp_dir}")
        print(f"   Final: {self.final_dir}")

    def _atomic_save(self, obj: Any, final_path: Path) -> bool:
        """Save with temp-file + rename to prevent corruption."""
        tmp_path = final_path.with_suffix(f".{self.pid}.tmp")
        try:
            torch.save(obj, tmp_path)
            tmp_path.replace(final_path)
            return True
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            print(f"⚠️ Checkpoint save failed ({final_path.name}): {repr(e)}")
            return False

    def save_checkpoint(
        self,
        checkpoint: Dict,
        epoch: int,
        is_best: bool,
        save_epoch_ckpt: bool = False,
        epoch_interval: int = 5,
        keep_n: int = 3,
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            checkpoint: Dict containing model state, optimizer state, etc.
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
            save_epoch_ckpt: Whether to save periodic epoch checkpoints.
            epoch_interval: Save epoch checkpoint every N epochs.
            keep_n: Number of epoch checkpoints to keep.
        """
        self._atomic_save(checkpoint, self.temp_dir / "last_checkpoint.pth")
        if is_best:
            self._atomic_save(checkpoint, self.temp_dir / "best_model.pth")

        if (
            save_epoch_ckpt
            and (not is_best)
            and (epoch % max(1, epoch_interval) == 0)
        ):
            ckpt_path = self.temp_dir / f"checkpoint_epoch_{epoch}.pth"
            if self._atomic_save(checkpoint, ckpt_path):
                self._cleanup_old_epochs(keep_n)

    def _cleanup_old_epochs(self, keep_n: int) -> None:
        """Remove old epoch checkpoints, keeping the most recent keep_n."""
        epoch_ckpts = sorted(
            self.temp_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda p: int(
                re.search(r"checkpoint_epoch_(\d+)", p.name).group(1)
            )
            if re.search(r"checkpoint_epoch_(\d+)", p.name)
            else 0,
        )
        if len(epoch_ckpts) <= keep_n:
            return
        for old in epoch_ckpts[:-keep_n]:
            try:
                old.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Generic artifact bundling (experiment-agnostic)
    # ------------------------------------------------------------------

    def copy_final_artifacts(
        self,
        best_ckpt_path: Union[str, Path],
        config_dict: Optional[Dict] = None,
        train_log: Optional[List[Dict]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        arrays: Optional[Dict[str, np.ndarray]] = None,
        extra_configs: Optional[Dict[str, Any]] = None,
        extra_files: Optional[Dict[str, Union[str, Path]]] = None,
        model_key: str = "model",
        ema_key: str = "ema_shadow",
    ) -> None:
        """
        Bundle all final artifacts into a clean directory structure.

        This is experiment-agnostic: it handles model weights, config,
        training logs, scalar metrics, and arbitrary numpy arrays / files.
        For classification-specific artifacts (confusion matrices, per-class
        metrics, etc.), call save_classification_artifacts() separately.

        Args:
            best_ckpt_path: Path to the best model checkpoint.
            config_dict: Training configuration dict (→ config/config.json).
            train_log: List of per-epoch log dicts (→ logs/train_log.csv).
            metrics: Dict of scalar evaluation metrics (→ metrics/eval_results.json).
            arrays: Dict of name → numpy array (→ raw_outputs/<name>.npy).
            extra_configs: Dict of name → dict to save as JSON in config/.
            extra_files: Dict of dest_name → source_path to copy into logs/.
            model_key: Key in the checkpoint dict for model state_dict.
            ema_key: Key in the checkpoint dict for EMA shadow weights.
        """
        best_ckpt_path = Path(best_ckpt_path)

        print("\n" + "=" * 80)
        print("COPYING FINAL ARTIFACTS")
        print("=" * 80)

        model_dir = self.final_dir / "model"
        config_dir = self.final_dir / "config"
        logs_dst = self.final_dir / "logs"
        metrics_dir = self.final_dir / "metrics"
        raw_dir = self.final_dir / "raw_outputs"

        # ---- Model checkpoint ----
        if not best_ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_ckpt_path}")

        shutil.copy2(best_ckpt_path, model_dir / "best_model.pth")

        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
        if isinstance(best_ckpt, dict) and model_key in best_ckpt:
            torch.save(best_ckpt[model_key], model_dir / "model_weights_only.pth")

            # Save EMA weights if present
            if ema_key in best_ckpt and isinstance(best_ckpt[ema_key], dict):
                ema_state = dict(best_ckpt[model_key])
                for k, v in best_ckpt[ema_key].items():
                    if k in ema_state:
                        ema_state[k] = v
                torch.save(ema_state, model_dir / "ema_model_weights_only.pth")

        # ---- Config ----
        if config_dict is not None:
            with open(config_dir / "config.json", "w") as f:
                json.dump(safe_json_dict(config_dict), f, indent=2)

        if extra_configs:
            for name, cfg in extra_configs.items():
                stem = name if name.endswith(".json") else f"{name}.json"
                data = safe_json_dict(cfg) if isinstance(cfg, dict) else cfg
                with open(config_dir / stem, "w") as f:
                    json.dump(data, f, indent=2)

        # ---- Training log ----
        if train_log:
            with open(logs_dst / "train_log.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=train_log[0].keys())
                writer.writeheader()
                writer.writerows(train_log)

        # ---- Extra files ----
        if extra_files:
            for dest_name, src_path in extra_files.items():
                src_path = Path(src_path)
                if src_path.exists():
                    dst = logs_dst / dest_name
                    if src_path.resolve() != dst.resolve():
                        shutil.copy2(src_path, dst)

        # ---- Scalar metrics ----
        if metrics is not None:
            safe_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    safe_metrics[k] = safe_json_dict(v)
                elif not isinstance(v, (np.ndarray, list)):
                    safe_metrics[k] = json_safe_scalar(v)
                else:
                    safe_metrics[k] = json_safe_scalar(v)
            with open(metrics_dir / "eval_results.json", "w") as f:
                json.dump(safe_metrics, f, indent=2)

        # ---- Raw numpy arrays ----
        if arrays:
            for name, arr in arrays.items():
                stem = name if name.endswith(".npy") else f"{name}.npy"
                np.save(raw_dir / stem, np.asarray(arr))

        print(f"✅ Final bundle ready at: {self.final_dir}")

    # ------------------------------------------------------------------
    # Classification-specific helpers (optional, requires scikit-learn)
    # ------------------------------------------------------------------

    def save_classification_artifacts(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        classes: List[str],
        class_to_idx: Optional[Dict[str, int]] = None,
        probs: Optional[np.ndarray] = None,
        save_cm_png: bool = True,
    ) -> None:
        """
        Save classification-specific evaluation artifacts.

        Call this *after* copy_final_artifacts() for classification experiments.
        Requires scikit-learn.

        Args:
            targets: Ground truth labels (1D int array).
            preds: Predicted labels (1D int array).
            classes: List of class names (index-aligned).
            class_to_idx: Mapping of class name → index (saved to config/).
            probs: Predicted probabilities (optional, saved to raw_outputs/).
            save_cm_png: Whether to save confusion matrix as PNG (requires matplotlib).
        """
        try:
            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
            )
        except ImportError:
            raise ImportError(
                "scikit-learn is required for classification artifacts. "
                "Install it with: pip install scikit-learn"
            )

        targets = np.asarray(targets)
        preds = np.asarray(preds)

        config_dir = self.final_dir / "config"
        metrics_dir = self.final_dir / "metrics"
        raw_dir = self.final_dir / "raw_outputs"
        fig_dir = self.final_dir / "figures"

        # Class mapping
        if class_to_idx is not None:
            with open(config_dir / "class_to_idx.json", "w") as f:
                json.dump(class_to_idx, f, indent=2)
        with open(config_dir / "classes.json", "w") as f:
            json.dump(classes, f, indent=2)

        # Confusion matrix
        cm = confusion_matrix(targets, preds)
        np.save(metrics_dir / "confusion_matrix.npy", cm)

        # Classification report
        rep = classification_report(
            targets, preds, target_names=classes, digits=4,
        )
        with open(metrics_dir / "classification_report.txt", "w") as f:
            f.write(rep)

        # Per-class metrics
        labels = list(range(len(classes)))
        per_f1 = f1_score(targets, preds, average=None, labels=labels)
        per_prec = precision_score(
            targets, preds, average=None, labels=labels, zero_division=0,
        )
        per_rec = recall_score(
            targets, preds, average=None, labels=labels, zero_division=0,
        )

        per_class_metrics = {}
        for i, cname in enumerate(classes):
            per_class_metrics[cname] = {
                "f1": float(per_f1[i]),
                "precision": float(per_prec[i]),
                "recall": float(per_rec[i]),
            }
        with open(metrics_dir / "per_class_metrics.json", "w") as f:
            json.dump(per_class_metrics, f, indent=2)

        # Raw outputs
        np.save(raw_dir / "predictions.npy", preds)
        np.save(raw_dir / "targets.npy", targets)
        if probs is not None:
            np.save(raw_dir / "probabilities.npy", probs)

        # Confusion matrix PNG
        if save_cm_png:
            self.save_confusion_matrix_png(
                cm, classes, fig_dir / "confusion_matrix.png"
            )

        print(f"✅ Classification artifacts saved")

    def save_confusion_matrix_png(
        self, cm: np.ndarray, classes: List[str], out_path: Union[str, Path],
    ) -> None:
        """Save confusion matrix as a PNG image (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not installed, skipping confusion matrix PNG")
            return

        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(cm, interpolation="nearest")
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(len(classes)),
                yticks=np.arange(len(classes)),
                xticklabels=classes,
                yticklabels=classes,
                ylabel="True label",
                xlabel="Predicted label",
                title="Confusion Matrix",
            )
            plt.setp(
                ax.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )

            thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ Could not save confusion_matrix.png: {e}")
