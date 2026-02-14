"""Metrics computation utilities for classification tasks."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    probs: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    compute_auc: bool = True,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        targets: Ground truth labels (1D array of ints).
        preds: Predicted labels (1D array of ints).
        probs: Predicted probabilities (2D array, shape [N, C]). Required for AUC.
        num_classes: Number of classes. Inferred from targets if not provided.
        compute_auc: Whether to compute macro AUC-ROC.

    Returns:
        Dict with keys: acc1, macro_f1, micro_f1, weighted_f1,
                        macro_precision, macro_recall, macro_auc.
    """
    targets = np.asarray(targets)
    preds = np.asarray(preds)

    correct = int((preds == targets).sum())
    total = len(targets)

    macro_f1 = float(f1_score(targets, preds, average="macro"))
    micro_f1 = float(f1_score(targets, preds, average="micro"))
    weighted_f1 = float(f1_score(targets, preds, average="weighted"))
    macro_precision = float(
        precision_score(targets, preds, average="macro", zero_division=0)
    )
    macro_recall = float(
        recall_score(targets, preds, average="macro", zero_division=0)
    )

    macro_auc = -1.0
    if compute_auc and probs is not None:
        if num_classes is None:
            num_classes = probs.shape[1] if probs.ndim == 2 else int(targets.max()) + 1
        try:
            macro_auc = float(
                roc_auc_score(
                    targets,
                    probs.astype(np.float64),
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(num_classes),
                )
            )
        except Exception as e:
            print(f"⚠️ AUC computation failed: {e}")
            macro_auc = -1.0

    return {
        "acc1": correct / max(1, total),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_auc": macro_auc,
    }


def compute_per_class_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    classes: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class F1, precision, and recall.

    Args:
        targets: Ground truth labels.
        preds: Predicted labels.
        classes: List of class names (index-aligned).

    Returns:
        Dict mapping class name -> {"f1": ..., "precision": ..., "recall": ...}.
    """
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    labels = list(range(len(classes)))

    per_f1 = f1_score(targets, preds, average=None, labels=labels)
    per_prec = precision_score(
        targets, preds, average=None, labels=labels, zero_division=0
    )
    per_rec = recall_score(
        targets, preds, average=None, labels=labels, zero_division=0
    )

    result = {}
    for i, cname in enumerate(classes):
        result[cname] = {
            "f1": float(per_f1[i]),
            "precision": float(per_prec[i]),
            "recall": float(per_rec[i]),
        }
    return result
