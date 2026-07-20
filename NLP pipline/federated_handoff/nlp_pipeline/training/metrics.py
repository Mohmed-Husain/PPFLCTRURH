"""
Evaluation metrics for hierarchical multi-label classification.

**Kept verbatim** from Phase 2 notebook:
  - ``evaluate_multilabel``            — full metric suite
  - ``check_hierarchical_consistency`` — hierarchy violation analysis
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

from nlp_pipeline.utils.helpers import timestamp


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    model_name: str = "Model",
    *,
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive multi-label evaluation.

    Returns a dict of metrics and optionally prints a detailed report.

    Kept from notebook 2 ``evaluate_multilabel()``.

    Parameters
    ----------
    y_true : np.ndarray of shape (N, C)
    y_pred : np.ndarray of shape (N, C)
    label_names : list[str]
    model_name : str
    print_report : bool
    """
    metrics: Dict[str, Any] = {
        "model": model_name,
        "timestamp": timestamp(),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "jaccard_micro": float(jaccard_score(y_true, y_pred, average="micro", zero_division=0)),
        "jaccard_macro": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-label metrics
    per_label: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(label_names):
        support = int(y_true[:, i].sum())
        if support > 0:
            per_label[name] = {
                "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "support": support,
            }
    metrics["per_label"] = per_label

    # Print report
    if print_report:
        print(f"\n{'=' * 65}")
        print(f"  EVALUATION REPORT: {model_name}")
        print(f"{'=' * 65}")
        print(f"  Hamming Loss      : {metrics['hamming_loss']:.4f}")
        print(f"  Subset Accuracy   : {metrics['subset_accuracy']:.4f}")
        print(f"  F1 Micro          : {metrics['f1_micro']:.4f}")
        print(f"  F1 Macro          : {metrics['f1_macro']:.4f}")
        print(f"  F1 Weighted       : {metrics['f1_weighted']:.4f}")
        print(f"  F1 Samples        : {metrics['f1_samples']:.4f}")
        print(f"  Precision (micro) : {metrics['precision_micro']:.4f}")
        print(f"  Recall (micro)    : {metrics['recall_micro']:.4f}")
        print(f"  Jaccard (micro)   : {metrics['jaccard_micro']:.4f}")
        print(f"{'=' * 65}")

        print(f"\n  {'Label':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'-' * 62}")
        for name, vals in sorted(per_label.items()):
            print(
                f"  {name:<30} {vals['precision']:>8.3f} {vals['recall']:>8.3f} "
                f"{vals['f1']:>8.3f} {vals['support']:>8}"
            )
        print(f"  {'-' * 62}")

    return metrics


def check_hierarchical_consistency(
    y_pred: np.ndarray,
    label_names: List[str],
    child_to_parent: Dict[str, str],
) -> Dict[str, Any]:
    """
    Check how often predictions violate the label hierarchy
    (child predicted without parent).

    Kept from notebook 2 ``check_hierarchical_consistency()``.

    Parameters
    ----------
    y_pred : np.ndarray  (N, C)
    label_names : list[str]
    child_to_parent : dict  mapping child→parent label strings
    """
    total_preds = 0
    violations = 0
    violation_details: Counter = Counter()

    for sample in y_pred:
        active_labels = set(
            label_names[i] for i in range(len(label_names)) if sample[i] == 1
        )
        for child_label in active_labels:
            if child_label in child_to_parent:
                parent_label = child_to_parent[child_label]
                total_preds += 1
                if parent_label not in active_labels:
                    violations += 1
                    violation_details[f"{child_label} without {parent_label}"] += 1

    consistency_rate = 1 - (violations / max(total_preds, 1))
    return {
        "total_child_predictions": total_preds,
        "violations": violations,
        "consistency_rate": consistency_rate,
        "top_violations": dict(violation_details.most_common(10)),
    }
