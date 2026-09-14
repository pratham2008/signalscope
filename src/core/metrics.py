"""Dependency-light binary-classification metrics used by training and reports."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _arrays(labels: Iterable[float], probabilities: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("labels and probabilities must be one-dimensional and equally sized")
    if len(y) == 0:
        raise ValueError("at least one prediction is required")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be 0 (real) or 1 (synthetic)")
    return y, np.clip(p, 0.0, 1.0)


def roc_auc(labels: Iterable[float], probabilities: Iterable[float]) -> float:
    """AUC using average ranks, including correct handling of tied scores."""
    y, p = _arrays(labels, probabilities)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def confusion(labels: Iterable[float], probabilities: Iterable[float], threshold: float) -> dict[str, int]:
    y, p = _arrays(labels, probabilities)
    predicted = p >= threshold
    return {
        "tn": int(((~predicted) & (y == 0)).sum()),
        "fp": int((predicted & (y == 0)).sum()),
        "fn": int(((~predicted) & (y == 1)).sum()),
        "tp": int((predicted & (y == 1)).sum()),
    }


def choose_threshold_at_fpr(labels: Iterable[float], probabilities: Iterable[float], target_fpr: float = 0.05) -> float:
    y, p = _arrays(labels, probabilities)
    real_scores = np.sort(p[y == 0])
    if len(real_scores) == 0:
        return 0.5
    # Highest threshold whose false-positive rate is no greater than target.
    index = max(0, math.ceil((1.0 - target_fpr) * len(real_scores)) - 1)
    return float(min(1.0, np.nextafter(real_scores[index], np.inf)))


def summary(labels: Iterable[float], probabilities: Iterable[float], threshold: float = 0.5) -> dict[str, float | int | dict[str, int]]:
    y, p = _arrays(labels, probabilities)
    matrix = confusion(y, p, threshold)
    total = len(y)
    precision_real = matrix["tn"] / max(matrix["tn"] + matrix["fn"], 1)
    recall_real = matrix["tn"] / max(matrix["tn"] + matrix["fp"], 1)
    precision_fake = matrix["tp"] / max(matrix["tp"] + matrix["fp"], 1)
    recall_fake = matrix["tp"] / max(matrix["tp"] + matrix["fn"], 1)
    f1_real = 2 * precision_real * recall_real / max(precision_real + recall_real, 1e-12)
    f1_fake = 2 * precision_fake * recall_fake / max(precision_fake + recall_fake, 1e-12)
    return {
        "samples": total,
        "roc_auc": roc_auc(y, p),
        "accuracy": (matrix["tp"] + matrix["tn"]) / total,
        "macro_f1": (f1_real + f1_fake) / 2,
        "false_positive_rate": matrix["fp"] / max(matrix["fp"] + matrix["tn"], 1),
        "threshold": threshold,
        "confusion_matrix": matrix,
    }
