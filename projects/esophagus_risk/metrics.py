from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    accuracy: float
    f1: float
    sensitivity: float
    specificity: float


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    probabilities = probabilities.astype(np.float64)

    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_count = int(pos_mask.sum())
    neg_count = int(neg_mask.sum())
    if pos_count == 0 or neg_count == 0:
        return 0.5

    order = np.argsort(probabilities, kind="mergesort")
    sorted_probs = probabilities[order]
    ranks = np.empty_like(sorted_probs, dtype=np.float64)

    start = 0
    while start < len(sorted_probs):
        end = start + 1
        while end < len(sorted_probs) and sorted_probs[end] == sorted_probs[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = avg_rank
        start = end

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    original_ranks = ranks[inv_order]
    rank_sum = float(original_ranks[pos_mask].sum())
    return (rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> BinaryMetrics:
    labels = labels.astype(np.int64)
    probabilities = probabilities.astype(np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    accuracy = _safe_divide(tp + tn, len(labels))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    specificity = _safe_divide(tn, tn + fp)

    return BinaryMetrics(
        auc=binary_auc(labels=labels, probabilities=probabilities),
        accuracy=accuracy,
        f1=f1,
        sensitivity=recall,
        specificity=specificity,
    )
