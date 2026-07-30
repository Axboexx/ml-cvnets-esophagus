"""Paired DeLong inference for correlated binary-classification ROC AUCs."""

import math
from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DelongEstimate:
    """AUC and two-sided confidence interval for one model."""

    auc: float
    standard_error: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class PairwiseDelongResult:
    """Paired comparison between two correlated ROC AUCs."""

    first_index: int
    second_index: int
    auc_difference: float
    standard_error: float
    z_statistic: float
    p_value: float


def _compute_midrank(values: FloatArray) -> FloatArray:
    """Return one-based midranks, including average ranks for ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end

    midranks = np.empty(values.size, dtype=np.float64)
    midranks[order] = sorted_ranks
    return midranks


def _validate_inputs(
    targets: Sequence[int],
    scores: Sequence[Sequence[float]],
) -> Tuple[IntArray, FloatArray]:
    raw_labels = np.asarray(targets)
    try:
        numeric_labels = raw_labels.astype(np.float64)
        predictions = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("targets and scores must be numeric") from error

    if numeric_labels.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if predictions.ndim != 2:
        raise ValueError("scores must have shape (n_models, n_cases)")
    if predictions.shape[1] != numeric_labels.size:
        raise ValueError("scores and targets contain different case counts")
    if predictions.shape[0] < 1:
        raise ValueError("at least one model is required")
    if numeric_labels.size == 0:
        raise ValueError("at least one case is required")
    if not np.all(np.isfinite(numeric_labels)) or not np.all(
        np.isin(numeric_labels, (0.0, 1.0))
    ):
        raise ValueError("targets must contain only 0 and 1")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("scores must all be finite")

    labels = numeric_labels.astype(np.int64)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count < 2 or negative_count < 2:
        raise ValueError("DeLong inference requires at least 2 cases per class")
    return labels, predictions


def _fast_delong(
    predictions_sorted_by_target: FloatArray,
    positive_count: int,
) -> Tuple[FloatArray, FloatArray]:
    """Compute correlated AUCs and their DeLong covariance matrix."""
    model_count, total_count = predictions_sorted_by_target.shape
    negative_count = total_count - positive_count
    positive_scores = predictions_sorted_by_target[:, :positive_count]
    negative_scores = predictions_sorted_by_target[:, positive_count:]

    positive_midranks = np.empty_like(positive_scores)
    negative_midranks = np.empty_like(negative_scores)
    total_midranks = np.empty_like(predictions_sorted_by_target)
    for model_index in range(model_count):
        positive_midranks[model_index] = _compute_midrank(positive_scores[model_index])
        negative_midranks[model_index] = _compute_midrank(negative_scores[model_index])
        total_midranks[model_index] = _compute_midrank(
            predictions_sorted_by_target[model_index]
        )

    aucs = total_midranks[:, :positive_count].sum(axis=1) / (
        positive_count * negative_count
    ) - (positive_count + 1.0) / (2.0 * negative_count)
    positive_components = (
        total_midranks[:, :positive_count] - positive_midranks
    ) / negative_count
    negative_components = (
        1.0 - (total_midranks[:, positive_count:] - negative_midranks) / positive_count
    )

    positive_covariance = np.atleast_2d(np.cov(positive_components, bias=False))
    negative_covariance = np.atleast_2d(np.cov(negative_components, bias=False))
    covariance = (
        positive_covariance / positive_count + negative_covariance / negative_count
    )
    return aucs, covariance


def delong_auc_covariance(
    targets: Sequence[int],
    scores: Sequence[Sequence[float]],
) -> Tuple[FloatArray, FloatArray]:
    """Return AUCs and covariance for models evaluated on identical cases."""
    labels, predictions = _validate_inputs(targets=targets, scores=scores)
    order = np.argsort(-labels, kind="stable")
    aucs, covariance = _fast_delong(
        predictions_sorted_by_target=predictions[:, order],
        positive_count=int(labels.sum()),
    )
    if not np.all(np.isfinite(aucs)) or not np.all(np.isfinite(covariance)):
        raise ValueError("DeLong calculation produced non-finite values")
    diagonal = np.diag(covariance)
    if np.any(diagonal < -1e-12):
        raise ValueError("DeLong calculation produced negative variance")
    return aucs, covariance


def _normal_quantile_for_95_percent() -> float:
    return 1.959963984540054


def delong_estimates(
    targets: Sequence[int],
    scores: Sequence[Sequence[float]],
) -> List[DelongEstimate]:
    """Calculate AUC and 95% DeLong confidence interval for each model."""
    aucs, covariance = delong_auc_covariance(targets=targets, scores=scores)
    estimates = []
    for model_index, auc in enumerate(aucs):
        variance = max(float(covariance[model_index, model_index]), 0.0)
        standard_error = math.sqrt(variance)
        margin = _normal_quantile_for_95_percent() * standard_error
        estimates.append(
            DelongEstimate(
                auc=float(auc),
                standard_error=standard_error,
                ci_lower=max(0.0, float(auc) - margin),
                ci_upper=min(1.0, float(auc) + margin),
            )
        )
    return estimates


def pairwise_delong_tests(
    targets: Sequence[int],
    scores: Sequence[Sequence[float]],
) -> List[PairwiseDelongResult]:
    """Run all two-sided paired DeLong AUC comparisons."""
    aucs, covariance = delong_auc_covariance(targets=targets, scores=scores)
    results = []
    for first_index, second_index in combinations(range(len(aucs)), 2):
        difference = float(aucs[first_index] - aucs[second_index])
        raw_variance = float(
            covariance[first_index, first_index]
            + covariance[second_index, second_index]
            - 2.0 * covariance[first_index, second_index]
        )
        if not math.isfinite(raw_variance) or raw_variance < -1e-12:
            raise ValueError("DeLong AUC difference variance is invalid")
        variance = max(raw_variance, 0.0)
        standard_error = math.sqrt(variance)
        if standard_error == 0.0:
            z_statistic = (
                0.0 if difference == 0.0 else math.copysign(math.inf, difference)
            )
            p_value = 1.0 if difference == 0.0 else 0.0
        else:
            z_statistic = difference / standard_error
            p_value = math.erfc(abs(z_statistic) / math.sqrt(2.0))
        results.append(
            PairwiseDelongResult(
                first_index=first_index,
                second_index=second_index,
                auc_difference=difference,
                standard_error=standard_error,
                z_statistic=z_statistic,
                p_value=p_value,
            )
        )
    return results
