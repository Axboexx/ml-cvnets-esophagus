"""Calibration and decision-curve calculations for binary classifiers."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationBin:
    """One probability bin for an empirical calibration curve."""

    mean_predicted: float
    observed_proportion: float
    count: int
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class DecisionCurve:
    """Unstandardized opt-in net benefit at fixed thresholds."""

    thresholds: NDArray[np.float64]
    model: NDArray[np.float64]
    treat_all: NDArray[np.float64]
    treat_none: NDArray[np.float64]


def _validate_binary_predictions(
    targets: NDArray[np.int64], scores: NDArray[np.float64]
) -> None:
    if targets.ndim != 1 or scores.ndim != 1:
        raise ValueError("targets and scores must be one-dimensional")
    if targets.size != scores.size or targets.size == 0:
        raise ValueError("targets and scores must have the same non-zero length")
    if not np.all(np.isin(targets, (0, 1))):
        raise ValueError("targets must contain only 0 and 1")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be finite probabilities in [0, 1]")


def _wilson_interval(
    events: int, count: int, z_value: float = 1.96
) -> tuple[float, float]:
    if count <= 0:
        raise ValueError("Wilson interval requires a positive count")
    proportion = events / count
    denominator = 1.0 + z_value**2 / count
    centre = (proportion + z_value**2 / (2.0 * count)) / denominator
    radius = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / count + z_value**2 / (4.0 * count**2)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def calibration_bins(
    targets: NDArray[np.int64],
    scores: NDArray[np.float64],
    n_bins: int,
) -> tuple[CalibrationBin, ...]:
    """Create quantile bins with observed proportions and Wilson 95% CIs."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    _validate_binary_predictions(targets, scores)
    edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        raise ValueError("calibration requires at least two unique probabilities")
    assignments = np.searchsorted(edges[1:-1], scores, side="left")
    bins = []
    for bin_index in range(edges.size - 1):
        mask = assignments == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        events = int(targets[mask].sum())
        ci_lower, ci_upper = _wilson_interval(events, count)
        bins.append(
            CalibrationBin(
                mean_predicted=float(scores[mask].mean()),
                observed_proportion=float(targets[mask].mean()),
                count=count,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        )
    if len(bins) < 2:
        raise ValueError("calibration requires at least two non-empty bins")
    return tuple(bins)


def decision_curve(
    targets: NDArray[np.int64],
    scores: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> DecisionCurve:
    """Compute opt-in, unstandardized net benefit without resampling."""
    _validate_binary_predictions(targets, scores)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("DCA thresholds must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("DCA thresholds must be finite")
    if np.any((thresholds <= 0.0) | (thresholds >= 1.0)):
        raise ValueError("DCA thresholds must be strictly between 0 and 1")
    predicted_positive = scores[:, None] >= thresholds[None, :]
    positive = targets[:, None] == 1
    true_positive = np.sum(predicted_positive & positive, axis=0)
    false_positive = np.sum(predicted_positive & ~positive, axis=0)
    odds = thresholds / (1.0 - thresholds)
    count = targets.size
    model = true_positive / count - false_positive / count * odds
    prevalence = float(targets.mean())
    treat_all = prevalence - (1.0 - prevalence) * odds
    return DecisionCurve(
        thresholds=thresholds,
        model=np.asarray(model, dtype=np.float64),
        treat_all=np.asarray(treat_all, dtype=np.float64),
        treat_none=np.zeros_like(thresholds),
    )
