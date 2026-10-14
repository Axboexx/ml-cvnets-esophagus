#!/usr/bin/env python3
"""Plot an rms::val.prob-style external calibration curve for one model."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray
from statsmodels.nonparametric.smoothers_lowess import lowess

try:
    from scripts.statistical_tests.validation_auc import load_prediction_partition
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from statistical_tests.validation_auc import load_prediction_partition

LOGGER = logging.getLogger(__name__)

DEFAULT_EXTERNAL_RUN = Path(
    "F3_esophagus_liver/"
    "f3_esophagus_liver_tongji82_min4cm_4cm_nomargin_seed42_youyi_external"
)
DEFAULT_STAGE = "youyi_external"
MODEL_COLOR = "#0F4D92"
LOGISTIC_COLOR = "#B64342"
REFERENCE_COLOR = "#888888"


@dataclass(frozen=True)
class CalibrationSummary:
    """External calibration statistics displayed on the figure."""

    sample_size: int
    events: int
    auc: float
    brier: float
    intercept: float
    slope: float
    emax: float
    eavg: float


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
        }
    )


def _load_external_predictions(
    prediction_path: Path, stage: str
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    predictions = load_prediction_partition(prediction_path, None, stage)
    case_ids = sorted(predictions)
    targets = np.asarray(
        [predictions[case_id].target for case_id in case_ids], dtype=np.int64
    )
    scores = np.asarray(
        [predictions[case_id].score for case_id in case_ids], dtype=np.float64
    )
    if np.unique(targets).tolist() != [0, 1]:
        raise ValueError("external predictions must contain both target classes")
    return targets, scores


def _fit_logistic_calibration(
    targets: NDArray[np.int64], scores: NDArray[np.float64]
) -> tuple[float, float]:
    clipped = np.clip(scores, 1e-8, 1.0 - 1e-8)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones(scores.size), logits])
    fitted_model = sm.GLM(
        targets,
        design,
        family=sm.families.Binomial(),
    ).fit()
    coefficients = np.asarray(fitted_model.params, dtype=np.float64)
    return float(coefficients[0]), float(coefficients[1])


def _auc(
    targets: NDArray[np.int64], scores: NDArray[np.float64]
) -> float:
    positive = scores[targets == 1]
    negative = scores[targets == 0]
    comparisons = positive[:, None] - negative[None, :]
    concordant = np.count_nonzero(comparisons > 0.0)
    tied = np.count_nonzero(comparisons == 0.0)
    return float((concordant + 0.5 * tied) / comparisons.size)


def _grouped_observations(
    targets: NDArray[np.int64],
    scores: NDArray[np.float64],
    observations_per_group: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if observations_per_group < 2:
        raise ValueError("observations_per_group must be at least 2")
    group_count = max(2, round(scores.size / observations_per_group))
    ordered_indices = np.argsort(scores, kind="stable")
    groups = np.array_split(ordered_indices, group_count)
    mean_scores = np.asarray([scores[group].mean() for group in groups])
    event_rates = np.asarray([targets[group].mean() for group in groups])
    return mean_scores, event_rates


def _calibration_components(
    targets: NDArray[np.int64], scores: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    CalibrationSummary,
]:
    smoothed = lowess(
        targets,
        scores,
        frac=2.0 / 3.0,
        it=0,
        delta=0.01 * float(np.ptp(scores)),
        is_sorted=False,
        return_sorted=True,
    )
    smooth_x = np.asarray(smoothed[:, 0], dtype=np.float64)
    smooth_y = np.asarray(smoothed[:, 1], dtype=np.float64)

    intercept, slope = _fit_logistic_calibration(targets, scores)
    logistic_x = np.linspace(1e-4, 1.0 - 1e-4, 500)
    logistic_logit = np.log(logistic_x / (1.0 - logistic_x))
    logistic_y = 1.0 / (
        1.0 + np.exp(-np.clip(intercept + slope * logistic_logit, -40.0, 40.0))
    )

    smooth_at_scores = np.interp(scores, smooth_x, smooth_y)
    errors = np.abs(scores - smooth_at_scores)
    summary = CalibrationSummary(
        sample_size=int(targets.size),
        events=int(targets.sum()),
        auc=_auc(targets, scores),
        brier=float(np.mean((scores - targets) ** 2)),
        intercept=intercept,
        slope=slope,
        emax=float(np.max(errors)),
        eavg=float(np.mean(errors)),
    )
    return smooth_x, smooth_y, logistic_x, logistic_y, summary


def _plot(
    targets: NDArray[np.int64],
    scores: NDArray[np.float64],
    observations_per_group: int,
    output_dir: Path,
    model_label: str,
    output_stem: str,
) -> CalibrationSummary:
    smooth_x, smooth_y, logistic_x, logistic_y, summary = _calibration_components(
        targets, scores
    )
    group_x, group_y = _grouped_observations(
        targets, scores, observations_per_group
    )

    figure, axis = plt.subplots(figsize=(6.0, 5.4), constrained_layout=True)
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=REFERENCE_COLOR,
        linewidth=3.0,
        label="Ideal",
        zorder=1,
    )
    axis.plot(
        smooth_x,
        smooth_y,
        color=MODEL_COLOR,
        linewidth=2.1,
        label="Nonparametric (LOWESS)",
        zorder=3,
    )
    axis.plot(
        logistic_x,
        logistic_y,
        color=LOGISTIC_COLOR,
        linewidth=1.6,
        label="Logistic calibration",
        zorder=2,
    )
    axis.scatter(
        group_x,
        group_y,
        marker="^",
        s=34,
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        label=f"Grouped observations (m ≈ {observations_per_group})",
        zorder=4,
    )

    rug_height = 0.018
    axis.vlines(
        scores[targets == 0],
        0.0,
        rug_height,
        color="#777777",
        alpha=0.35,
        linewidth=0.5,
        zorder=0,
    )
    axis.vlines(
        scores[targets == 1],
        0.0,
        rug_height,
        color=MODEL_COLOR,
        alpha=0.35,
        linewidth=0.5,
        zorder=0,
    )

    statistics = (
        f"n = {summary.sample_size}  |  Events = {summary.events}\n"
        f"AUC = {summary.auc:.3f}  |  Brier = {summary.brier:.3f}\n"
        f"Intercept = {summary.intercept:.3f}  |  Slope = {summary.slope:.3f}\n"
        f"Eavg = {summary.eavg:.3f}  |  Emax = {summary.emax:.3f}"
    )
    axis.text(
        0.025,
        0.975,
        statistics,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.92,
        },
        zorder=5,
    )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Predicted Probability",
        ylabel="Observed Probability",
        title=f"{model_label} Calibration Curve — External Validation",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="lower right")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_stem
    figure.savefig(f"{output_path}.png", dpi=600, bbox_inches="tight")
    figure.savefig(f"{output_path}.pdf", bbox_inches="tight")
    plt.close(figure)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a val.prob-style E+L external calibration curve."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="External prediction CSV; defaults below --results-root.",
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path.home() / "work" / "results"
    )
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--observations-per-group", type=int, default=20)
    parser.add_argument("--model-label", default="E+L")
    parser.add_argument(
        "--output-stem",
        default="esophagus_liver_external_calibration_valprob",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home()
        / "work"
        / "results"
        / "esophagus_liver_external_calibration_valprob",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Load external case predictions and export a val.prob-style figure."""
    options = _parser().parse_args(arguments)
    results_root = options.results_root.expanduser().resolve()
    prediction_path = (
        options.predictions.expanduser().resolve()
        if options.predictions is not None
        else results_root / DEFAULT_EXTERNAL_RUN / "external_eval_predictions.csv"
    )
    output_dir = options.output_dir.expanduser().resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f"prediction CSV not found: {prediction_path}")

    _configure_style()
    targets, scores = _load_external_predictions(prediction_path, options.stage)
    summary = _plot(
        targets,
        scores,
        options.observations_per_group,
        output_dir,
        options.model_label,
        options.output_stem,
    )
    LOGGER.info(
        "Saved %s external calibration figure: n=%d, AUC=%.3f, slope=%.3f, %s",
        options.model_label,
        summary.sample_size,
        summary.auc,
        summary.slope,
        output_dir,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    raise SystemExit(main())
