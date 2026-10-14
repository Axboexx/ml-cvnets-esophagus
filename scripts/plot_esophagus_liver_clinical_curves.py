#!/usr/bin/env python3
"""Plot E+L calibration and decision curves for internal and external data."""

import argparse
import hashlib
import logging
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

try:
    from scripts.statistical_tests.clinical_curves import (
        calibration_bins as _calibration_bins,
    )
    from scripts.statistical_tests.clinical_curves import (
        decision_curve as _decision_curve,
    )
    from scripts.statistical_tests.validation_auc import (
        CasePrediction,
        load_prediction_partition,
        select_best_epoch,
    )
except ModuleNotFoundError:
    from statistical_tests.clinical_curves import (
        calibration_bins as _calibration_bins,
    )
    from statistical_tests.clinical_curves import (
        decision_curve as _decision_curve,
    )
    from statistical_tests.validation_auc import (
        CasePrediction,
        load_prediction_partition,
        select_best_epoch,
    )

LOGGER = logging.getLogger(__name__)

INTERNAL_RUN = Path(
    "F3_esophagus_liver/f3_esophagus_liver_tongji82_min4cm_4cm_nomargin_seed42"
)
EXTERNAL_RUN = Path(
    "F3_esophagus_liver/"
    "f3_esophagus_liver_tongji82_min4cm_4cm_nomargin_seed42_youyi_external"
)
FIGURE_STEMS = (
    "esophagus_liver_internal_calibration",
    "esophagus_liver_external_calibration",
    "esophagus_liver_internal_dca",
    "esophagus_liver_external_dca",
)

MODEL_COLOR = "#0F4D92"
REFERENCE_COLOR = "#767676"
TREAT_ALL_COLOR = "#B64342"


@dataclass(frozen=True)
class DatasetInput:
    """Validated case-level scores for one evaluation dataset."""

    display_name: str
    targets: NDArray[np.int64]
    scores: NDArray[np.float64]
    calibration_bins: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hashes(input_hashes: Mapping[Path, str]) -> None:
    changed = [
        str(path) for path, digest in input_hashes.items() if _sha256(path) != digest
    ]
    if changed:
        raise ValueError(f"input files changed during analysis: {changed}")


def _arrays(
    predictions: Mapping[str, CasePrediction],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    case_ids = sorted(predictions)
    targets = np.asarray(
        [predictions[case_id].target for case_id in case_ids], dtype=np.int64
    )
    scores = np.asarray(
        [predictions[case_id].score for case_id in case_ids], dtype=np.float64
    )
    if np.unique(targets).size != 2:
        raise ValueError("each evaluation dataset must contain both target classes")
    return targets, scores


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
        }
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _plot_calibration(dataset: DatasetInput, output_dir: Path, stem: str) -> None:
    bins = _calibration_bins(
        dataset.targets, dataset.scores, n_bins=dataset.calibration_bins
    )
    mean_predicted = np.asarray([item.mean_predicted for item in bins])
    observed = np.asarray([item.observed_proportion for item in bins])
    lower = observed - np.asarray([item.ci_lower for item in bins])
    upper = np.asarray([item.ci_upper for item in bins]) - observed

    figure, axis = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=REFERENCE_COLOR,
        linestyle="-",
        linewidth=1.2,
        label="Ideal calibration",
    )
    axis.errorbar(
        mean_predicted,
        observed,
        yerr=np.vstack([lower, upper]),
        color=MODEL_COLOR,
        ecolor=MODEL_COLOR,
        linestyle="-",
        marker="o",
        markersize=5.0,
        linewidth=1.8,
        elinewidth=1.0,
        capsize=3.0,
        label="E+L (Wilson 95% CI)",
    )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Mean Predicted Probability",
        ylabel="Observed Proportion",
        title=f"Calibration Curve — {dataset.display_name}",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="lower right")
    axis.text(
        0.02,
        0.98,
        f"n = {dataset.targets.size}; {len(bins)} bins",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=REFERENCE_COLOR,
    )
    _save_figure(figure, output_dir, stem)


def _plot_dca(dataset: DatasetInput, output_dir: Path, stem: str) -> None:
    thresholds = np.linspace(0.05, 0.95, 91, dtype=np.float64)
    curve = _decision_curve(dataset.targets, dataset.scores, thresholds)
    upper_limit = max(0.1, float(np.max(curve.model)), float(dataset.targets.mean()))
    upper_limit = min(1.0, upper_limit + 0.05)

    figure, axis = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
    axis.plot(
        curve.thresholds,
        curve.model,
        color=MODEL_COLOR,
        linestyle="-",
        linewidth=2.0,
        label="E+L",
    )
    axis.plot(
        curve.thresholds,
        curve.treat_all,
        color=TREAT_ALL_COLOR,
        linestyle="-",
        linewidth=1.3,
        label="Treat All",
    )
    axis.plot(
        curve.thresholds,
        curve.treat_none,
        color=REFERENCE_COLOR,
        linestyle="-",
        linewidth=1.3,
        label="Treat None",
    )
    axis.set(
        xlim=(0.05, 0.95),
        ylim=(-0.05, upper_limit),
        xlabel="Threshold Probability",
        ylabel="Net Benefit",
        title=f"Decision Curve Analysis — {dataset.display_name}",
    )
    axis.legend(loc="upper right")
    axis.text(
        0.02,
        0.98,
        f"n = {dataset.targets.size}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=REFERENCE_COLOR,
    )
    _save_figure(figure, output_dir, stem)


def _load_inputs(
    validation_path: Path,
    external_path: Path,
    metrics_path: Path,
    epoch_override: int | None,
) -> tuple[DatasetInput, DatasetInput, dict[Path, str]]:
    paths = [validation_path, external_path]
    if epoch_override is None:
        paths.append(metrics_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required input files not found: {missing}")

    selected_epoch = (
        epoch_override
        if epoch_override is not None
        else select_best_epoch(metrics_path)
    )
    input_hashes = {path: _sha256(path) for path in paths}
    if epoch_override is None and select_best_epoch(metrics_path) != selected_epoch:
        raise ValueError("best epoch changed while inputs were being captured")

    internal_predictions = load_prediction_partition(
        validation_path, selected_epoch, "validation"
    )
    external_predictions = load_prediction_partition(
        external_path, None, "youyi_external"
    )
    internal_targets, internal_scores = _arrays(internal_predictions)
    external_targets, external_scores = _arrays(external_predictions)
    _verify_hashes(input_hashes)
    LOGGER.info(
        "Selected internal epoch %d; internal n=%d, external n=%d",
        selected_epoch,
        internal_targets.size,
        external_targets.size,
    )
    return (
        DatasetInput("Internal Validation", internal_targets, internal_scores, 5),
        DatasetInput("External Validation", external_targets, external_scores, 10),
        input_hashes,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path.home() / "work" / "results"
    )
    parser.add_argument("--validation-predictions", type=Path, default=None)
    parser.add_argument("--external-predictions", type=Path, default=None)
    parser.add_argument("--training-metrics", type=Path, default=None)
    parser.add_argument("--internal-epoch", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate two calibration plots and two DCA plots."""
    arguments = build_parser().parse_args(argv)
    if arguments.internal_epoch is not None and arguments.internal_epoch < 0:
        raise ValueError("--internal-epoch must be non-negative")
    results_root = arguments.results_root.expanduser().resolve()
    internal_run = results_root / INTERNAL_RUN
    external_run = results_root / EXTERNAL_RUN
    validation_path = (
        arguments.validation_predictions.expanduser().resolve()
        if arguments.validation_predictions is not None
        else internal_run / "validation_predictions.csv"
    )
    external_path = (
        arguments.external_predictions.expanduser().resolve()
        if arguments.external_predictions is not None
        else external_run / "external_eval_predictions.csv"
    )
    metrics_path = (
        arguments.training_metrics.expanduser().resolve()
        if arguments.training_metrics is not None
        else internal_run / "training_metrics.csv"
    )
    output_dir = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else results_root / "esophagus_liver_clinical_curves"
    )
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    internal, external, input_hashes = _load_inputs(
        validation_path,
        external_path,
        metrics_path,
        arguments.internal_epoch,
    )
    _configure_style()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-"
    ) as temp_name:
        temp_dir = Path(temp_name)
        _plot_calibration(internal, temp_dir, FIGURE_STEMS[0])
        _plot_calibration(external, temp_dir, FIGURE_STEMS[1])
        _plot_dca(internal, temp_dir, FIGURE_STEMS[2])
        _plot_dca(external, temp_dir, FIGURE_STEMS[3])
        _verify_hashes(input_hashes)
        temp_dir.rename(output_dir)
    LOGGER.info("Completed four E+L clinical utility figures in %s", output_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    LOGGER.setLevel(logging.INFO)
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(2) from error
