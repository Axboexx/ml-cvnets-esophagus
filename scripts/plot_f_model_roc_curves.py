#!/usr/bin/env python3
"""Plot target=1 internal and external ROC curves for F-series risk models."""

import argparse
import csv
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
    from scripts.compare_validation_auc_delong import (
        capture_input_hashes,
        verify_input_hashes,
    )
    from scripts.statistical_tests.delong import DelongEstimate, delong_estimates
    from scripts.statistical_tests.validation_auc import (
        ModelInput,
        align_predictions,
        resolve_model_inputs,
    )
except ModuleNotFoundError:
    from compare_validation_auc_delong import capture_input_hashes, verify_input_hashes
    from statistical_tests.delong import DelongEstimate, delong_estimates
    from statistical_tests.validation_auc import (
        ModelInput,
        align_predictions,
        resolve_model_inputs,
    )

LOGGER = logging.getLogger(__name__)
@dataclass(frozen=True)
class ModelSpec:
    name: str
    validation_run: str
    external_predictions: str
@dataclass(frozen=True)
class FigureSpec:
    stem: str
    analysis_set: str
    models: tuple[str, ...]
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "F1",
        "F1/f1_tongji82_min4cm_4cm_nomargin_seed42",
        "F1/F1_external/f1_youyi_external_best/external_eval_predictions.csv",
    ),
    ModelSpec(
        "F2a",
        "F2a/f2a_tongji82_min4cm_4cm_nomargin_4seg_seed42",
        "F2a/f2a_tongji82_min4cm_4cm_nomargin_4seg_seed42_youyi_external/"
        "external_eval_predictions.csv",
    ),
    ModelSpec(
        "F2b",
        "F2b/f2b_tongji82_min4cm_4cm_nomargin_4seg_seed42",
        "F2b/f2b_tongji82_min4cm_4cm_nomargin_4seg_seed42_youyi_exterl/"
        "external_eval_predictions.csv",
    ),
    ModelSpec(
        "E+L",
        "F3_esophagus_liver/"
        "f3_esophagus_liver_tongji82_min4cm_4cm_nomargin_seed42",
        "F3_esophagus_liver/"
        "f3_esophagus_liver_tongji82_min4cm_4cm_nomargin_seed42_youyi_external/"
        "external_eval_predictions.csv",
    ),
    ModelSpec(
        "E+S",
        "F3_esophagus_spleen/"
        "f3_esophagus_spleen_tongji82_min4cm_4cm_nomargin_seed42",
        "F3_esophagus_spleen/"
        "f3_esophagus_spleen_tongji82_min4cm_4cm_nomargin_seed42_youyi_external/"
        "external_eval_predictions.csv",
    ),
    ModelSpec(
        "E+PVS",
        "F3_esophagus_portal_venous_system/"
        "f3_esophagus_portal_venous_system_tongji82_min4cm_4cm_nomargin_seed42",
        "F3_esophagus_portal_venous_system/"
        "f3_esophagus_portal_venous_system_tongji82_min4cm_4cm_nomargin_seed42_"
        "youyi_external/external_eval_predictions.csv",
    ),
    ModelSpec(
        "E+L+S",
        "F3_esophagus_liver_spleen/"
        "f3_esophagus_liver_spleen_tongji82_min4cm_4cm_nomargin_seed42",
        "F3_esophagus_liver_spleen/"
        "f3_esophagus_liver_spleen_tongji82_min4cm_4cm_nomargin_seed42_"
        "youyi_external/external_eval_predictions.csv",
    ),
    ModelSpec(
        "E+L+PVS",
        "F3_esophagus_liver_portal/"
        "f3_esophagus_liver_portal_tongji82_min4cm_4cm_nomargin_seed42",
        "F3_esophagus_liver_portal/"
        "f3_esophagus_liver_portal_tongji82_min4cm_4cm_nomargin_seed42_"
        "youyi_external/external_eval_predictions.csv",
    ),
    ModelSpec(
        "E+S+PVS",
        "F3_esophagus_spleen_portal_venous_system/"
        "f3_esophagus_spleen_portal_venous_system_tongji82_min4cm_4cm_"
        "nomargin_seed42",
        "F3_esophagus_spleen_portal_venous_system/"
        "f3_esophagus_spleen_portal_venous_system_tongji82_min4cm_4cm_"
        "nomargin_seed42_youyi_external/external_eval_predictions.csv",
    ),
    ModelSpec(
        "E+L+S+PVS",
        "F3_esophagus_liver_spleen_portal/"
        "f3_esophagus_liver_spleen_portal_tongji82_min4cm_4cm_nomargin_seed42",
        "F3_esophagus_liver_spleen_portal/"
        "f3_esophagus_liver_spleen_portal_tongji82_min4cm_4cm_nomargin_seed42_"
        "youyi_external/external_eval_predictions.csv",
    ),
)

THREE_MODELS = ("F1", "F2a", "F2b")
EIGHT_MODELS = (
    "F2b",
    "E+L",
    "E+S",
    "E+PVS",
    "E+L+S",
    "E+L+PVS",
    "E+S+PVS",
    "E+L+S+PVS",
)
FIGURE_SPECS = (
    FigureSpec("f1_f2a_f2b_internal_validation_roc", "validation", THREE_MODELS),
    FigureSpec("f1_f2a_f2b_external_validation_roc", "external", THREE_MODELS),
    FigureSpec("f2b_multiorgan_internal_validation_roc", "validation", EIGHT_MODELS),
    FigureSpec("f2b_multiorgan_external_validation_roc", "external", EIGHT_MODELS),
)

COLORS = {
    "F1": "#3B7A9E",
    "F2a": "#6BAED6",
    "F2b": "#E15759",
    "E+L": "#4E79A7",
    "E+S": "#F28E2B",
    "E+PVS": "#59A14F",
    "E+L+S": "#B07AA1",
    "E+L+PVS": "#9C755F",
    "E+S+PVS": "#EDC948",
    "E+L+S+PVS": "#5F5F5F",
}
def _parse_epoch_overrides(values: Sequence[str]) -> dict[str, int]:
    valid_names = {item.name for item in MODEL_SPECS}
    parsed: dict[str, int] = {}
    for value in values:
        name, separator, epoch_text = value.partition("=")
        name = name.strip()
        epoch_text = epoch_text.strip()
        if not separator or not name or not epoch_text:
            raise ValueError(f"--epoch must use MODEL=EPOCH: {value!r}")
        if name not in valid_names:
            raise ValueError(f"unknown model in --epoch: {name!r}")
        if name in parsed:
            raise ValueError(f"duplicate --epoch override for {name}")
        try:
            epoch = int(epoch_text)
        except ValueError as error:
            raise ValueError(f"invalid epoch for {name}: {epoch_text!r}") from error
        if str(epoch) != epoch_text or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer for {name}")
        parsed[name] = epoch
    return parsed


def _model_inputs(
    results_root: Path,
    figure_spec: FigureSpec,
    epoch_overrides: Mapping[str, int],
) -> list[ModelInput]:
    by_name = {item.name: item for item in MODEL_SPECS}
    model_values = []
    epoch_values = []
    for name in figure_spec.models:
        spec = by_name[name]
        if figure_spec.analysis_set == "validation":
            path = results_root / spec.validation_run / "validation_predictions.csv"
            if name in epoch_overrides:
                epoch_values.append(f"{name}={epoch_overrides[name]}")
        else:
            path = results_root / spec.external_predictions
        model_values.append(f"{name}={path}")
    try:
        return resolve_model_inputs(
            model_values=model_values,
            metrics_values=[],
            epoch_values=epoch_values,
            analysis_set=figure_spec.analysis_set,
        )
    except FileNotFoundError as error:
        if figure_spec.analysis_set == "validation" and "training_metrics" in str(error):
            raise FileNotFoundError(
                f"{error}. Supply the audited epoch with --epoch MODEL=EPOCH."
            ) from error
        raise


def _roc_curve(
    targets: NDArray[np.int64], scores: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    threshold_indices = np.r_[
        np.where(np.diff(sorted_scores) != 0)[0], sorted_targets.size - 1
    ]
    true_positives = np.cumsum(sorted_targets)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    true_positives = np.r_[0, true_positives].astype(np.float64)
    false_positives = np.r_[0, false_positives].astype(np.float64)
    return false_positives / false_positives[-1], true_positives / true_positives[-1]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
        }
    )


def _plot_figure(
    output_dir: Path,
    figure_spec: FigureSpec,
    model_inputs: Sequence[ModelInput],
    stage: str,
    case_ids: Sequence[str],
    targets: NDArray[np.int64],
    scores: NDArray[np.float64],
    estimates: Sequence[DelongEstimate],
    input_hashes: Mapping[Path, str],
) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    summary_rows: list[Mapping[str, object]] = []
    source_rows: list[Mapping[str, object]] = []

    for model_input, model_scores, estimate in zip(model_inputs, scores, estimates):
        false_positive_rate, true_positive_rate = _roc_curve(targets, model_scores)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=COLORS[model_input.name],
            linestyle="-",
            linewidth=1.8,
            label=f"{model_input.name} (AUC = {estimate.auc:.3f})",
        )
        summary_rows.append(
            {
                "model": model_input.name,
                "analysis_set": figure_spec.analysis_set,
                "stage": stage,
                "selected_epoch": (
                    model_input.epoch if model_input.epoch is not None else ""
                ),
                "epoch_selection": model_input.epoch_selection,
                "n_cases": len(case_ids),
                "n_high_risk": int(targets.sum()),
                "n_low_risk": int(targets.size - targets.sum()),
                "auc": f"{estimate.auc:.10f}",
                "ci_95_lower": f"{estimate.ci_lower:.10f}",
                "ci_95_upper": f"{estimate.ci_upper:.10f}",
                "ci_method": "DeLong normal approximation",
                "positive_class": "target=1 (high risk)",
                "prediction_csv": str(model_input.predictions_path),
                "prediction_sha256": input_hashes[model_input.predictions_path],
                "metrics_sha256": (
                    input_hashes[model_input.metrics_path]
                    if model_input.metrics_path is not None
                    else ""
                ),
            }
        )
        source_rows.extend(
            {
                "model": model_input.name,
                "false_positive_rate": f"{fpr:.10f}",
                "true_positive_rate": f"{tpr:.10f}",
            }
            for fpr, tpr in zip(false_positive_rate, true_positive_rate)
        )

    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.grid(False)
    axis.legend(loc="lower right", borderaxespad=0.8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)

    stem = output_dir / figure_spec.stem
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    _write_csv(output_dir / f"{figure_spec.stem}_auc.csv", summary_rows)
    _write_csv(output_dir / f"{figure_spec.stem}_source_data.csv", source_rows)
    LOGGER.info("Wrote %s.[png|pdf] and companion CSV files", stem)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path.home() / "work" / "results")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epoch", action="append", default=[], metavar="MODEL=EPOCH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate all four ROC comparison figures."""
    arguments = build_parser().parse_args(argv)
    results_root = arguments.results_root.expanduser().resolve()
    output_dir = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else results_root / "roc_f_models"
    )
    if not results_root.is_dir():
        raise FileNotFoundError(f"results root not found: {results_root}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    epoch_overrides = _parse_epoch_overrides(arguments.epoch)
    _configure_style()
    resolved_figures = []
    input_hashes: dict[Path, str] = {}
    for figure_spec in FIGURE_SPECS:
        model_inputs = _model_inputs(results_root, figure_spec, epoch_overrides)
        for path, digest in capture_input_hashes(model_inputs).items():
            if path in input_hashes and input_hashes[path] != digest:
                raise ValueError(f"input changed between figure snapshots: {path}")
            input_hashes[path] = digest
        stage = "validation" if figure_spec.analysis_set == "validation" else "youyi_external"
        case_ids, targets, scores = align_predictions(model_inputs, stage)
        estimates = delong_estimates(targets=targets, scores=scores)
        resolved_figures.append(
            (figure_spec, model_inputs, stage, case_ids, targets, scores, estimates)
        )

    verify_input_hashes(input_hashes)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-"
    ) as temp_name:
        temp_dir = Path(temp_name)
        for resolved in resolved_figures:
            _plot_figure(temp_dir, *resolved, input_hashes)
        verify_input_hashes(input_hashes)
        temp_dir.rename(output_dir)
    LOGGER.info("Completed four ROC figures in %s", output_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    LOGGER.setLevel(logging.INFO)
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(2) from error
