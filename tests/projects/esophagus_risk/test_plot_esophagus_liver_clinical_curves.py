import csv
from pathlib import Path

import numpy as np
import pytest

import scripts.plot_esophagus_liver_clinical_curves as clinical_plot

PREDICTION_FIELDS = [
    "epoch",
    "stage",
    "case_id",
    "target",
    "positive_probability",
]


def _write_predictions(path: Path, stage: str, epoch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "epoch": epoch,
            "stage": stage,
            "case_id": f"case-{index}",
            "target": target,
            "positive_probability": score,
        }
        for index, (target, score) in enumerate(
            [(0, 0.05), (0, 0.20), (0, 0.35), (1, 0.65), (1, 0.80), (1, 0.95)]
        )
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics(path: Path, best_epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "is_best", "best_metric_source"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "epoch": 1,
                    "is_best": "True",
                    "best_metric_source": "validation",
                },
                {
                    "epoch": best_epoch,
                    "is_best": "True",
                    "best_metric_source": "validation",
                },
            ]
        )


def test_calibration_bins_use_observed_rate_and_wilson_interval() -> None:
    targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)

    bins = clinical_plot._calibration_bins(targets, scores, n_bins=2)

    assert len(bins) == 2
    assert bins[0].count == 2
    assert bins[0].mean_predicted == pytest.approx(0.15)
    assert bins[0].observed_proportion == pytest.approx(0.0)
    assert bins[1].mean_predicted == pytest.approx(0.85)
    assert bins[1].observed_proportion == pytest.approx(1.0)
    assert bins[0].ci_lower == pytest.approx(0.0)
    assert bins[0].ci_upper == pytest.approx(0.6576280471)
    assert bins[1].ci_lower == pytest.approx(0.3423719529)
    assert bins[1].ci_upper == pytest.approx(1.0)


def test_calibration_bins_put_quantile_boundary_in_left_bin() -> None:
    targets = np.asarray([0, 1, 1, 0], dtype=np.int64)
    scores = np.asarray([0.1, 0.2, 0.2, 0.3], dtype=np.float64)

    bins = clinical_plot._calibration_bins(targets, scores, n_bins=2)

    assert [bin_.count for bin_ in bins] == [3, 1]
    assert bins[0].mean_predicted == pytest.approx(1.0 / 6.0)
    assert bins[0].observed_proportion == pytest.approx(2.0 / 3.0)


def test_calibration_bins_reject_too_few_unique_probabilities() -> None:
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    scores = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

    with pytest.raises(ValueError, match="at least two unique"):
        clinical_plot._calibration_bins(targets, scores, n_bins=2)


def test_decision_curve_matches_opt_in_net_benefit_formula() -> None:
    targets = np.asarray([1, 1, 0, 0], dtype=np.int64)
    scores = np.asarray([0.9, 0.4, 0.6, 0.1], dtype=np.float64)
    thresholds = np.asarray([0.25, 0.4, 0.75], dtype=np.float64)

    curve = clinical_plot._decision_curve(targets, scores, thresholds)

    assert curve.model == pytest.approx([5.0 / 12.0, 1.0 / 3.0, 0.25])
    assert curve.treat_all == pytest.approx([1.0 / 3.0, 1.0 / 6.0, -1.0])
    assert curve.treat_none == pytest.approx([0.0, 0.0, 0.0])


def test_main_creates_only_four_png_pdf_figure_pairs(tmp_path) -> None:
    results_root = tmp_path / "results"
    internal = results_root / clinical_plot.INTERNAL_RUN
    external = results_root / clinical_plot.EXTERNAL_RUN
    _write_predictions(internal / "validation_predictions.csv", "validation", "25")
    _write_metrics(internal / "training_metrics.csv", best_epoch=25)
    _write_predictions(external / "external_eval_predictions.csv", "youyi_external", "")
    output_dir = tmp_path / "clinical-curves"

    assert (
        clinical_plot.main(
            [
                "--results-root",
                str(results_root),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert {path.name for path in output_dir.iterdir()} == {
        f"{stem}{suffix}"
        for stem in clinical_plot.FIGURE_STEMS
        for suffix in (".png", ".pdf")
    }
    assert not list(output_dir.glob("*.csv"))


def test_main_rejects_existing_output_directory(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    output_dir = tmp_path / "clinical-curves"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output directory already exists"):
        clinical_plot.main(
            [
                "--results-root",
                str(results_root),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert marker.read_text(encoding="utf-8") == "unchanged"
