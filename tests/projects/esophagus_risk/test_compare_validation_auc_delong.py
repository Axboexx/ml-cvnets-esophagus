import csv
import json
from pathlib import Path

import pytest

from scripts.compare_validation_auc_delong import (
    ModelInput,
    align_predictions,
    capture_input_hashes,
    holm_adjust,
    main,
    verify_input_hashes,
)
from scripts.statistical_tests.validation_auc import (
    resolve_model_inputs,
    select_best_epoch,
)

PREDICTION_FIELDS = [
    "epoch",
    "stage",
    "case_id",
    "target",
    "positive_probability",
]


def _write_csv(path: Path, fields: list, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_run(
    root: Path,
    name: str,
    scores: list,
    case_order: list = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    _write_csv(
        run_dir / "training_metrics.csv",
        ["epoch", "is_best", "best_metric_source"],
        [
            {"epoch": 1, "is_best": "True", "best_metric_source": "validation"},
            {"epoch": 2, "is_best": "false", "best_metric_source": "validation"},
            {"epoch": 3, "is_best": "1", "best_metric_source": "validation"},
        ],
    )
    targets = {"p1": 1, "p2": 1, "n1": 0, "n2": 0}
    ordered_ids = case_order or list(targets)
    rows = [
        {
            "epoch": 1,
            "stage": "validation",
            "case_id": case_id,
            "target": targets[case_id],
            "positive_probability": 0.5,
        }
        for case_id in ordered_ids
    ]
    rows.extend(
        {
            "epoch": 3,
            "stage": "validation",
            "case_id": case_id,
            "target": targets[case_id],
            "positive_probability": scores[list(targets).index(case_id)],
        }
        for case_id in ordered_ids
    )
    rows.extend(
        {
            "epoch": 3,
            "stage": "validation (EMA)",
            "case_id": case_id,
            "target": targets[case_id],
            "positive_probability": 0.0,
        }
        for case_id in ordered_ids
    )
    _write_csv(run_dir / "validation_predictions.csv", PREDICTION_FIELDS, rows)
    return run_dir


def _write_external_run(root: Path, name: str, scores: list) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    targets = {"p1": 1, "p2": 1, "n1": 0, "n2": 0}
    rows = [
        {
            "epoch": "",
            "stage": "evaluation",
            "case_id": case_id,
            "target": target,
            "positive_probability": scores[index],
        }
        for index, (case_id, target) in enumerate(targets.items())
    ]
    _write_csv(run_dir / "external_eval_predictions.csv", PREDICTION_FIELDS, rows)
    return run_dir


def test_select_best_epoch_uses_last_best_row(tmp_path) -> None:
    run_dir = _write_run(tmp_path, "F1", [0.9, 0.8, 0.2, 0.1])

    assert select_best_epoch(run_dir / "training_metrics.csv") == 3


def test_select_best_epoch_rejects_training_selected_checkpoint(tmp_path) -> None:
    path = tmp_path / "training_metrics.csv"
    _write_csv(
        path,
        ["epoch", "is_best", "best_metric_source"],
        [{"epoch": 1, "is_best": "True", "best_metric_source": "training"}],
    )

    with pytest.raises(ValueError, match="selected from training"):
        select_best_epoch(path)


def test_explicit_missing_metrics_path_is_not_silently_ignored(tmp_path) -> None:
    first = _write_run(tmp_path, "F1", [0.9, 0.8, 0.2, 0.1])
    second = _write_run(tmp_path, "F2a", [0.8, 0.7, 0.3, 0.2])

    with pytest.raises(FileNotFoundError, match="explicit training metrics"):
        resolve_model_inputs(
            model_values=[
                f"F1={first / 'validation_predictions.csv'}",
                f"F2a={second / 'validation_predictions.csv'}",
            ],
            metrics_values=[f"F1={tmp_path / 'missing.csv'}"],
            epoch_values=["F1=3", "F2a=3"],
        )


def test_metrics_change_after_epoch_selection_is_detected(tmp_path) -> None:
    first = _write_run(tmp_path, "F1", [0.9, 0.8, 0.2, 0.1])
    second = _write_run(tmp_path, "F2a", [0.8, 0.7, 0.3, 0.2])
    inputs = resolve_model_inputs(
        model_values=[
            f"F1={first / 'validation_predictions.csv'}",
            f"F2a={second / 'validation_predictions.csv'}",
        ],
        metrics_values=[],
        epoch_values=[],
    )
    hashes = capture_input_hashes(inputs)
    metrics_path = first / "training_metrics.csv"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="changed during analysis"):
        verify_input_hashes(hashes)


def test_align_predictions_reorders_cases_and_excludes_ema(tmp_path) -> None:
    first = _write_run(tmp_path, "F1", [0.9, 0.8, 0.2, 0.1])
    second = _write_run(
        tmp_path,
        "F2a",
        [0.8, 0.7, 0.3, 0.2],
        case_order=["n2", "p2", "n1", "p1"],
    )
    inputs = [
        ModelInput("F1", first / "validation_predictions.csv", None, 3),
        ModelInput("F2a", second / "validation_predictions.csv", None, 3),
    ]

    case_ids, targets, scores = align_predictions(inputs, "validation")

    assert case_ids == ["n1", "n2", "p1", "p2"]
    assert targets.tolist() == [0, 0, 1, 1]
    assert scores[0].tolist() == pytest.approx([0.2, 0.1, 0.9, 0.8])
    assert scores[1].tolist() == pytest.approx([0.3, 0.2, 0.8, 0.7])


def test_align_predictions_rejects_cohort_and_target_mismatch(tmp_path) -> None:
    first = _write_run(tmp_path, "F1", [0.9, 0.8, 0.2, 0.1])
    second = _write_run(tmp_path, "F2a", [0.8, 0.7, 0.3, 0.2])
    prediction_path = second / "validation_predictions.csv"
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8")))
    rows = [row for row in rows if not (row["epoch"] == "3" and row["case_id"] == "n2")]
    _write_csv(prediction_path, PREDICTION_FIELDS, rows)
    inputs = [
        ModelInput("F1", first / "validation_predictions.csv", None, 3),
        ModelInput("F2a", prediction_path, None, 3),
    ]
    with pytest.raises(ValueError, match="cohort mismatch"):
        align_predictions(inputs, "validation")

    _write_run(tmp_path, "F2b", [0.8, 0.7, 0.3, 0.2])
    third_path = tmp_path / "F2b" / "validation_predictions.csv"
    third_rows = list(csv.DictReader(third_path.open(encoding="utf-8")))
    for row in third_rows:
        if row["epoch"] == "3" and row["stage"] == "validation":
            row["target"] = str(1 - int(row["target"]))
            break
    _write_csv(third_path, PREDICTION_FIELDS, third_rows)
    inputs[1] = ModelInput("F2b", third_path, None, 3)
    with pytest.raises(ValueError, match="target mismatch"):
        align_predictions(inputs, "validation")


def test_holm_adjustment() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_cli_writes_three_pairwise_comparisons(tmp_path) -> None:
    run_dirs = {
        "F1": _write_run(tmp_path, "F1", [0.9, 0.8, 0.2, 0.1]),
        "F2a": _write_run(tmp_path, "F2a", [0.85, 0.7, 0.4, 0.3]),
        "F2b": _write_run(tmp_path, "F2b", [0.7, 0.6, 0.5, 0.4]),
    }
    output_dir = tmp_path / "results"
    arguments = []
    for name, run_dir in run_dirs.items():
        arguments.extend(
            ["--model", f"{name}={run_dir / 'validation_predictions.csv'}"]
        )
    arguments.extend(["--output-dir", str(output_dir)])

    assert main(arguments) == 0
    model_rows = list(
        csv.DictReader((output_dir / "model_auc.csv").open(encoding="utf-8"))
    )
    pair_rows = list(
        csv.DictReader((output_dir / "pairwise_delong.csv").open(encoding="utf-8"))
    )
    metadata = json.loads(
        (output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )

    assert [row["model"] for row in model_rows] == ["F1", "F2a", "F2b"]
    assert [(row["model_a"], row["model_b"]) for row in pair_rows] == [
        ("F1", "F2a"),
        ("F1", "F2b"),
        ("F2a", "F2b"),
    ]
    assert metadata["n_cases"] == 4
    assert metadata["interpretation_boundary"].startswith("Internal validation")
    assert metadata["auc_order"] == ["F1", "F2a", "F2b"]
    assert len(metadata["delong_covariance"]) == 3
    assert metadata["models"][0]["epoch_selection"] == "auto_last_is_best"
    assert metadata["implementation"]["python"]
    assert metadata["implementation"]["numpy"]
    assert len(metadata["implementation"]["cli_sha256"]) == 64
    assert len(metadata["implementation"]["delong_sha256"]) == 64


def test_cli_supports_external_predictions_with_blank_epoch(tmp_path) -> None:
    run_dirs = {
        "F1": _write_external_run(tmp_path, "F1", [0.9, 0.8, 0.2, 0.1]),
        "F2a": _write_external_run(tmp_path, "F2a", [0.8, 0.7, 0.3, 0.2]),
        "F2b": _write_external_run(tmp_path, "F2b", [0.7, 0.6, 0.5, 0.4]),
    }
    output_dir = tmp_path / "external_results"
    arguments = ["--analysis-set", "external"]
    for name, run_dir in run_dirs.items():
        arguments.extend(
            ["--model", f"{name}={run_dir / 'external_eval_predictions.csv'}"]
        )
    arguments.extend(["--output-dir", str(output_dir)])

    assert main(arguments) == 0
    metadata = json.loads(
        (output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    readme = (output_dir / "README.md").read_text(encoding="utf-8")

    assert metadata["analysis_set"] == "external"
    assert metadata["stage"] == "evaluation"
    assert metadata["interpretation_boundary"].startswith("External validation")
    assert all(model["epoch"] is None for model in metadata["models"])
    assert all(
        model["epoch_selection"] == "external_fixed_checkpoint"
        for model in metadata["models"]
    )
    assert "# External validation AUC paired DeLong analysis" in readme
    assert "independent generalization evidence" in readme
