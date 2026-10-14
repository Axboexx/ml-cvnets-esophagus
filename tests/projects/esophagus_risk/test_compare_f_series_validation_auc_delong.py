"""Tests for the complete F-series paired validation AUC comparison."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pytest

from scripts.compare_f_series_validation_auc_delong import (
    MODEL_NAMES,
    main,
    parse_model_directories,
    resolve_run_directory,
    select_auc_best_epoch,
)
from scripts import compare_f_series_validation_auc_delong
from scripts.statistical_tests.delong import delong_auc_covariance


def _write_run(parent: Path, name: str, model_index: int) -> Path:
    """Write two epochs of paired case-level predictions and AUC records."""
    run_directory = parent / name / "selected_run"
    run_directory.mkdir(parents=True)
    targets = [index % 2 for index in range(20)]
    scores = [
        min(0.99, max(0.01, 0.24 + 0.28 * target +
                       0.06 * ((index * (model_index + 3)) % 9)))
        for index, target in enumerate(targets)
    ]
    auc = float(delong_auc_covariance(targets, [scores])[0][0])

    with (run_directory / "validation_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "stage", "case_id", "target", "positive_probability"],
        )
        writer.writeheader()
        for epoch, epoch_scores in [(0, [0.5] * 20), (1, scores)]:
            for index, (target, score) in enumerate(zip(targets, epoch_scores)):
                writer.writerow(
                    {
                        "epoch": epoch,
                        "stage": "validation",
                        "case_id": f"case_{index:03d}",
                        "target": target,
                        "positive_probability": score,
                    }
                )

    metrics = [
        {"epoch": 0, "stage": "validation", "auc": 0.5},
        {"epoch": 1, "stage": "validation", "auc": auc},
    ]
    (run_directory / "validation_confusion_matrix.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8"
    )
    return run_directory


def _write_all_runs(root: Path) -> Dict[str, Path]:
    return {
        name: _write_run(root, name, index)
        for index, name in enumerate(MODEL_NAMES)
    }


def test_select_auc_best_epoch_uses_recorded_auc_and_earliest_tie(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation_confusion_matrix.jsonl"
    rows = [
        {"epoch": 5, "stage": "validation", "auc": 0.8},
        {"epoch": 3, "stage": "validation", "auc": 0.8},
        {"epoch": 9, "stage": "evaluation", "auc": 0.9},
        {"epoch": 7, "stage": "validation", "auc": 0.7},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    epoch, auc, digest = select_auc_best_epoch(path)

    assert (epoch, auc) == (3, 0.8)
    assert len(digest) == 64


def test_model_directory_requires_specific_run(tmp_path: Path) -> None:
    runs = _write_all_runs(tmp_path)
    model_args = [f"{name}={run}" for name, run in runs.items()]
    assert list(parse_model_directories(model_args)) == list(MODEL_NAMES)
    assert resolve_run_directory("F1", runs["F1"]) == runs["F1"]

    with pytest.raises(ValueError, match="missing --model-dir"):
        parse_model_directories(model_args[:-1])
    with pytest.raises(ValueError, match="duplicate --model-dir"):
        parse_model_directories(model_args + [model_args[0]])

    with pytest.raises(FileNotFoundError, match="specific training run"):
        resolve_run_directory("F1", runs["F1"].parent)


def test_cli_writes_12_auc_rows_and_66_pairs(tmp_path: Path) -> None:
    runs = _write_all_runs(tmp_path)
    arguments: List[str] = []
    for name, run in runs.items():
        arguments.extend(["--model-dir", f"{name}={run}"])
    output_dir = tmp_path / "delong_output"
    arguments.extend(["--output-dir", str(output_dir)])

    assert main(arguments) == 0
    with (output_dir / "model_auc.csv").open(encoding="utf-8") as handle:
        model_rows = list(csv.DictReader(handle))
    with (output_dir / "pairwise_delong.csv").open(encoding="utf-8") as handle:
        pair_rows = list(csv.DictReader(handle))
    metadata = json.loads(
        (output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )

    assert [row["model"] for row in model_rows] == list(MODEL_NAMES)
    assert len(pair_rows) == 66
    assert metadata["n_cases"] == 20
    assert metadata["implementation"]["cli_sha256"] == hashlib.sha256(
        Path(compare_f_series_validation_auc_delong.__file__).read_bytes()
    ).hexdigest()
    assert all(row["epoch"] == "1" for row in model_rows)
    assert all(
        row["epoch_selection"] == "max_recorded_validation_auc_earliest_tie"
        for row in model_rows
    )
    assert all(row["p_value_holm"] for row in pair_rows)


def test_cli_rejects_changed_case_cohort(tmp_path: Path) -> None:
    runs = _write_all_runs(tmp_path)
    path = runs["F5"] / "validation_predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row for row in rows
        if not (row["epoch"] == "1" and row["case_id"] == "case_019")
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    arguments: List[str] = []
    for name, run in runs.items():
        arguments.extend(["--model-dir", f"{name}={run}"])
    arguments.extend(["--output-dir", str(tmp_path / "output")])

    with pytest.raises(SystemExit, match="cohort mismatch"):
        main(arguments)
    assert not (tmp_path / "output").exists()
