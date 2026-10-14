"""Tests for the complete F-series paired external AUC comparison."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import compare_f_series_external_auc_delong
from scripts.compare_f_series_external_auc_delong import (
    EXTERNAL_STAGE,
    MODEL_NAMES,
    main,
    parse_model_directories,
    resolve_external_run_directory,
)


def _write_external_run(parent: Path, name: str, model_index: int) -> Path:
    """Write one paired external prediction set for a model."""
    run_directory = parent / name / "selected_external_run"
    run_directory.mkdir(parents=True)
    targets = [index % 2 for index in range(20)]
    scores = [
        min(
            0.99,
            max(
                0.01,
                0.24 + 0.28 * target + 0.06 * ((index * (model_index + 3)) % 9),
            ),
        )
        for index, target in enumerate(targets)
    ]

    with (run_directory / "external_eval_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "stage", "case_id", "target", "positive_probability"],
        )
        writer.writeheader()
        for index, (target, score) in enumerate(zip(targets, scores)):
            writer.writerow(
                {
                    "epoch": "",
                    "stage": EXTERNAL_STAGE,
                    "case_id": f"case_{index:03d}",
                    "target": target,
                    "positive_probability": score,
                }
            )
    return run_directory


def _write_all_runs(root: Path) -> dict[str, Path]:
    return {
        name: _write_external_run(root, name, index)
        for index, name in enumerate(MODEL_NAMES)
    }


def test_model_directory_requires_specific_external_run(tmp_path: Path) -> None:
    runs = _write_all_runs(tmp_path)
    model_args = [f"{name}={run}" for name, run in runs.items()]

    assert list(parse_model_directories(model_args)) == list(MODEL_NAMES)
    assert resolve_external_run_directory("F1", runs["F1"]) == runs["F1"]

    with pytest.raises(ValueError, match="missing --model-dir"):
        parse_model_directories(model_args[:-1])
    with pytest.raises(ValueError, match="duplicate --model-dir"):
        parse_model_directories(model_args + [model_args[0]])
    with pytest.raises(FileNotFoundError, match="specific external-evaluation run"):
        resolve_external_run_directory("F1", runs["F1"].parent)


def test_cli_writes_12_auc_rows_and_66_external_pairs(tmp_path: Path) -> None:
    runs = _write_all_runs(tmp_path)
    arguments: list[str] = []
    for name, run in runs.items():
        arguments.extend(["--model-dir", f"{name}={run}"])
    output_dir = tmp_path / "delong_external_output"
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
    assert metadata["analysis_set"] == "external"
    assert metadata["stage"] == EXTERNAL_STAGE
    assert metadata["n_cases"] == 20
    assert metadata["n_positive"] == 10
    assert metadata["n_negative"] == 10
    assert (
        metadata["implementation"]["cli_sha256"]
        == hashlib.sha256(
            Path(compare_f_series_external_auc_delong.__file__).read_bytes()
        ).hexdigest()
    )
    assert all(row["epoch"] == "" for row in model_rows)
    assert all(
        row["epoch_selection"] == "external_fixed_checkpoint" for row in model_rows
    )
    assert all(row["p_value_holm"] for row in pair_rows)


def test_cli_rejects_changed_external_cohort(tmp_path: Path) -> None:
    runs = _write_all_runs(tmp_path)
    path = runs["F5"] / "external_eval_predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case_id"] != "case_019"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    arguments: list[str] = []
    for name, run in runs.items():
        arguments.extend(["--model-dir", f"{name}={run}"])
    arguments.extend(["--output-dir", str(tmp_path / "output")])

    with pytest.raises(SystemExit, match="cohort mismatch"):
        main(arguments)
    assert not (tmp_path / "output").exists()
