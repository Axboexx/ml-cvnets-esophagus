import csv
from pathlib import Path

import pytest

import scripts.plot_f_model_roc_curves as roc_plot

PREDICTION_FIELDS = [
    "epoch",
    "stage",
    "case_id",
    "target",
    "positive_probability",
]


def _model_paths() -> dict[str, tuple[Path, Path]]:
    return {
        spec.name: (
            Path(spec.validation_run) / "validation_predictions.csv",
            Path(spec.external_predictions),
        )
        for spec in roc_plot.MODEL_SPECS
    }


def _write_prediction_csv(path: Path, stage: str, epoch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "epoch": epoch,
            "stage": stage,
            "case_id": case_id,
            "target": target,
            "positive_probability": score,
        }
        for case_id, target, score in [
            ("p1", 1, 0.9),
            ("p2", 1, 0.7),
            ("n1", 0, 0.3),
            ("n2", 0, 0.1),
        ]
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_main_inputs(results_root: Path) -> list[str]:
    arguments = ["--results-root", str(results_root)]
    for name, (validation_path, external_path) in _model_paths().items():
        _write_prediction_csv(
            results_root / validation_path, stage="validation", epoch="2"
        )
        _write_prediction_csv(
            results_root / external_path, stage="youyi_external", epoch=""
        )
        arguments.extend(["--epoch", f"{name}=2"])
    return arguments


def _replace_figure_writes_with_placeholders(
    monkeypatch, expected_output_dir: Path | None = None
) -> None:
    def fake_savefig(_figure, path, *args, **kwargs) -> None:
        del args, kwargs
        write_path = Path(path)
        if expected_output_dir is not None:
            assert not expected_output_dir.exists()
            assert write_path.parent.parent == expected_output_dir.parent
            assert write_path.parent.name.startswith(f".{expected_output_dir.name}-")
        write_path.touch()

    monkeypatch.setattr(roc_plot.plt.Figure, "savefig", fake_savefig)


def test_main_publishes_complete_output_contract_atomically(
    monkeypatch, tmp_path
) -> None:
    results_root = tmp_path / "results"
    output_dir = tmp_path / "published"
    arguments = _write_main_inputs(results_root)
    arguments.extend(["--output-dir", str(output_dir)])
    _replace_figure_writes_with_placeholders(monkeypatch, output_dir)

    assert roc_plot.main(arguments) == 0

    expected_names = {
        f"{stem}{suffix}"
        for stem in [
            "f1_f2a_f2b_internal_validation_roc",
            "f1_f2a_f2b_external_validation_roc",
            "f2b_multiorgan_internal_validation_roc",
            "f2b_multiorgan_external_validation_roc",
        ]
        for suffix in [".png", ".pdf", "_auc.csv", "_source_data.csv"]
    }
    assert {path.name for path in output_dir.iterdir()} == expected_names
    assert list(tmp_path.glob(".published-*")) == []

    for stem, expected_models in [
        ("f1_f2a_f2b_external_validation_roc", 3),
        ("f2b_multiorgan_external_validation_roc", 8),
    ]:
        with (output_dir / f"{stem}_auc.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected_models
        assert {row["analysis_set"] for row in rows} == {"external"}
        assert {row["stage"] for row in rows} == {"youyi_external"}
        assert all(len(row["prediction_sha256"]) == 64 for row in rows)


def test_main_delong_preflight_failure_does_not_create_output(
    monkeypatch, tmp_path
) -> None:
    results_root = tmp_path / "results"
    output_dir = tmp_path / "published"
    arguments = _write_main_inputs(results_root)
    arguments.extend(["--output-dir", str(output_dir)])
    for _, external_path_value in _model_paths().values():
        external_path = results_root / external_path_value
        rows = list(csv.DictReader(external_path.open(encoding="utf-8")))
        rows[1]["target"] = "0"
        with external_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    _replace_figure_writes_with_placeholders(monkeypatch)

    with pytest.raises(ValueError, match="at least 2 cases per class"):
        roc_plot.main(arguments)

    assert not output_dir.exists()
    assert list(tmp_path.glob(".published-*")) == []


def test_main_rejects_existing_output_directory(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    output_dir = tmp_path / "published"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output directory already exists"):
        roc_plot.main(
            [
                "--results-root",
                str(results_root),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_main_input_hash_change_does_not_publish(monkeypatch, tmp_path) -> None:
    results_root = tmp_path / "results"
    output_dir = tmp_path / "published"
    arguments = _write_main_inputs(results_root)
    arguments.extend(["--output-dir", str(output_dir)])
    changed_path = results_root / _model_paths()["F1"][0]
    original_verify = roc_plot.verify_input_hashes
    verification_count = 0

    def verify_then_mutate(input_hashes) -> None:
        nonlocal verification_count
        original_verify(input_hashes)
        verification_count += 1
        if verification_count == 1:
            changed_path.write_text(
                changed_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )

    monkeypatch.setattr(roc_plot, "verify_input_hashes", verify_then_mutate)
    _replace_figure_writes_with_placeholders(monkeypatch)

    with pytest.raises(ValueError, match="changed during analysis"):
        roc_plot.main(arguments)

    assert verification_count == 1
    assert not output_dir.exists()
    assert list(tmp_path.glob(".published-*")) == []


def test_main_rejects_inconsistent_digest_for_repeated_input_path(
    monkeypatch, tmp_path
) -> None:
    results_root = tmp_path / "results"
    output_dir = tmp_path / "published"
    arguments = _write_main_inputs(results_root)
    arguments.extend(["--output-dir", str(output_dir)])
    original_capture = roc_plot.capture_input_hashes
    seen_paths: set[Path] = set()
    capture_count = 0

    def capture_with_changed_repeated_digest(model_inputs):
        nonlocal capture_count
        capture_count += 1
        hashes = original_capture(model_inputs)
        for path in hashes:
            if path in seen_paths:
                hashes[path] = "f" * 64
                break
        seen_paths.update(hashes)
        return hashes

    monkeypatch.setattr(
        roc_plot, "capture_input_hashes", capture_with_changed_repeated_digest
    )

    with pytest.raises(ValueError, match="input changed between figure snapshots"):
        roc_plot.main(arguments)

    assert capture_count == 3
    assert not output_dir.exists()
    assert list(tmp_path.glob(".published-*")) == []
