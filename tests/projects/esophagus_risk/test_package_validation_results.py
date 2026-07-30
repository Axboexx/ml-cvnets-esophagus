import zipfile
from pathlib import Path

import pytest

from scripts.package_validation_results import (
    _write_archive_entry,
    collect_archive_entries,
    create_archive,
    main,
)


def _make_input_directories(results_root: Path) -> list[Path]:
    directories = [
        results_root / "F1",
        results_root / "F2a",
        results_root / "F2b",
        results_root / "delong_results",
    ]
    for directory in directories:
        directory.mkdir(parents=True)
    return directories


def test_create_archive_preserves_structure_and_excludes_checkpoints(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    f1_dir, f2a_dir, f2b_dir, delong_dir = _make_input_directories(results_root)
    f1_run = f1_dir / "f1_run"
    f1_other_run = f1_dir / "f1_other_run"
    f2a_run = f2a_dir / "f2a_run"
    f2b_run = f2b_dir / "f2b_run"
    for run_directory in [f1_run, f1_other_run, f2a_run, f2b_run]:
        run_directory.mkdir()
    (f1_run / "validation_predictions.csv").write_text(
        "prediction",
        encoding="utf-8",
    )
    (f1_run / "checkpoint_best.pt").write_text("weights", encoding="utf-8")
    (f1_other_run / "config.yaml").write_text("other config", encoding="utf-8")
    (f2a_run / "CHECKPOINT_last.PT").write_text("weights", encoding="utf-8")
    (f2a_run / "config.yaml").write_text("config", encoding="utf-8")
    (f2b_run / "training_checkpoint_summary.csv").write_text(
        "checkpoint",
        encoding="utf-8",
    )
    nested = f2b_run / "reports"
    nested.mkdir()
    (nested / "summary.json").write_text("{}", encoding="utf-8")
    (delong_dir / "pairwise_delong.csv").write_text("p_value", encoding="utf-8")
    output_path = results_root / "validation_results.zip"

    included_count, excluded_count = create_archive(
        results_root,
        [f1_dir, f2a_dir, f2b_dir, delong_dir],
        output_path,
    )

    assert included_count == 5
    assert excluded_count == 3
    with zipfile.ZipFile(output_path) as archive:
        assert archive.namelist() == [
            "F1/f1_other_run/config.yaml",
            "F1/f1_run/validation_predictions.csv",
            "F2a/f2a_run/config.yaml",
            "F2b/f2b_run/reports/summary.json",
            "delong_results/pairwise_delong.csv",
        ]


def test_create_archive_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    directories = _make_input_directories(results_root)
    (directories[0] / "config.yaml").write_text("config", encoding="utf-8")
    output_path = results_root / "existing.zip"
    output_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        create_archive(results_root, directories, output_path)

    assert output_path.read_bytes() == b"existing"


def test_create_archive_rejects_output_inside_an_input_directory(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    directories = _make_input_directories(results_root)
    (directories[0] / "config.yaml").write_text("config", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be inside an input"):
        create_archive(
            results_root,
            directories,
            directories[0] / "bundle.zip",
        )


def test_create_archive_rejects_missing_input_without_creating_output(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    directories = _make_input_directories(results_root)
    output_path = results_root / "bundle.zip"

    with pytest.raises(NotADirectoryError, match="input directory not found"):
        create_archive(
            results_root,
            [*directories[:3], results_root / "missing_delong"],
            output_path,
        )

    assert not output_path.exists()


def test_collect_archive_entries_rejects_directory_outside_root(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    directories = _make_input_directories(results_root)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="must be inside results root"):
        collect_archive_entries(results_root, [*directories[:3], outside])


def test_collect_archive_entries_rejects_duplicate_or_nested_inputs(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    directories = _make_input_directories(results_root)

    with pytest.raises(ValueError, match="must be distinct"):
        collect_archive_entries(results_root, [directories[0], directories[0]])

    with pytest.raises(ValueError, match="must not contain one another"):
        collect_archive_entries(
            results_root,
            [directories[0].parent, directories[0]],
        )


def test_collect_archive_entries_rejects_symbolic_links(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    directories = _make_input_directories(results_root)
    external_file = tmp_path / "external.txt"
    external_file.write_text("external", encoding="utf-8")
    (directories[0] / "linked.txt").symlink_to(external_file)

    with pytest.raises(ValueError, match="symbolic links"):
        collect_archive_entries(results_root, directories)


def test_collect_archive_entries_rejects_symbolic_link_input_root(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    directories = _make_input_directories(results_root)
    linked_input = results_root / "linked_f1"
    linked_input.symlink_to(directories[0], target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic links"):
        collect_archive_entries(
            results_root,
            [linked_input, *directories[1:]],
        )


def test_archive_write_rejects_file_replaced_after_collection(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    directories = _make_input_directories(results_root)
    source = directories[0] / "config.yaml"
    source.write_text("config", encoding="utf-8")
    entries, _ = collect_archive_entries(results_root, directories)
    external_file = tmp_path / "external.txt"
    external_file.write_text("external", encoding="utf-8")
    source.unlink()
    source.symlink_to(external_file)

    with zipfile.ZipFile(
        tmp_path / "bundle.zip",
        mode="w",
    ) as archive, pytest.raises((OSError, ValueError)):
        _write_archive_entry(archive, entries[0])


def test_cli_expands_home_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    results_root = home / "work" / "results"
    directories = _make_input_directories(results_root)
    for directory in directories:
        (directory / "result.csv").write_text("result", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    assert (
        main(
            [
                "--f1-dir",
                "$HOME/work/results/F1",
                "--f2a-dir",
                "$HOME/work/results/F2a",
                "--f2b-dir",
                "$HOME/work/results/F2b",
                "--delong-dir",
                "$HOME/work/results/delong_results",
                "--output",
                "$HOME/work/results/bundle.zip",
            ]
        )
        == 0
    )
    assert (results_root / "bundle.zip").is_file()
