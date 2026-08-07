from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.export_tongji_split import build_split_entries, write_split_csv


def _make_case(
    root: Path,
    label_name: str,
    case_name: str,
) -> Path:
    case_dir = root / label_name / case_name
    case_dir.mkdir(parents=True)
    (case_dir / "ct.nii.gz").write_bytes(b"")
    mask_path = case_dir / "mask.nii.gz"
    mask_path.write_bytes(b"")
    return mask_path


def test_export_matches_filtered_stratified_split(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    mask_heights: dict[Path, int] = {}
    for label_name in ("nhr", "hr"):
        for case_index in range(5):
            mask_path = _make_case(
                root=tmp_path,
                label_name=label_name,
                case_name=f"case_{case_index:02d}",
            )
            mask_heights[mask_path] = (
                10 if (label_name, case_index) == ("nhr", 0) else 20
            )

    def read_nifti(path: Path) -> SimpleNamespace:
        mask = np.zeros((4, 4, 24), dtype=np.int16)
        mask[1:3, 1:3, 2 : 2 + mask_heights[path]] = 15
        return SimpleNamespace(data=mask, spacing=(1.0, 1.0, 2.0))

    monkeypatch.setattr(
        "data.datasets.classification.esophagus_risk_roi_3d.read_nifti",
        read_nifti,
    )

    entries = build_split_entries(
        dataset_root=tmp_path,
        val_ratio=0.2,
        split_seed=42,
        esophagus_label_id=15,
        min_esophagus_height_mm=40.0,
    )

    assert sum(entry.split == "train" for entry in entries) == 7
    assert sum(entry.split == "validation" for entry in entries) == 2
    assert sum(entry.split == "excluded_height" for entry in entries) == 1
    assert {entry.record.label for entry in entries if entry.split == "validation"} == {
        0,
        1,
    }

    output_path = tmp_path / "split.csv"
    write_split_csv(entries=entries, output_path=output_path)
    with output_path.open(encoding="utf-8-sig", newline="") as output_file:
        rows = list(csv.DictReader(output_file))

    assert len(rows) == 10
    assert {row["split"] for row in rows} == {
        "train",
        "validation",
        "excluded_height",
    }


def test_export_script_supports_direct_execution() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "export_tongji_split.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset-root" in result.stdout
