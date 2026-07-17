from __future__ import annotations

import csv
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from analyze_esophagus_roi_height import CaseInput, analyze_case, main


def _write_nifti(
    path: Path,
    data: np.ndarray,
    slice_spacing_mm: float,
    slice_axis_direction: int,
) -> None:
    header = bytearray(352)
    struct.pack_into("<I", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *data.shape, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)
    struct.pack_into(
        "<8f",
        header,
        76,
        1.0,
        1.0,
        1.0,
        slice_spacing_mm,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<4f", header, 280, 1.0, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, 1.0, 0.0, 0.0)
    struct.pack_into(
        "<4f",
        header,
        312,
        0.0,
        0.0,
        slice_axis_direction * slice_spacing_mm,
        0.0,
    )
    path.write_bytes(header + data.astype(np.float32).tobytes(order="F"))


def _case(tmp_path: Path, mask_depth_indices: list[int], direction: int) -> CaseInput:
    shape = (2, 3, 40)
    ct = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    mask[:, :, mask_depth_indices] = 15
    ct_path = tmp_path / "ct.nii"
    mask_path = tmp_path / "mask.nii"
    _write_nifti(ct_path, ct, slice_spacing_mm=3.0, slice_axis_direction=direction)
    _write_nifti(mask_path, mask, slice_spacing_mm=3.0, slice_axis_direction=direction)
    return CaseInput("tongji", "HR/case-001", 1, ct_path, mask_path)


def _write_case_pair(ct_path: Path, mask_path: Path) -> None:
    shape = (2, 3, 40)
    ct = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    mask[:, :, 10:12] = 15
    ct_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    _write_nifti(ct_path, ct, slice_spacing_mm=3.0, slice_axis_direction=1)
    _write_nifti(mask_path, mask, slice_spacing_mm=3.0, slice_axis_direction=1)


@pytest.mark.parametrize(
    ("direction", "indices", "expected_bounds"),
    [
        (1, [10, 11], (10, 27)),
        (-1, [29, 30], (14, 31)),
    ],
)
def test_analyze_case_matches_current_crop_bounds(
    tmp_path: Path,
    direction: int,
    indices: list[int],
    expected_bounds: tuple[int, int],
) -> None:
    row = analyze_case(
        _case(tmp_path, indices, direction),
        label_id=15,
        requested_height_mm=50.0,
        crop_mode="inferior-height",
        roi_margin_z=16,
    )

    assert row["status"] == "ok"
    assert (row["crop_start_index"], row["crop_end_index_exclusive"]) == expected_bounds
    assert row["crop_num_slices"] == 17
    assert row["crop_height_mm"] == pytest.approx(51.0)
    assert row["crop_height_cm"] == pytest.approx(5.1)
    assert row["boundary_clipped"] is False


def test_analyze_case_reports_boundary_clipping(tmp_path: Path) -> None:
    row = analyze_case(
        _case(tmp_path, [38, 39], direction=1),
        label_id=15,
        requested_height_mm=50.0,
        crop_mode="inferior-height",
        roi_margin_z=16,
    )

    assert row["status"] == "ok"
    assert row["crop_num_slices"] == 2
    assert row["crop_height_cm"] == pytest.approx(0.6)
    assert row["boundary_clipped"] is True


def test_analyze_case_records_missing_label_as_error(tmp_path: Path) -> None:
    row = analyze_case(
        _case(tmp_path, [10], direction=1),
        label_id=99,
        requested_height_mm=50.0,
        crop_mode="bbox-margin",
        roi_margin_z=16,
    )

    assert row["status"] == "error"
    assert "label 99 is missing" in str(row["error"])


def test_analyze_case_matches_old_bbox_margin_crop(tmp_path: Path) -> None:
    row = analyze_case(
        _case(tmp_path, [10, 11], direction=1),
        label_id=15,
        requested_height_mm=50.0,
        crop_mode="bbox-margin",
        roi_margin_z=16,
    )

    assert row["status"] == "ok"
    assert (row["crop_start_index"], row["crop_end_index_exclusive"]) == (0, 28)
    assert row["crop_num_slices"] == 28
    assert row["crop_height_cm"] == pytest.approx(8.4)
    assert row["requested_height_mm"] == ""
    assert row["boundary_clipped"] is True


def test_main_writes_separate_dataset_csvs_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tongji_root = tmp_path / "tongji"
    youyi_root = tmp_path / "youyi"
    output_dir = tmp_path / "output"
    _write_case_pair(
        tongji_root / "HR" / "case-t" / "ct.nii",
        tongji_root / "HR" / "case-t" / "mask.nii",
    )
    _write_case_pair(
        youyi_root / "youyi-nii" / "noHR" / "case-y" / "ct.nii",
        youyi_root / "youyi-mask" / "noHR" / "case-y" / "mask.nii",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_esophagus_roi_height.py",
            "--tongji-root",
            str(tongji_root),
            "--youyi-root",
            str(youyi_root),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    tongji_csv = output_dir / "tongji_esophagus_roi_z_height.csv"
    youyi_csv = output_dir / "youyi_esophagus_roi_z_height.csv"
    summary_csv = output_dir / "esophagus_roi_z_height_summary.csv"
    assert tongji_csv.is_file()
    assert youyi_csv.is_file()
    assert summary_csv.is_file()
    with tongji_csv.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["case_id"] == "HR/case-t"
    assert row["crop_mode"] == "bbox-margin"
    assert float(row["crop_height_cm"]) == pytest.approx(8.4)
    with summary_csv.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert [row["dataset"] for row in summary_rows] == ["tongji", "youyi"]
