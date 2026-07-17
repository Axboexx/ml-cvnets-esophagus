from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
from pathlib import Path
from typing import Sequence

import numpy as np

from projects.esophagus_risk.nifti_utils import read_nifti
from projects.esophagus_risk.roi_height_case_discovery import (
    CaseInput,
    discover_tongji_cases,
    discover_youyi_cases,
)


logger = logging.getLogger(__name__)


_BBOX_MARGIN_MODE = "bbox-margin"
_INFERIOR_HEIGHT_MODE = "inferior-height"
_CASE_FIELDS = (
    "dataset",
    "case_id",
    "label",
    "ct_path",
    "mask_path",
    "status",
    "error",
    "crop_mode",
    "roi_margin_z",
    "ct_depth_slices",
    "slice_spacing_mm",
    "slice_axis_direction",
    "esophagus_min_depth_index",
    "esophagus_max_depth_index",
    "crop_start_index",
    "crop_end_index_exclusive",
    "crop_num_slices",
    "crop_height_mm",
    "crop_height_cm",
    "requested_height_mm",
    "requested_num_slices",
    "boundary_clipped",
)


def _base_row(
    case: CaseInput,
    crop_mode: str,
    requested_height_mm: float,
    roi_margin_z: int,
) -> dict[str, object]:
    return {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "label": "" if case.label is None else case.label,
        "ct_path": "" if case.ct_path is None else str(case.ct_path),
        "mask_path": "" if case.mask_path is None else str(case.mask_path),
        "status": "error",
        "error": case.discovery_error,
        "crop_mode": crop_mode,
        "roi_margin_z": roi_margin_z if crop_mode == _BBOX_MARGIN_MODE else "",
        "requested_height_mm": (
            requested_height_mm if crop_mode == _INFERIOR_HEIGHT_MODE else ""
        ),
    }


def _bbox_margin_depth_bounds(
    depth_indices: np.ndarray,
    volume_depth: int,
    roi_margin_z: int,
) -> tuple[int, int, bool]:
    unbounded_start = int(depth_indices.min()) - roi_margin_z
    unbounded_end = int(depth_indices.max()) + roi_margin_z + 1
    return (
        max(0, unbounded_start),
        min(volume_depth, unbounded_end),
        unbounded_start < 0 or unbounded_end > volume_depth,
    )


def _inferior_height_depth_bounds(
    depth_indices: np.ndarray,
    volume_depth: int,
    height_mm: float,
    slice_spacing_mm: float,
    slice_axis_direction: int,
) -> tuple[int, int, int]:
    requested_num_slices = max(1, math.ceil(height_mm / slice_spacing_mm))
    if slice_axis_direction > 0:
        crop_start = int(depth_indices.min())
        crop_end = min(volume_depth, crop_start + requested_num_slices)
    elif slice_axis_direction < 0:
        crop_end = int(depth_indices.max()) + 1
        crop_start = max(0, crop_end - requested_num_slices)
    else:
        raise ValueError("slice_axis_direction must be -1 or 1")
    return crop_start, crop_end, requested_num_slices


def analyze_case(
    case: CaseInput,
    label_id: int,
    requested_height_mm: float,
    crop_mode: str,
    roi_margin_z: int,
) -> dict[str, object]:
    row = _base_row(case, crop_mode, requested_height_mm, roi_margin_z)
    if case.discovery_error:
        return row
    if case.ct_path is None or case.mask_path is None:
        row["error"] = "CT or mask path is unavailable"
        return row

    try:
        ct_nifti = read_nifti(case.ct_path)
        mask_nifti = read_nifti(case.mask_path)
        if ct_nifti.data.shape != mask_nifti.data.shape:
            raise ValueError(
                f"CT and mask shapes differ for {case.case_id}: "
                f"{ct_nifti.data.shape} != {mask_nifti.data.shape}"
            )
        mask_depth_first = np.transpose(mask_nifti.data, (2, 0, 1))
        depth_indices = np.argwhere(mask_depth_first == label_id)[:, 0]
        if depth_indices.size == 0:
            raise ValueError(f"Esophagus label {label_id} is missing in mask")

        slice_spacing_mm = float(ct_nifti.spacing[2])
        volume_depth = int(ct_nifti.data.shape[2])
        if crop_mode == _BBOX_MARGIN_MODE:
            crop_start, crop_end, boundary_clipped = _bbox_margin_depth_bounds(
                depth_indices, volume_depth, roi_margin_z
            )
            requested_num_slices: int | str = ""
            slice_axis_direction: int | str = getattr(
                ct_nifti, "slice_axis_direction", ""
            )
        else:
            slice_axis_direction = getattr(ct_nifti, "slice_axis_direction", None)
            if slice_axis_direction is None:
                raise ValueError(
                    "inferior-height mode requires the updated nifti_utils.py"
                )
            crop_start, crop_end, requested_num_slices = (
                _inferior_height_depth_bounds(
                    depth_indices,
                    volume_depth,
                    requested_height_mm,
                    slice_spacing_mm,
                    slice_axis_direction,
                )
            )
            boundary_clipped = crop_end - crop_start < requested_num_slices
        crop_num_slices = crop_end - crop_start
        crop_height_mm = crop_num_slices * slice_spacing_mm
        row.update(
            {
                "status": "ok",
                "error": "",
                "ct_depth_slices": volume_depth,
                "slice_spacing_mm": slice_spacing_mm,
                "slice_axis_direction": slice_axis_direction,
                "esophagus_min_depth_index": int(depth_indices.min()),
                "esophagus_max_depth_index": int(depth_indices.max()),
                "crop_start_index": crop_start,
                "crop_end_index_exclusive": crop_end,
                "crop_num_slices": crop_num_slices,
                "crop_height_mm": crop_height_mm,
                "crop_height_cm": crop_height_mm / 10.0,
                "requested_num_slices": requested_num_slices,
                "boundary_clipped": boundary_clipped,
            }
        )
    except Exception as error:
        # Isolate malformed or unsupported files to the current case.
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def _write_case_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    dataset: str,
    rows: Sequence[dict[str, object]],
    crop_mode: str,
    requested_height_mm: float,
    roi_margin_z: int,
) -> dict[str, object]:
    heights = [float(row["crop_height_cm"]) for row in rows if row["status"] == "ok"]
    clipped = sum(row.get("boundary_clipped") is True for row in rows)
    return {
        "dataset": dataset,
        "total_cases": len(rows),
        "valid_cases": len(heights),
        "error_cases": len(rows) - len(heights),
        "boundary_clipped_cases": clipped,
        "crop_mode": crop_mode,
        "roi_margin_z": roi_margin_z if crop_mode == _BBOX_MARGIN_MODE else "",
        "requested_height_cm": (
            requested_height_mm / 10.0
            if crop_mode == _INFERIOR_HEIGHT_MODE
            else ""
        ),
        "mean_height_cm": statistics.fmean(heights) if heights else "",
        "std_height_cm": statistics.pstdev(heights) if heights else "",
        "median_height_cm": statistics.median(heights) if heights else "",
        "min_height_cm": min(heights) if heights else "",
        "max_height_cm": max(heights) if heights else "",
    }


def _write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = tuple(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the pre-resampling physical Z height of each esophagus ROI."
    )
    parser.add_argument("--tongji-root", type=Path, required=True)
    parser.add_argument("--youyi-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-id", type=int, default=15)
    parser.add_argument(
        "--crop-mode",
        choices=(_BBOX_MARGIN_MODE, _INFERIOR_HEIGHT_MODE),
        default=_BBOX_MARGIN_MODE,
    )
    parser.add_argument("--roi-margin-z", type=int, default=16)
    parser.add_argument("--height-mm", type=float, default=40.0)
    parser.add_argument("--youyi-nii-dir-name", default="youyi-nii")
    parser.add_argument("--youyi-mask-dir-name", default="youyi-mask")
    args = parser.parse_args()
    if not math.isfinite(args.height_mm) or args.height_mm <= 0.0:
        parser.error("--height-mm must be a finite positive value")
    if args.roi_margin_z < 0:
        parser.error("--roi-margin-z must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    tongji_root = args.tongji_root.expanduser().resolve()
    youyi_root = args.youyi_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not tongji_root.is_dir():
        raise FileNotFoundError(f"Tongji dataset root does not exist: {tongji_root}")
    if not youyi_root.is_dir():
        raise FileNotFoundError(f"Youyi dataset root does not exist: {youyi_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_by_dataset = {
        "tongji": discover_tongji_cases(tongji_root),
        "youyi": discover_youyi_cases(
            youyi_root,
            nii_dir_name=args.youyi_nii_dir_name,
            mask_dir_name=args.youyi_mask_dir_name,
        ),
    }
    summaries: list[dict[str, object]] = []
    for dataset, cases in cases_by_dataset.items():
        logger.info("Analyzing %d %s cases", len(cases), dataset)
        rows = [
            analyze_case(
                case,
                label_id=args.label_id,
                requested_height_mm=args.height_mm,
                crop_mode=args.crop_mode,
                roi_margin_z=args.roi_margin_z,
            )
            for case in cases
        ]
        _write_case_csv(output_dir / f"{dataset}_esophagus_roi_z_height.csv", rows)
        summaries.append(
            _summary_row(
                dataset,
                rows,
                args.crop_mode,
                args.height_mm,
                args.roi_margin_z,
            )
        )
    _write_summary_csv(output_dir / "esophagus_roi_z_height_summary.csv", summaries)
    logger.info("Wrote ROI height reports to %s", output_dir)


if __name__ == "__main__":
    main()
