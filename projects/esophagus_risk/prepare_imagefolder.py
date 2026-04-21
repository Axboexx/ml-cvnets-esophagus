from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

PROJECTS_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from projects.esophagus_risk.nifti_utils import read_nifti


CLASS_NAMES = {0: "nHR", 1: "HR"}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    risk_label: int
    label_name: str
    ct_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export esophagus risk NIfTI volumes into ImageFolder PNG datasets for ml-cvnets."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to standardized manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output ImageFolder root")
    parser.add_argument(
        "--mode",
        type=str,
        default="roi_crop",
        choices=("roi_crop", "gt_roi", "whole_ct"),
        help="PNG export mode. roi_crop/gt_roi uses Label-15 ROI crops. whole_ct exports full axial slices.",
    )
    parser.add_argument("--esophagus-label", type=int, default=15, help="Esophagus label id in multi-class mask")
    parser.add_argument("--num-slices-per-case", type=int, default=5, help="Number of axial PNGs exported per case")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation case ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for case-level split")
    parser.add_argument("--image-size", type=int, default=256, help="Output square PNG size")
    parser.add_argument("--roi-margin", type=int, default=12, help="Pixel margin around the projected bbox")
    parser.add_argument("--window-min", type=float, default=-150.0, help="CT window lower bound")
    parser.add_argument("--window-max", type=float, default=250.0, help="CT window upper bound")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> list[CaseRecord]:
    project_root = manifest_path.parent.parent.parent
    cases: list[CaseRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "ok":
                continue
            risk_label = int(row["risk_label"])
            label_name = row["label"]
            cases.append(
                CaseRecord(
                    case_id=row["case_id"],
                    risk_label=risk_label,
                    label_name=label_name,
                    ct_path=(project_root / row["ct_path"]).resolve(),
                    mask_path=(project_root / row["mask_path"]).resolve(),
                )
            )
    if not cases:
        raise ValueError(f"No valid cases found in manifest: {manifest_path}")
    return cases


def normalize_mode(mode: str) -> str:
    if mode == "gt_roi":
        return "roi_crop"
    return mode


def split_cases(records: Sequence[CaseRecord], val_ratio: float, seed: int) -> dict[str, list[CaseRecord]]:
    grouped: dict[int, list[CaseRecord]] = {0: [], 1: []}
    for record in records:
        grouped[record.risk_label].append(record)

    rng = random.Random(seed)
    train_records: list[CaseRecord] = []
    val_records: list[CaseRecord] = []

    for risk_label, items in grouped.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_ratio))
        if len(shuffled) > 1:
            val_count = max(1, min(len(shuffled) - 1, val_count))
        else:
            val_count = 0
        val_records.extend(shuffled[:val_count])
        train_records.extend(shuffled[val_count:])

    return {"train": sorted(train_records, key=lambda x: x.case_id), "val": sorted(val_records, key=lambda x: x.case_id)}


def project_bbox(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot compute bbox from an empty mask.")

    x_min = max(int(coords[:, 0].min()) - margin, 0)
    x_max = min(int(coords[:, 0].max()) + margin, mask.shape[0] - 1)
    y_min = max(int(coords[:, 1].min()) - margin, 0)
    y_max = min(int(coords[:, 1].max()) + margin, mask.shape[1] - 1)
    return x_min, x_max, y_min, y_max


def select_gt_roi_slices(mask: np.ndarray, num_slices: int) -> list[int]:
    slice_area = mask.sum(axis=(0, 1))
    non_zero_indices = np.flatnonzero(slice_area > 0)
    if non_zero_indices.size == 0:
        return []
    ranked = sorted(non_zero_indices.tolist(), key=lambda idx: int(slice_area[idx]), reverse=True)
    selected = sorted(ranked[:num_slices])
    return selected


def select_whole_ct_slices(depth: int, num_slices: int) -> list[int]:
    if num_slices <= 1:
        return [depth // 2]
    start = int(depth * 0.2)
    end = int(depth * 0.8)
    if end <= start:
        start, end = 0, max(depth - 1, 0)
    indices = np.linspace(start, end, num=num_slices, dtype=int)
    return sorted(set(int(index) for index in indices.tolist()))


def window_ct(slice_2d: np.ndarray, window_min: float, window_max: float) -> np.ndarray:
    clipped = np.clip(slice_2d.astype(np.float32), window_min, window_max)
    normalized = (clipped - window_min) / max(window_max - window_min, 1e-6)
    uint8_image = np.round(normalized * 255.0).astype(np.uint8)
    return uint8_image


def to_rgb_image(slice_2d: np.ndarray, image_size: int) -> Image.Image:
    rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((image_size, image_size), resample=Image.BICUBIC)


def export_case(
    record: CaseRecord,
    split_name: str,
    output_dir: Path,
    mode: str,
    esophagus_label: int,
    num_slices_per_case: int,
    image_size: int,
    roi_margin: int,
    window_min: float,
    window_max: float,
) -> list[dict[str, str | int]]:
    mode = normalize_mode(mode)
    ct_volume = read_nifti(record.ct_path).data
    mask_volume = read_nifti(record.mask_path).data
    if ct_volume.shape != mask_volume.shape:
        raise ValueError(f"Shape mismatch for case {record.case_id}: {ct_volume.shape} vs {mask_volume.shape}")

    if mode == "roi_crop":
        esophagus_mask = mask_volume == esophagus_label
        selected_slices = select_gt_roi_slices(esophagus_mask, num_slices_per_case)
        if not selected_slices:
            raise ValueError(f"Case {record.case_id} has no esophagus voxels for label {esophagus_label}.")
        x_min, x_max, y_min, y_max = project_bbox(esophagus_mask, roi_margin)
    else:
        selected_slices = select_whole_ct_slices(ct_volume.shape[2], num_slices_per_case)
        x_min, x_max, y_min, y_max = 0, ct_volume.shape[0] - 1, 0, ct_volume.shape[1] - 1

    target_dir = output_dir / split_name / record.label_name
    target_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for slice_index in selected_slices:
        ct_slice = ct_volume[x_min : x_max + 1, y_min : y_max + 1, slice_index]
        image_uint8 = window_ct(ct_slice, window_min=window_min, window_max=window_max)
        image = to_rgb_image(image_uint8, image_size=image_size)
        file_name = f"{record.case_id}_z{slice_index:03d}.png"
        image_path = target_dir / file_name
        image.save(image_path)
        rows.append(
            {
                "split": split_name,
                "label_name": record.label_name,
                "risk_label": record.risk_label,
                "case_id": record.case_id,
                "slice_index": slice_index,
                "mode": mode,
                "png_path": str(image_path.relative_to(output_dir)),
                "bbox": f"{x_min}:{x_max};{y_min}:{y_max}",
            }
        )
    return rows


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(output_dir: Path, rows: Iterable[dict[str, str | int]]) -> None:
    rows = list(rows)
    csv_path = output_dir / "export_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["split", "label_name", "risk_label", "case_id", "slice_index", "mode", "png_path", "bbox"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    output_dir: Path,
    manifest_path: Path,
    split_records: dict[str, list[CaseRecord]],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256sum(manifest_path),
        "mode": normalize_mode(args.mode),
        "esophagus_label": args.esophagus_label,
        "num_slices_per_case": args.num_slices_per_case,
        "image_size": args.image_size,
        "roi_margin": args.roi_margin,
        "window_min": args.window_min,
        "window_max": args.window_max,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "splits": {
            split: [record.case_id for record in records] for split, records in split_records.items()
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.mode = normalize_mode(args.mode)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(args.manifest.resolve())
    split_records = split_cases(records, val_ratio=args.val_ratio, seed=args.seed)

    exported_rows: list[dict[str, str | int]] = []
    for split_name, split_cases_list in split_records.items():
        for record in split_cases_list:
            exported_rows.extend(
                export_case(
                    record=record,
                    split_name=split_name,
                    output_dir=output_dir,
                    mode=args.mode,
                    esophagus_label=args.esophagus_label,
                    num_slices_per_case=args.num_slices_per_case,
                    image_size=args.image_size,
                    roi_margin=args.roi_margin,
                    window_min=args.window_min,
                    window_max=args.window_max,
                )
            )

    write_rows(output_dir, exported_rows)
    write_metadata(output_dir, args.manifest.resolve(), split_records=split_records, args=args)
    print(f"Exported {len(exported_rows)} PNGs to {output_dir}")


if __name__ == "__main__":
    main()
