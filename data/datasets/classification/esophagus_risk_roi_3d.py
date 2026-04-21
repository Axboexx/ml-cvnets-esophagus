#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from data.datasets import DATASET_REGISTRY
from data.datasets.dataset_base import BaseDataset
from projects.esophagus_risk.nifti_utils import read_nifti


_NEGATIVE_LABEL_NAMES = {
    "nohr",
    "no_hr",
    "no-hr",
    "nonhr",
    "non_hr",
    "non-hr",
    "nhr",
}
_POSITIVE_LABEL_NAMES = {"hr"}
_MASK_KEYWORDS = ("mask", "seg", "label")
_NIFTI_SUFFIXES = (".nii", ".nii.gz")


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    label: int
    ct_path: Path
    mask_path: Path


def _is_nifti_file(path: Path) -> bool:
    return path.is_file() and str(path).lower().endswith(_NIFTI_SUFFIXES)


def _normalize_label_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "-")


def _contains_keyword(path: Path, keywords: Sequence[str]) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    return any(keyword in name or keyword in stem for keyword in keywords)


def _prefer_shorter_path(paths: Iterable[Path]) -> Path:
    return sorted(paths, key=lambda path: (len(path.parts), len(path.name), str(path)))[0]


def _resolve_case_files(case_dir: Path) -> tuple[Path, Path]:
    nifti_files = sorted(path for path in case_dir.rglob("*") if _is_nifti_file(path))
    if len(nifti_files) < 2:
        raise ValueError(f"Expected at least 2 NIfTI files in {case_dir}, got {len(nifti_files)}")

    mask_candidates = [path for path in nifti_files if _contains_keyword(path, _MASK_KEYWORDS)]
    if not mask_candidates:
        raise ValueError(f"Unable to identify mask file in {case_dir}")
    mask_path = _prefer_shorter_path(mask_candidates)

    ct_candidates = [path for path in nifti_files if path != mask_path and not _contains_keyword(path, _MASK_KEYWORDS)]
    if not ct_candidates:
        ct_candidates = [path for path in nifti_files if path != mask_path]
    if not ct_candidates:
        raise ValueError(f"Unable to identify CT file in {case_dir}")
    ct_path = _prefer_shorter_path(ct_candidates)
    return ct_path, mask_path


def _discover_records(root: Path) -> list[CaseRecord]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    records: list[CaseRecord] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label_name = _normalize_label_name(class_dir.name)
        if label_name in _POSITIVE_LABEL_NAMES:
            label = 1
        elif label_name in _NEGATIVE_LABEL_NAMES:
            label = 0
        else:
            continue

        for case_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            ct_path, mask_path = _resolve_case_files(case_dir)
            records.append(
                CaseRecord(
                    case_id=f"{class_dir.name}/{case_dir.name}",
                    label=label,
                    ct_path=ct_path,
                    mask_path=mask_path,
                )
            )
    if not records:
        raise ValueError(f"No valid cases found under {root}")
    return records


def _split_train_val_records(
    records: Sequence[CaseRecord],
    val_ratio: float,
    split_seed: int,
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1). Got: {val_ratio}")

    train_records: list[CaseRecord] = []
    val_records: list[CaseRecord] = []
    grouped: dict[int, list[CaseRecord]] = {0: [], 1: []}
    for record in records:
        grouped[record.label].append(record)

    rng = random.Random(split_seed)
    for label, label_records in grouped.items():
        ordered = sorted(label_records, key=lambda item: item.case_id)
        rng.shuffle(ordered)

        n_val = int(round(len(ordered) * val_ratio))
        if len(ordered) > 1:
            n_val = max(1, min(len(ordered) - 1, n_val))
        val_slice = ordered[:n_val]
        train_slice = ordered[n_val:]
        if not train_slice and val_slice:
            train_slice = [val_slice.pop()]

        train_records.extend(train_slice)
        val_records.extend(val_slice)

    return sorted(train_records, key=lambda item: item.case_id), sorted(
        val_records, key=lambda item: item.case_id
    )


def _to_depth_first(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    return np.transpose(volume, (2, 0, 1))


def _expand_bounds(start: int, end: int, max_size: int, margin: int) -> tuple[int, int]:
    return max(0, start - margin), min(max_size, end + margin + 1)


@DATASET_REGISTRY.register(name="esophagus_risk_roi_3d", type="classification")
class EsophagusRiskRoi3DDataset(BaseDataset):
    """Direct 3D CT classification dataset with online esophagus ROI cropping."""

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)

        self.label_id = int(getattr(opts, "dataset.esophagus_roi_3d.label_id"))
        self.input_depth = int(getattr(opts, "dataset.esophagus_roi_3d.input_depth"))
        self.input_height = int(getattr(opts, "dataset.esophagus_roi_3d.input_height"))
        self.input_width = int(getattr(opts, "dataset.esophagus_roi_3d.input_width"))
        self.roi_margin_xy = int(getattr(opts, "dataset.esophagus_roi_3d.roi_margin_xy"))
        self.roi_margin_z = int(getattr(opts, "dataset.esophagus_roi_3d.roi_margin_z"))
        self.val_split_ratio = float(getattr(opts, "dataset.esophagus_roi_3d.val_split_ratio"))
        self.split_seed = int(getattr(opts, "dataset.esophagus_roi_3d.split_seed"))
        self.use_windowing = bool(getattr(opts, "dataset.esophagus_roi_3d.use_windowing"))
        self.window_center = float(getattr(opts, "dataset.esophagus_roi_3d.window_center"))
        self.window_width = float(getattr(opts, "dataset.esophagus_roi_3d.window_width"))
        self.clip_percentile_lower = float(
            getattr(opts, "dataset.esophagus_roi_3d.clip_percentile_lower")
        )
        self.clip_percentile_upper = float(
            getattr(opts, "dataset.esophagus_roi_3d.clip_percentile_upper")
        )

        setattr(opts, "model.classification.n_classes", 2)
        setattr(opts, "dataset.collate_fn_name_train", "image_classification_data_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "image_classification_data_collate_fn")
        setattr(opts, "dataset.collate_fn_name_test", "image_classification_data_collate_fn")

        self.records = self._load_records()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != EsophagusRiskRoi3DDataset:
            return parser

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument("--dataset.esophagus-roi-3d.label-id", type=int, default=15)
        group.add_argument("--dataset.esophagus-roi-3d.input-depth", type=int, default=96)
        group.add_argument("--dataset.esophagus-roi-3d.input-height", type=int, default=96)
        group.add_argument("--dataset.esophagus-roi-3d.input-width", type=int, default=96)
        group.add_argument("--dataset.esophagus-roi-3d.roi-margin-xy", type=int, default=24)
        group.add_argument("--dataset.esophagus-roi-3d.roi-margin-z", type=int, default=8)
        group.add_argument("--dataset.esophagus-roi-3d.val-split-ratio", type=float, default=0.2)
        group.add_argument("--dataset.esophagus-roi-3d.split-seed", type=int, default=42)
        group.add_argument(
            "--dataset.esophagus-roi-3d.use-windowing",
            action="store_true",
            default=True,
        )
        group.add_argument(
            "--dataset.esophagus-roi-3d.no-windowing",
            action="store_false",
            dest="dataset.esophagus_roi_3d.use_windowing",
        )
        group.add_argument("--dataset.esophagus-roi-3d.window-center", type=float, default=50.0)
        group.add_argument("--dataset.esophagus-roi-3d.window-width", type=float, default=350.0)
        group.add_argument(
            "--dataset.esophagus-roi-3d.clip-percentile-lower",
            type=float,
            default=0.5,
        )
        group.add_argument(
            "--dataset.esophagus-roi-3d.clip-percentile-upper",
            type=float,
            default=99.5,
        )
        return parser

    def _load_records(self) -> list[CaseRecord]:
        train_root = Path(getattr(self.opts, "dataset.root_train")).expanduser()
        val_root = (
            Path(getattr(self.opts, "dataset.root_val")).expanduser()
            if getattr(self.opts, "dataset.root_val")
            else None
        )
        test_root = (
            Path(getattr(self.opts, "dataset.root_test")).expanduser()
            if getattr(self.opts, "dataset.root_test")
            else None
        )

        if self.mode == "train":
            all_records = _discover_records(train_root)
            train_records, _ = _split_train_val_records(
                all_records,
                val_ratio=self.val_split_ratio,
                split_seed=self.split_seed,
            )
            self.root = str(train_root)
            return train_records

        if self.mode == "val":
            if val_root and val_root != train_root:
                self.root = str(val_root)
                return _discover_records(val_root)

            all_records = _discover_records(train_root)
            _, val_records = _split_train_val_records(
                all_records,
                val_ratio=self.val_split_ratio,
                split_seed=self.split_seed,
            )
            self.root = str(train_root)
            return val_records

        effective_test_root = test_root or val_root or train_root
        self.root = str(effective_test_root)
        return _discover_records(effective_test_root)

    def _parse_index(self, sample_size_and_index: Any) -> int:
        if isinstance(sample_size_and_index, (tuple, list)):
            return int(sample_size_and_index[-1])
        return int(sample_size_and_index)

    def _extract_roi(self, ct_volume: np.ndarray, mask_volume: np.ndarray) -> np.ndarray:
        esophagus_mask = mask_volume == self.label_id
        coords = np.argwhere(esophagus_mask)
        if coords.size == 0:
            raise ValueError(f"Esophagus label {self.label_id} is missing in mask")

        d_min, h_min, w_min = coords.min(axis=0).tolist()
        d_max, h_max, w_max = coords.max(axis=0).tolist()

        d_start, d_end = _expand_bounds(d_min, d_max, ct_volume.shape[0], self.roi_margin_z)
        h_start, h_end = _expand_bounds(h_min, h_max, ct_volume.shape[1], self.roi_margin_xy)
        w_start, w_end = _expand_bounds(w_min, w_max, ct_volume.shape[2], self.roi_margin_xy)
        return ct_volume[d_start:d_end, h_start:h_end, w_start:w_end]

    def _normalize_roi(self, roi: np.ndarray) -> np.ndarray:
        if self.use_windowing:
            lower = self.window_center - self.window_width / 2.0
            upper = self.window_center + self.window_width / 2.0
        else:
            lower = float(np.percentile(roi, self.clip_percentile_lower))
            upper = float(np.percentile(roi, self.clip_percentile_upper))

        if upper <= lower:
            upper = lower + 1.0

        roi = np.clip(roi, lower, upper)
        roi = (roi - lower) / (upper - lower)
        roi = (roi * 2.0) - 1.0
        return roi.astype(np.float32, copy=False)

    def _resize_roi(self, roi: np.ndarray) -> torch.Tensor:
        roi_tensor = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0)
        roi_tensor = F.interpolate(
            roi_tensor,
            size=(self.input_depth, self.input_height, self.input_width),
            mode="trilinear",
            align_corners=False,
        )
        return roi_tensor.squeeze(0)

    def __getitem__(self, sample_size_and_index: Any) -> Dict[str, Any]:
        sample_index = self._parse_index(sample_size_and_index)
        record = self.records[sample_index]

        ct_volume = _to_depth_first(read_nifti(record.ct_path).data.astype(np.float32, copy=False))
        mask_volume = _to_depth_first(read_nifti(record.mask_path).data.astype(np.int32, copy=False))

        roi = self._extract_roi(ct_volume=ct_volume, mask_volume=mask_volume)
        roi = self._normalize_roi(roi=roi)
        sample_tensor = self._resize_roi(roi=roi)

        return {
            "samples": sample_tensor,
            "targets": record.label,
            "sample_id": sample_index,
        }

    def __len__(self) -> int:
        return len(self.records)

    def get_item_metadata(self, item_idx: int) -> Dict:
        return {}

    def share_dataset_arguments(self) -> Dict[str, Any]:
        return {"model.classification.n_classes": 2}

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"\n\tnum_cases={len(self.records)}"
            + f"\n\tlabel_id={self.label_id}"
            + f"\n\troi_size=({self.input_depth}, {self.input_height}, {self.input_width})"
        )
