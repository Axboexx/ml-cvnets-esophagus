#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from data.datasets import DATASET_REGISTRY
from data.datasets.classification.esophagus_risk_roi_3d import (
    CaseRecord,
    _discover_records,
    _expand_bounds,
    _filter_records_by_esophagus_height,
    _split_train_val_records,
    _to_depth_first,
)
from data.datasets.dataset_base import BaseDataset
from projects.esophagus_risk.nifti_utils import read_nifti, validate_spatial_alignment
from projects.esophagus_risk.roi_utils import inferior_to_superior_depth_bounds
from utils import logger


@dataclass(frozen=True)
class OrganSpec:
    name: str
    label_id: int
    margin_xy: int
    margin_z: int
    required: bool


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in _parse_csv(value)]


def _broadcast(values: Sequence[int], target_len: int, field_name: str) -> list[int]:
    if len(values) == target_len:
        return list(values)
    if len(values) == 1:
        return [int(values[0])] * target_len
    raise ValueError(
        f"{field_name} must contain either 1 or {target_len} values, got {len(values)}"
    )


def _build_organ_specs(opts: argparse.Namespace) -> list[OrganSpec]:
    prefix = "dataset.e3a_multi_organ_roi_3d"
    organ_names = _parse_csv(getattr(opts, f"{prefix}.organ_names"))
    label_ids = _parse_int_csv(getattr(opts, f"{prefix}.label_ids"))
    margin_xy = _broadcast(
        _parse_int_csv(getattr(opts, f"{prefix}.roi_margin_xy")),
        target_len=len(organ_names),
        field_name="roi_margin_xy",
    )
    margin_z = _broadcast(
        _parse_int_csv(getattr(opts, f"{prefix}.roi_margin_z")),
        target_len=len(organ_names),
        field_name="roi_margin_z",
    )
    required_names = set(_parse_csv(getattr(opts, f"{prefix}.required_organs")))

    if len(organ_names) != len(label_ids):
        raise ValueError(
            "organ_names and label_ids must have the same length. "
            f"Got {len(organ_names)} and {len(label_ids)}."
        )
    if not organ_names:
        raise ValueError("At least one organ must be configured for E3a.")

    specs: list[OrganSpec] = []
    for name, label_id, organ_margin_xy, organ_margin_z in zip(
        organ_names,
        label_ids,
        margin_xy,
        margin_z,
    ):
        specs.append(
            OrganSpec(
                name=name,
                label_id=label_id,
                margin_xy=organ_margin_xy,
                margin_z=organ_margin_z,
                required=name in required_names,
            )
        )
    return specs


@DATASET_REGISTRY.register(name="esophagus_risk_e3a_multi_organ_roi_3d", type="classification")
class EsophagusRiskE3aMultiOrganRoi3DDataset(BaseDataset):
    """E3a multi-organ 3D CT dataset with organ-specific CT and mask crops."""

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)

        self.organ_specs = _build_organ_specs(opts)
        self.input_depth = int(getattr(opts, "dataset.e3a_multi_organ_roi_3d.input_depth"))
        self.input_height = int(getattr(opts, "dataset.e3a_multi_organ_roi_3d.input_height"))
        self.input_width = int(getattr(opts, "dataset.e3a_multi_organ_roi_3d.input_width"))
        self.esophagus_inferior_to_superior_height_mm = float(
            getattr(
                opts,
                "dataset.e3a_multi_organ_roi_3d.esophagus_inferior_to_superior_height_mm",
                40.0,
            )
        )
        self.min_esophagus_height_mm = float(
            getattr(
                opts,
                "dataset.e3a_multi_organ_roi_3d.min_esophagus_height_mm",
                40.0,
            )
        )
        self.val_split_ratio = float(
            getattr(opts, "dataset.e3a_multi_organ_roi_3d.val_split_ratio")
        )
        self.split_seed = int(getattr(opts, "dataset.e3a_multi_organ_roi_3d.split_seed"))
        self.use_windowing = bool(getattr(opts, "dataset.e3a_multi_organ_roi_3d.use_windowing"))
        self.window_center = float(getattr(opts, "dataset.e3a_multi_organ_roi_3d.window_center"))
        self.window_width = float(getattr(opts, "dataset.e3a_multi_organ_roi_3d.window_width"))
        self.clip_percentile_lower = float(
            getattr(opts, "dataset.e3a_multi_organ_roi_3d.clip_percentile_lower")
        )
        self.clip_percentile_upper = float(
            getattr(opts, "dataset.e3a_multi_organ_roi_3d.clip_percentile_upper")
        )

        setattr(opts, "model.classification.n_classes", 2)
        setattr(opts, "dataset.collate_fn_name_train", "image_classification_data_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "image_classification_data_collate_fn")
        setattr(opts, "dataset.collate_fn_name_test", "image_classification_data_collate_fn")

        self.num_cases_before_height_filter = 0
        self.num_cases_excluded_by_height = 0
        self.records = self._load_records()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != EsophagusRiskE3aMultiOrganRoi3DDataset:
            return parser

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.organ-names",
            type=str,
            default="esophagus,liver,spleen,portal_venous_system",
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.label-ids",
            type=str,
            default="15,5,1,64",
        )
        group.add_argument("--dataset.e3a-multi-organ-roi-3d.input-depth", type=int, default=128)
        group.add_argument("--dataset.e3a-multi-organ-roi-3d.input-height", type=int, default=160)
        group.add_argument("--dataset.e3a-multi-organ-roi-3d.input-width", type=int, default=160)
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.roi-margin-xy",
            type=str,
            default="0,0,0,0",
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.roi-margin-z",
            type=str,
            default="0,0,0,0",
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.esophagus-inferior-to-superior-height-mm",
            type=float,
            default=40.0,
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.min-esophagus-height-mm",
            type=float,
            default=40.0,
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.required-organs",
            type=str,
            default="esophagus",
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.val-split-ratio",
            type=float,
            default=0.2,
        )
        group.add_argument("--dataset.e3a-multi-organ-roi-3d.split-seed", type=int, default=42)
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.use-windowing",
            action="store_true",
            default=True,
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.no-windowing",
            action="store_false",
            dest="dataset.e3a_multi_organ_roi_3d.use_windowing",
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.window-center",
            type=float,
            default=50.0,
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.window-width",
            type=float,
            default=350.0,
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.clip-percentile-lower",
            type=float,
            default=0.5,
        )
        group.add_argument(
            "--dataset.e3a-multi-organ-roi-3d.clip-percentile-upper",
            type=float,
            default=99.5,
        )
        return parser

    def _filter_short_esophagus_records(
        self,
        records: Sequence[CaseRecord],
    ) -> list[CaseRecord]:
        esophagus_spec = next(
            (spec for spec in self.organ_specs if spec.name.lower() == "esophagus"),
            None,
        )
        if esophagus_spec is None:
            raise ValueError("E3a requires an esophagus organ specification for height filtering")

        included, excluded = _filter_records_by_esophagus_height(
            records=records,
            label_id=esophagus_spec.label_id,
            min_height_mm=self.min_esophagus_height_mm,
        )
        self.num_cases_before_height_filter = len(records)
        self.num_cases_excluded_by_height = len(excluded)
        if excluded:
            examples = ", ".join(
                f"{record.case_id} ({height_mm:.2f} mm)"
                for record, height_mm in excluded[:5]
            )
            logger.log(
                f"Excluded {len(excluded)} of {len(records)} E3a cases with esophagus "
                f"height < {self.min_esophagus_height_mm:.2f} mm. Examples: {examples}"
            )
        if not included:
            raise ValueError(
                "No E3a cases remain after applying the minimum esophagus height filter "
                f"({self.min_esophagus_height_mm:.2f} mm)."
            )
        return included

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
        resolved_train_root = train_root.resolve()

        def is_distinct_root(candidate: Path | None) -> bool:
            return candidate is not None and candidate.resolve() != resolved_train_root

        if self.mode == "train":
            all_records = self._filter_short_esophagus_records(
                _discover_records(train_root)
            )
            if self.val_split_ratio == 0.0:
                self.root = str(train_root)
                return sorted(all_records, key=lambda item: item.case_id)
            train_records, _ = _split_train_val_records(
                all_records,
                val_ratio=self.val_split_ratio,
                split_seed=self.split_seed,
            )
            self.root = str(train_root)
            return train_records

        if self.mode == "val":
            if is_distinct_root(val_root):
                self.root = str(val_root)
                return self._filter_short_esophagus_records(_discover_records(val_root))

            all_records = self._filter_short_esophagus_records(
                _discover_records(train_root)
            )
            if self.val_split_ratio == 0.0:
                self.root = str(train_root)
                return []
            _, val_records = _split_train_val_records(
                all_records,
                val_ratio=self.val_split_ratio,
                split_seed=self.split_seed,
            )
            self.root = str(train_root)
            return val_records

        if is_distinct_root(test_root):
            self.root = str(test_root)
            return self._filter_short_esophagus_records(_discover_records(test_root))
        if is_distinct_root(val_root):
            self.root = str(val_root)
            return self._filter_short_esophagus_records(_discover_records(val_root))

        all_records = self._filter_short_esophagus_records(_discover_records(train_root))
        _, test_records = _split_train_val_records(
            all_records,
            val_ratio=self.val_split_ratio,
            split_seed=self.split_seed,
        )
        self.root = str(train_root)
        return test_records

    def _parse_index(self, sample_size_and_index: Any) -> int:
        if isinstance(sample_size_and_index, (tuple, list)):
            return int(sample_size_and_index[-1])
        return int(sample_size_and_index)

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

    def _resize_ct_and_mask(self, ct_roi: np.ndarray, mask_roi: np.ndarray) -> torch.Tensor:
        ct_tensor = torch.from_numpy(ct_roi).unsqueeze(0).unsqueeze(0)
        ct_tensor = F.interpolate(
            ct_tensor,
            size=(self.input_depth, self.input_height, self.input_width),
            mode="trilinear",
            align_corners=False,
        )

        mask_tensor = torch.from_numpy(
            mask_roi.astype(np.float32, copy=False)
        ).unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(self.input_depth, self.input_height, self.input_width),
            mode="nearest",
        )
        return torch.cat([ct_tensor, mask_tensor], dim=1).squeeze(0)

    def _zero_organ_tensor(self) -> torch.Tensor:
        return torch.zeros(
            2,
            self.input_depth,
            self.input_height,
            self.input_width,
            dtype=torch.float32,
        )

    def _extract_organ_tensor(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        spec: OrganSpec,
        slice_spacing_mm: float,
        slice_axis_direction: int,
    ) -> torch.Tensor:
        organ_mask = mask_volume == spec.label_id
        coords = np.argwhere(organ_mask)
        if coords.size == 0:
            if spec.required:
                raise ValueError(f"Required organ {spec.name} label {spec.label_id} is missing")
            return self._zero_organ_tensor()

        d_min, h_min, w_min = coords.min(axis=0).tolist()
        d_max, h_max, w_max = coords.max(axis=0).tolist()
        if spec.name.lower() == "esophagus" and self.esophagus_inferior_to_superior_height_mm > 0.0:
            d_start, d_end = inferior_to_superior_depth_bounds(
                mask_depth_indices=coords[:, 0],
                volume_depth=ct_volume.shape[0],
                height_mm=self.esophagus_inferior_to_superior_height_mm,
                slice_spacing_mm=slice_spacing_mm,
                slice_axis_direction=slice_axis_direction,
            )
            segment_coords = np.argwhere(organ_mask[d_start:d_end])
            h_min, w_min = segment_coords[:, 1:].min(axis=0).tolist()
            h_max, w_max = segment_coords[:, 1:].max(axis=0).tolist()
            h_start, h_end = h_min, h_max + 1
            w_start, w_end = w_min, w_max + 1
        else:
            d_start, d_end = _expand_bounds(
                d_min,
                d_max,
                ct_volume.shape[0],
                spec.margin_z,
            )
            h_start, h_end = _expand_bounds(
                h_min,
                h_max,
                ct_volume.shape[1],
                spec.margin_xy,
            )
            w_start, w_end = _expand_bounds(
                w_min,
                w_max,
                ct_volume.shape[2],
                spec.margin_xy,
            )

        ct_roi = ct_volume[d_start:d_end, h_start:h_end, w_start:w_end]
        mask_roi = organ_mask[d_start:d_end, h_start:h_end, w_start:w_end]
        ct_roi = self._normalize_roi(roi=ct_roi)
        return self._resize_ct_and_mask(ct_roi=ct_roi, mask_roi=mask_roi)

    def __getitem__(self, sample_size_and_index: Any) -> Dict[str, Any]:
        sample_index = self._parse_index(sample_size_and_index)
        record = self.records[sample_index]

        ct_nifti = read_nifti(record.ct_path)
        mask_nifti = read_nifti(record.mask_path)
        validate_spatial_alignment(ct_nifti, mask_nifti, record.case_id)
        ct_volume = _to_depth_first(ct_nifti.data.astype(np.float32, copy=False))
        mask_volume = _to_depth_first(mask_nifti.data.astype(np.int32, copy=False))

        organ_tensors = [
            self._extract_organ_tensor(
                ct_volume=ct_volume,
                mask_volume=mask_volume,
                spec=spec,
                slice_spacing_mm=ct_nifti.spacing[2],
                slice_axis_direction=ct_nifti.slice_axis_direction,
            )
            for spec in self.organ_specs
        ]
        sample_tensor = torch.stack(organ_tensors, dim=0)

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
        organ_repr = ", ".join(f"{spec.name}:{spec.label_id}" for spec in self.organ_specs)
        return (
            super().extra_repr()
            + f"\n\tnum_cases={len(self.records)}"
            + f"\n\torgans={organ_repr}"
            + "\n\tesophagus_inferior_to_superior_height_mm="
            + f"{self.esophagus_inferior_to_superior_height_mm}"
            + f"\n\tmin_esophagus_height_mm={self.min_esophagus_height_mm}"
            + f"\n\tnum_cases_excluded_by_height={self.num_cases_excluded_by_height}"
            + f"\n\troi_size=({self.input_depth}, {self.input_height}, {self.input_width})"
        )
