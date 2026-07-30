#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from data.datasets import DATASET_REGISTRY
from data.datasets.classification.esophagus_risk_roi_3d import (
    _MASK_KEYWORDS,
    _NEGATIVE_LABEL_NAMES,
    _POSITIVE_LABEL_NAMES,
    CaseRecord,
    EsophagusRiskRoi3DDataset,
    _contains_keyword,
    _is_nifti_file,
    _normalize_label_name,
    _prefer_shorter_path,
    _split_train_val_records,
)


def _normalize_case_name(name: str) -> str:
    return name.strip().lower()


def _find_matching_dir(parent: Path, name: str) -> Path | None:
    exact_match = parent / name
    if exact_match.is_dir():
        return exact_match

    normalized_name = _normalize_case_name(name)
    for child in sorted(path for path in parent.iterdir() if path.is_dir()):
        if _normalize_case_name(child.name) == normalized_name:
            return child
    return None


def _resolve_single_nifti(case_dir: Path, *, prefer_mask: bool) -> Path:
    nifti_files = sorted(path for path in case_dir.rglob("*") if _is_nifti_file(path))
    if not nifti_files:
        raise ValueError(f"No NIfTI file found in {case_dir}")

    if prefer_mask:
        mask_candidates = [
            path for path in nifti_files if _contains_keyword(path, _MASK_KEYWORDS)
        ]
        if mask_candidates:
            return _prefer_shorter_path(mask_candidates)
    else:
        ct_candidates = [
            path for path in nifti_files if not _contains_keyword(path, _MASK_KEYWORDS)
        ]
        if ct_candidates:
            return _prefer_shorter_path(ct_candidates)

    return _prefer_shorter_path(nifti_files)


def _format_examples(paths: Iterable[str], max_items: int = 5) -> str:
    examples = list(paths)[:max_items]
    return ", ".join(examples)


def _discover_youyi_records(
    root: Path,
    nii_dir_name: str,
    mask_dir_name: str,
) -> list[CaseRecord]:
    if not root.exists():
        raise FileNotFoundError(f"Youyi dataset root does not exist: {root}")

    nii_root = root / nii_dir_name
    mask_root = root / mask_dir_name
    if not nii_root.is_dir():
        raise FileNotFoundError(f"Youyi CT root does not exist: {nii_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Youyi mask root does not exist: {mask_root}")

    records: list[CaseRecord] = []
    missing_mask_cases: list[str] = []
    for nii_class_dir in sorted(path for path in nii_root.iterdir() if path.is_dir()):
        label_name = _normalize_label_name(nii_class_dir.name)
        if label_name in _POSITIVE_LABEL_NAMES:
            label = 1
        elif label_name in _NEGATIVE_LABEL_NAMES:
            label = 0
        else:
            continue

        mask_class_dir = _find_matching_dir(mask_root, nii_class_dir.name)
        if mask_class_dir is None:
            raise FileNotFoundError(
                f"Missing mask class directory for {nii_class_dir.name}: {mask_root}"
            )

        for nii_case_dir in sorted(path for path in nii_class_dir.iterdir() if path.is_dir()):
            mask_case_dir = _find_matching_dir(mask_class_dir, nii_case_dir.name)
            if mask_case_dir is None:
                missing_mask_cases.append(f"{nii_class_dir.name}/{nii_case_dir.name}")
                continue

            records.append(
                CaseRecord(
                    case_id=f"{nii_class_dir.name}/{nii_case_dir.name}",
                    label=label,
                    ct_path=_resolve_single_nifti(nii_case_dir, prefer_mask=False),
                    mask_path=_resolve_single_nifti(mask_case_dir, prefer_mask=True),
                )
            )

    if missing_mask_cases:
        examples = _format_examples(missing_mask_cases)
        raise FileNotFoundError(
            f"Missing Youyi mask case directories for {len(missing_mask_cases)} cases. "
            f"Examples: {examples}"
        )
    if not records:
        raise ValueError(f"No valid Youyi cases found under {root}")
    return records


@DATASET_REGISTRY.register(name="esophagus_risk_youyi_roi_3d", type="classification")
class EsophagusRiskYouyiRoi3DDataset(EsophagusRiskRoi3DDataset):
    """Youyi 3D CT classification dataset with separated CT and mask roots."""

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        self.nii_dir_name = str(
            getattr(opts, "dataset.esophagus_youyi_roi_3d.nii_dir_name", "youyi-nii")
        )
        self.mask_dir_name = str(
            getattr(opts, "dataset.esophagus_youyi_roi_3d.mask_dir_name", "youyi-mask")
        )
        super().__init__(opts=opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--dataset.esophagus-youyi-roi-3d.nii-dir-name",
            type=str,
            default="youyi-nii",
        )
        group.add_argument(
            "--dataset.esophagus-youyi-roi-3d.mask-dir-name",
            type=str,
            default="youyi-mask",
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
            all_records = self._filter_short_esophagus_records(
                _discover_youyi_records(
                    train_root,
                    nii_dir_name=self.nii_dir_name,
                    mask_dir_name=self.mask_dir_name,
                )
            )
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
                return self._filter_short_esophagus_records(
                    _discover_youyi_records(
                        val_root,
                        nii_dir_name=self.nii_dir_name,
                        mask_dir_name=self.mask_dir_name,
                    )
                )

            all_records = self._filter_short_esophagus_records(
                _discover_youyi_records(
                    train_root,
                    nii_dir_name=self.nii_dir_name,
                    mask_dir_name=self.mask_dir_name,
                )
            )
            _, val_records = _split_train_val_records(
                all_records,
                val_ratio=self.val_split_ratio,
                split_seed=self.split_seed,
            )
            self.root = str(train_root)
            return val_records

        effective_test_root = test_root or val_root or train_root
        self.root = str(effective_test_root)
        return self._filter_short_esophagus_records(
            _discover_youyi_records(
                effective_test_root,
                nii_dir_name=self.nii_dir_name,
                mask_dir_name=self.mask_dir_name,
            )
        )
