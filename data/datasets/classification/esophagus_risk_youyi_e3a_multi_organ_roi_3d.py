#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

import argparse
from pathlib import Path

from data.datasets import DATASET_REGISTRY
from data.datasets.classification.esophagus_risk_e3a_multi_organ_roi_3d import (
    EsophagusRiskE3aMultiOrganRoi3DDataset,
)
from data.datasets.classification.esophagus_risk_roi_3d import (
    CaseRecord,
    _split_train_val_records,
)
from data.datasets.classification.esophagus_risk_youyi_roi_3d import _discover_youyi_records


@DATASET_REGISTRY.register(
    name="esophagus_risk_youyi_e3a_multi_organ_roi_3d",
    type="classification",
)
class EsophagusRiskYouyiE3aMultiOrganRoi3DDataset(EsophagusRiskE3aMultiOrganRoi3DDataset):
    """Youyi E3a multi-organ dataset with separated CT and mask roots."""

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
