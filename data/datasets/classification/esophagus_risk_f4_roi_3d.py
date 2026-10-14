#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

"""F4 clinical-laboratory fusion datasets for the 4 cm protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from data.datasets import DATASET_REGISTRY
from data.datasets.classification.esophagus_risk_roi_3d import (
    CaseRecord,
    EsophagusRiskRoi3DDataset,
)
from data.datasets.classification.esophagus_risk_youyi_roi_3d import (
    EsophagusRiskYouyiRoi3DDataset,
)
from projects.esophagus_risk.clinical_data import (
    CLINICAL_FEATURE_NAMES,
    ClinicalRecord,
    case_id_from_record,
    load_clinical_records,
    validate_clinical_matches,
)


class _F4ClinicalDatasetMixin:
    clinical_records: dict[str, ClinicalRecord]

    def _initialize_clinical_data(self) -> None:
        workbook_path = str(
            getattr(self.opts, "dataset.esophagus_f4.clinical_workbook", "")
        ).strip()
        if not workbook_path:
            raise ValueError(
                "F4 requires --dataset.esophagus-f4.clinical-workbook."
            )
        self.clinical_workbook = Path(workbook_path).expanduser()
        self.clinical_records = load_clinical_records(self.clinical_workbook)
        validate_clinical_matches(
            [(record.case_id, record.label) for record in self.records],
            self.clinical_records,
        )
        setattr(self.opts, "dataset.collate_fn_name_train", "pytorch_default_collate_fn")
        setattr(self.opts, "dataset.collate_fn_name_val", "pytorch_default_collate_fn")
        setattr(self.opts, "dataset.collate_fn_name_test", "pytorch_default_collate_fn")

    def _clinical_features(self, record: CaseRecord) -> torch.Tensor:
        normalized_id = case_id_from_record(record.case_id)
        return torch.tensor(
            self.clinical_records[normalized_id].features,
            dtype=torch.float32,
        )

    def __getitem__(self, sample_size_and_index: Any) -> Dict[str, Any]:
        item = super().__getitem__(sample_size_and_index)
        sample_index = self._parse_index(sample_size_and_index)
        image = item["samples"]
        item["samples"] = {
            "image": image,
            "clinical": self._clinical_features(self.records[sample_index]),
        }
        return item

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"\n\tclinical_workbook={self.clinical_workbook}"
            + f"\n\tclinical_features={len(CLINICAL_FEATURE_NAMES)}"
        )


@DATASET_REGISTRY.register(name="esophagus_risk_f4_roi_3d", type="classification")
class EsophagusRiskF4Roi3DDataset(
    _F4ClinicalDatasetMixin,
    EsophagusRiskRoi3DDataset,
):
    """Tongji F4 dataset using the new F-series 4 cm ROI protocol."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self._initialize_clinical_data()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != EsophagusRiskF4Roi3DDataset:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--dataset.esophagus-f4.clinical-workbook",
            type=str,
            default="",
            help="Path to the F4 clinical XLSX workbook for the active centre.",
        )
        return parser


@DATASET_REGISTRY.register(
    name="esophagus_risk_youyi_f4_roi_3d",
    type="classification",
)
class EsophagusRiskYouyiF4Roi3DDataset(
    _F4ClinicalDatasetMixin,
    EsophagusRiskYouyiRoi3DDataset,
):
    """Youyi external F4 dataset with separated CT and mask roots."""

    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool = True,
        is_evaluation: bool = False,
        *args,
        **kwargs,
    ) -> None:
        if is_training or not is_evaluation:
            raise ValueError(
                "esophagus_risk_youyi_f4_roi_3d is external-test only. "
                "Use esophagus_risk_f4_roi_3d for training and validation."
            )
        super().__init__(
            opts=opts,
            is_training=is_training,
            is_evaluation=is_evaluation,
            *args,
            **kwargs,
        )
        self._initialize_clinical_data()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser
