#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

"""F5 esophagus-liver and clinical datasets for the 4 cm protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from data.datasets import DATASET_REGISTRY
from data.datasets.classification.esophagus_risk_e3a_multi_organ_roi_3d import (
    EsophagusRiskE3aMultiOrganRoi3DDataset,
)
from data.datasets.classification.esophagus_risk_roi_3d import CaseRecord
from data.datasets.classification.esophagus_risk_youyi_e3a_multi_organ_roi_3d import (
    EsophagusRiskYouyiE3aMultiOrganRoi3DDataset,
)
from projects.esophagus_risk.clinical_data import (
    CLINICAL_FEATURE_NAMES,
    ClinicalRecord,
    case_id_from_record,
    load_clinical_records,
    validate_clinical_matches,
)


class _F5ClinicalDatasetMixin:
    clinical_records: dict[str, ClinicalRecord]

    def _initialize_clinical_data(self) -> None:
        expected_specs = [
            ("esophagus", 15, 0, 0, True),
            ("liver", 5, 0, 0, True),
        ]
        actual_specs = [
            (spec.name, spec.label_id, spec.margin_xy, spec.margin_z, spec.required)
            for spec in self.organ_specs
        ]
        if actual_specs != expected_specs:
            raise ValueError(
                "F5 requires ordered, zero-margin, required organ specs "
                f"{expected_specs}; got {actual_specs}."
            )
        workbook_path = str(
            getattr(self.opts, "dataset.esophagus_f5.clinical_workbook", "")
        ).strip()
        if not workbook_path:
            raise ValueError(
                "F5 requires --dataset.esophagus-f5.clinical-workbook."
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
        item["samples"] = {
            "image": item["samples"],
            "clinical": self._clinical_features(self.records[sample_index]),
        }
        return item

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"\n\tclinical_workbook={self.clinical_workbook}"
            + f"\n\tclinical_features={len(CLINICAL_FEATURE_NAMES)}"
        )


@DATASET_REGISTRY.register(
    name="esophagus_risk_f5_clinical_roi_3d",
    type="classification",
)
class EsophagusRiskF5ClinicalRoi3DDataset(
    _F5ClinicalDatasetMixin,
    EsophagusRiskE3aMultiOrganRoi3DDataset,
):
    """Tongji F5 dataset with esophagus, liver, and clinical predictors."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self._initialize_clinical_data()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != EsophagusRiskF5ClinicalRoi3DDataset:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--dataset.esophagus-f5.clinical-workbook",
            type=str,
            default="",
            help="Path to the F5 clinical XLSX workbook for the active centre.",
        )
        return parser


@DATASET_REGISTRY.register(
    name="esophagus_risk_youyi_f5_clinical_roi_3d",
    type="classification",
)
class EsophagusRiskYouyiF5ClinicalRoi3DDataset(
    _F5ClinicalDatasetMixin,
    EsophagusRiskYouyiE3aMultiOrganRoi3DDataset,
):
    """Youyi external-only F5 dataset."""

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
                "esophagus_risk_youyi_f5_clinical_roi_3d is external-test only. "
                "Use esophagus_risk_f5_clinical_roi_3d for training and validation."
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
