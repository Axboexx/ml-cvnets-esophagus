#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

"""G4 esophagus-liver[+spleen] clinical datasets for the 4 cm protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
    G4_CLINICAL_FEATURE_NAMES,
    ClinicalRecord,
    case_id_from_record,
    encode_g4_clinical_features,
    load_clinical_records,
    validate_clinical_matches,
)


class _G4ClinicalDatasetMixin:
    clinical_records: dict[str, ClinicalRecord]

    def _initialize_clinical_data(self) -> None:
        expected_pairs = [
            [("esophagus", 15), ("liver", 5)],
            [("esophagus", 15), ("liver", 5), ("spleen", 1)],
        ]
        actual_pairs = [(spec.name, spec.label_id) for spec in self.organ_specs]
        if (
            actual_pairs not in expected_pairs
            or not self.organ_specs[0].required
            or any(spec.margin_xy or spec.margin_z for spec in self.organ_specs)
        ):
            raise ValueError(
                "G4 requires ordered zero-margin esophagus+liver[+spleen] "
                "with labels 15,5[,1] and required esophagus."
            )
        workbook_path = str(
            getattr(self.opts, "dataset.esophagus_g4.clinical_workbook", "")
        ).strip()
        if not workbook_path:
            raise ValueError(
                "G4 requires --dataset.esophagus-g4.clinical-workbook."
            )
        self.clinical_workbook = Path(workbook_path).expanduser()
        self.clinical_records = load_clinical_records(self.clinical_workbook)
        validate_clinical_matches(
            [(record.case_id, record.label) for record in self._clinical_eligible_records],
            self.clinical_records,
        )
        self.encoded_clinical_features = {
            case_id_from_record(record.case_id): encode_g4_clinical_features(
                self.clinical_records[case_id_from_record(record.case_id)]
            )
            for record in self._clinical_eligible_records
        }
        setattr(self.opts, "dataset.collate_fn_name_train", "pytorch_default_collate_fn")
        setattr(self.opts, "dataset.collate_fn_name_val", "pytorch_default_collate_fn")
        setattr(self.opts, "dataset.collate_fn_name_test", "pytorch_default_collate_fn")

    def _filter_short_esophagus_records(
        self, records: list[CaseRecord]
    ) -> list[CaseRecord]:
        eligible_records = super()._filter_short_esophagus_records(records)
        self._clinical_eligible_records = eligible_records
        return eligible_records

    def _clinical_features(self, record: CaseRecord) -> torch.Tensor:
        normalized_id = case_id_from_record(record.case_id)
        return torch.tensor(
            self.encoded_clinical_features[normalized_id],
            dtype=torch.float32,
        )

    def __getitem__(self, sample_size_and_index: Any) -> dict[str, Any]:
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
            + f"\n\tclinical_features={len(G4_CLINICAL_FEATURE_NAMES)}"
        )


@DATASET_REGISTRY.register(
    name="esophagus_risk_g4_clinical_roi_3d",
    type="classification",
)
class EsophagusRiskG4ClinicalRoi3DDataset(
    _G4ClinicalDatasetMixin,
    EsophagusRiskE3aMultiOrganRoi3DDataset,
):
    """Tongji G4 dataset with esophagus, liver, and clinical predictors."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self._initialize_clinical_data()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != EsophagusRiskG4ClinicalRoi3DDataset:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--dataset.esophagus-g4.clinical-workbook",
            type=str,
            default="",
            help="Path to the G4 clinical XLSX workbook for the active centre.",
        )
        return parser


@DATASET_REGISTRY.register(
    name="esophagus_risk_youyi_g4_clinical_roi_3d",
    type="classification",
)
class EsophagusRiskYouyiG4ClinicalRoi3DDataset(
    _G4ClinicalDatasetMixin,
    EsophagusRiskYouyiE3aMultiOrganRoi3DDataset,
):
    """Youyi external-only G4 dataset."""

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
                "esophagus_risk_youyi_g4_clinical_roi_3d is external-test only. "
                "Use esophagus_risk_g4_clinical_roi_3d for training and validation."
            )
        super().__init__(
            opts,
            *args,
            is_training=is_training,
            is_evaluation=is_evaluation,
            **kwargs,
        )
        self._initialize_clinical_data()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser
