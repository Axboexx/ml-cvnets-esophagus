from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from data.datasets.classification.esophagus_risk_e3a_multi_organ_roi_3d import (
    EsophagusRiskE3aMultiOrganRoi3DDataset,
    OrganSpec,
    _build_organ_specs,
)
from data.datasets.classification.esophagus_risk_roi_3d import (
    CaseRecord,
    EsophagusRiskRoi3DDataset,
    _filter_records_by_esophagus_height,
)
from data.datasets.classification.esophagus_risk_youyi_roi_3d import (
    EsophagusRiskYouyiRoi3DDataset,
)
from data.datasets.classification.esophagus_risk_youyi_e3a_multi_organ_roi_3d import (
    EsophagusRiskYouyiE3aMultiOrganRoi3DDataset,
)


def _make_tongji_dataset(root: Path, cases_per_class: int = 10) -> None:
    for class_name in ("nhr", "hr"):
        for case_index in range(cases_per_class):
            case_dir = root / class_name / f"case_{case_index:02d}"
            case_dir.mkdir(parents=True)
            (case_dir / "ct.nii.gz").write_bytes(b"")
            (case_dir / "mask.nii.gz").write_bytes(b"")


def _make_youyi_dataset(root: Path, cases_per_class: int = 3) -> None:
    for class_name in ("nhr", "hr"):
        for case_index in range(cases_per_class):
            case_name = f"case_{case_index:02d}"
            ct_dir = root / "youyi-nii" / class_name / case_name
            mask_dir = root / "youyi-mask" / class_name / case_name
            ct_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            (ct_dir / "ct.nii.gz").write_bytes(b"")
            (mask_dir / "mask.nii.gz").write_bytes(b"")


def _record_loader(
    dataset_class: type[EsophagusRiskRoi3DDataset],
    root: Path,
    mode: str,
    test_root: Path | None = None,
) -> list[CaseRecord]:
    dataset = object.__new__(dataset_class)
    dataset.opts = argparse.Namespace()
    setattr(dataset.opts, "dataset.root_train", str(root))
    setattr(dataset.opts, "dataset.root_val", "")
    setattr(dataset.opts, "dataset.root_test", str(test_root) if test_root else "")
    dataset.mode = mode
    dataset.val_split_ratio = 0.2
    dataset.split_seed = 42
    dataset.label_id = 15
    dataset.min_esophagus_height_mm = 0.0
    if dataset_class is EsophagusRiskYouyiRoi3DDataset:
        dataset.nii_dir_name = "youyi-nii"
        dataset.mask_dir_name = "youyi-mask"
    return dataset._load_records()


def test_height_filter_excludes_only_esophagus_masks_below_40_mm(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    records: list[CaseRecord] = []
    mask_volumes: dict[Path, SimpleNamespace] = {}
    for case_id, num_slices in (("short", 19), ("exact", 20), ("long", 21)):
        mask_path = tmp_path / f"{case_id}.nii.gz"
        mask = np.zeros((4, 4, 30), dtype=np.int16)
        mask[1:3, 1:3, 2 : 2 + num_slices] = 15
        records.append(CaseRecord(case_id, 0, tmp_path / "ct.nii.gz", mask_path))
        mask_volumes[mask_path] = SimpleNamespace(
            data=mask,
            spacing=(1.0, 1.0, 2.0),
        )

    monkeypatch.setattr(
        "data.datasets.classification.esophagus_risk_roi_3d.read_nifti",
        lambda path: mask_volumes[path],
    )

    included, excluded = _filter_records_by_esophagus_height(
        records=records,
        label_id=15,
        min_height_mm=40.0,
    )

    assert [record.case_id for record in included] == ["exact", "long"]
    assert [(record.case_id, height_mm) for record, height_mm in excluded] == [
        ("short", 38.0)
    ]


def test_tongji_uses_same_stratified_holdout_for_validation_and_test(
    tmp_path: Path,
) -> None:
    _make_tongji_dataset(tmp_path)

    train_records = _record_loader(EsophagusRiskRoi3DDataset, tmp_path, "train")
    val_records = _record_loader(EsophagusRiskRoi3DDataset, tmp_path, "val")
    test_records = _record_loader(EsophagusRiskRoi3DDataset, tmp_path, "test")

    assert len(train_records) == 16
    assert len(val_records) == 4
    assert [record.case_id for record in val_records] == [
        record.case_id for record in test_records
    ]
    assert {record.case_id for record in train_records}.isdisjoint(
        record.case_id for record in val_records
    )
    assert {record.label for record in val_records} == {0, 1}


def test_youyi_test_mode_keeps_the_full_external_dataset(tmp_path: Path) -> None:
    _make_youyi_dataset(tmp_path)

    records = _record_loader(EsophagusRiskYouyiRoi3DDataset, tmp_path, "test")

    assert len(records) == 6
    assert {record.label for record in records} == {0, 1}


def test_youyi_e3a_test_mode_applies_esophagus_height_filter(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    _make_youyi_dataset(tmp_path)
    dataset = object.__new__(EsophagusRiskYouyiE3aMultiOrganRoi3DDataset)
    dataset.opts = argparse.Namespace()
    setattr(dataset.opts, "dataset.root_train", "")
    setattr(dataset.opts, "dataset.root_val", "")
    setattr(dataset.opts, "dataset.root_test", str(tmp_path))
    dataset.mode = "test"
    dataset.nii_dir_name = "youyi-nii"
    dataset.mask_dir_name = "youyi-mask"
    filter_calls: list[list[CaseRecord]] = []

    def record_filter(records: list[CaseRecord]) -> list[CaseRecord]:
        filter_calls.append(records)
        dataset.num_cases_before_height_filter = len(records)
        dataset.num_cases_excluded_by_height = 1
        return records[1:]

    monkeypatch.setattr(dataset, "_filter_short_esophagus_records", record_filter)

    records = dataset._load_records()

    assert len(filter_calls) == 1
    assert len(filter_calls[0]) == 6
    assert len(records) == 5
    assert dataset.num_cases_before_height_filter == 6
    assert dataset.num_cases_excluded_by_height == 1


def test_equivalent_test_root_does_not_bypass_tongji_holdout(tmp_path: Path) -> None:
    train_root = tmp_path / "tongji"
    alias_root = tmp_path / "tongji-alias"
    _make_tongji_dataset(train_root)
    alias_root.symlink_to(train_root, target_is_directory=True)

    records = _record_loader(
        EsophagusRiskRoi3DDataset,
        train_root,
        "test",
        test_root=alias_root,
    )

    assert len(records) == 4


def test_fixed_height_roi_uses_tight_xy_mask_bounds() -> None:
    dataset = object.__new__(EsophagusRiskRoi3DDataset)
    dataset.label_id = 15
    dataset.roi_margin_xy = 99
    dataset.roi_margin_z = 99
    dataset.inferior_to_superior_height_mm = 40.0
    ct_volume = np.arange(30 * 20 * 20, dtype=np.float32).reshape(30, 20, 20)
    mask_volume = np.zeros_like(ct_volume, dtype=np.int32)
    mask_volume[5:10, 4:8, 6:11] = 15

    roi = dataset._extract_roi(
        ct_volume=ct_volume,
        mask_volume=mask_volume,
        slice_spacing_mm=2.0,
        slice_axis_direction=1,
    )

    assert roi.shape == (20, 4, 5)
    np.testing.assert_array_equal(roi, ct_volume[5:25, 4:8, 6:11])


def test_multi_organ_specs_disable_margins_for_every_organ() -> None:
    opts = argparse.Namespace()
    prefix = "dataset.e3a_multi_organ_roi_3d"
    setattr(opts, f"{prefix}.organ_names", "esophagus,liver,spleen,portal_venous_system")
    setattr(opts, f"{prefix}.label_ids", "15,5,1,64")
    setattr(opts, f"{prefix}.roi_margin_xy", "0,0,0,0")
    setattr(opts, f"{prefix}.roi_margin_z", "0,0,0,0")
    setattr(opts, f"{prefix}.required_organs", "esophagus")

    specs = _build_organ_specs(opts)

    assert [spec.name for spec in specs] == [
        "esophagus",
        "liver",
        "spleen",
        "portal_venous_system",
    ]
    assert all(spec.margin_xy == 0 for spec in specs)
    assert all(spec.margin_z == 0 for spec in specs)


def test_non_esophagus_roi_uses_tight_xyz_mask_bounds() -> None:
    dataset = object.__new__(EsophagusRiskE3aMultiOrganRoi3DDataset)
    dataset.use_windowing = True
    dataset.window_center = 50.0
    dataset.window_width = 350.0
    dataset.esophagus_inferior_to_superior_height_mm = 40.0
    dataset._resize_ct_and_mask = lambda ct_roi, mask_roi: (ct_roi, mask_roi)
    ct_volume = np.arange(20 * 20 * 20, dtype=np.float32).reshape(20, 20, 20)
    mask_volume = np.zeros_like(ct_volume, dtype=np.int32)
    mask_volume[5:9, 4:8, 6:11] = 1
    spec = OrganSpec(
        name="spleen",
        label_id=1,
        margin_xy=0,
        margin_z=0,
        required=True,
    )

    ct_roi, mask_roi = dataset._extract_organ_tensor(
        ct_volume=ct_volume,
        mask_volume=mask_volume,
        spec=spec,
        slice_spacing_mm=2.0,
        slice_axis_direction=1,
    )

    assert ct_roi.shape == (4, 4, 5)
    assert mask_roi.shape == (4, 4, 5)
    assert mask_roi.all()
