from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from projects.esophagus_risk.nifti_utils import read_nifti, validate_spatial_alignment


def _write_test_nifti(
    path: Path,
    slice_spacing_mm: float,
    slice_axis_direction: int,
    z_offset_mm: float = 0.0,
) -> None:
    shape = (2, 3, 4)
    header = bytearray(352)
    struct.pack_into("<I", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *shape, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)
    struct.pack_into(
        "<8f",
        header,
        76,
        1.0,
        1.0,
        1.0,
        slice_spacing_mm,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<4f", header, 280, 1.0, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, 1.0, 0.0, 0.0)
    struct.pack_into(
        "<4f",
        header,
        312,
        0.0,
        0.0,
        slice_axis_direction * slice_spacing_mm,
        z_offset_mm,
    )
    payload = np.arange(np.prod(shape), dtype=np.float32).tobytes(order="F")
    path.write_bytes(header + payload)


@pytest.mark.parametrize("slice_axis_direction", [-1, 1])
def test_read_nifti_reads_spacing_and_slice_direction(
    tmp_path: Path,
    slice_axis_direction: int,
) -> None:
    nifti_path = tmp_path / "volume.nii"
    _write_test_nifti(
        path=nifti_path,
        slice_spacing_mm=2.5,
        slice_axis_direction=slice_axis_direction,
    )

    volume = read_nifti(nifti_path)

    assert volume.data.shape == (2, 3, 4)
    assert volume.spacing == pytest.approx((1.0, 1.0, 2.5))
    assert volume.slice_axis_direction == slice_axis_direction


@pytest.mark.parametrize(("qfac", "expected_direction"), [(1.0, 1), (-1.0, -1)])
def test_read_nifti_uses_qform_when_sform_is_absent(
    tmp_path: Path,
    qfac: float,
    expected_direction: int,
) -> None:
    nifti_path = tmp_path / "qform_volume.nii"
    _write_test_nifti(nifti_path, slice_spacing_mm=2.5, slice_axis_direction=1)
    payload = bytearray(nifti_path.read_bytes())
    struct.pack_into("<2h", payload, 252, 1, 0)
    struct.pack_into("<f", payload, 76, qfac)
    nifti_path.write_bytes(payload)

    volume = read_nifti(nifti_path)

    assert volume.affine[2][2] == pytest.approx(expected_direction * 2.5)
    assert volume.slice_axis_direction == expected_direction


def test_valid_sform_does_not_fall_back_to_conflicting_qform(tmp_path: Path) -> None:
    nifti_path = tmp_path / "conflicting_forms.nii"
    _write_test_nifti(nifti_path, slice_spacing_mm=2.5, slice_axis_direction=1)
    payload = bytearray(nifti_path.read_bytes())
    struct.pack_into("<h", payload, 252, 1)
    struct.pack_into("<f", payload, 76, -1.0)
    nifti_path.write_bytes(payload)

    volume = read_nifti(nifti_path)

    assert volume.slice_axis_direction == 1


def test_valid_sform_with_unresolved_depth_direction_raises(tmp_path: Path) -> None:
    nifti_path = tmp_path / "invalid_sform.nii"
    _write_test_nifti(nifti_path, slice_spacing_mm=2.5, slice_axis_direction=1)
    payload = bytearray(nifti_path.read_bytes())
    struct.pack_into("<h", payload, 252, 1)
    struct.pack_into("<4f", payload, 312, 2.5, 0.0, 0.0, 0.0)
    nifti_path.write_bytes(payload)

    with pytest.raises(ValueError, match="no resolvable superior-inferior"):
        read_nifti(nifti_path)


def test_spatial_alignment_rejects_different_affines(tmp_path: Path) -> None:
    ct_path = tmp_path / "ct.nii"
    mask_path = tmp_path / "mask.nii"
    _write_test_nifti(ct_path, 2.5, 1, z_offset_mm=0.0)
    _write_test_nifti(mask_path, 2.5, 1, z_offset_mm=5.0)

    with pytest.raises(ValueError, match="affines differ"):
        validate_spatial_alignment(
            read_nifti(ct_path),
            read_nifti(mask_path),
            case_id="case-001",
        )
