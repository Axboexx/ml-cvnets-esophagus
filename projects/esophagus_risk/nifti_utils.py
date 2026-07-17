from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


_NIFTI_DTYPE_MAP = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
    1280: np.uint64,
}


@dataclass(frozen=True)
class NiftiVolume:
    data: np.ndarray
    spacing: Tuple[float, float, float]
    affine: Tuple[Tuple[float, float, float, float], ...]
    slice_axis_direction: int


def _direction_sign(delta: float) -> int | None:
    if not np.isfinite(delta) or abs(delta) < 1.0e-6:
        return None
    return 1 if delta > 0.0 else -1


def _qform_affine(
    header: bytes,
    endian: str,
    pixdim: tuple[float, ...],
) -> Tuple[Tuple[float, float, float, float], ...]:
    b, c, d = struct.unpack(f"{endian}3f", header[256:268])
    squared_norm = b * b + c * c + d * d
    if squared_norm > 1.0:
        scale = 1.0 / np.sqrt(squared_norm)
        b, c, d = b * scale, c * scale, d * scale
        a = 0.0
    else:
        a = float(np.sqrt(max(0.0, 1.0 - squared_norm)))

    rotation = np.asarray(
        [
            [
                a * a + b * b - c * c - d * d,
                2.0 * (b * c - a * d),
                2.0 * (b * d + a * c),
            ],
            [
                2.0 * (b * c + a * d),
                a * a + c * c - b * b - d * d,
                2.0 * (c * d - a * b),
            ],
            [
                2.0 * (b * d - a * c),
                2.0 * (c * d + a * b),
                a * a + d * d - b * b - c * c,
            ],
        ],
        dtype=np.float64,
    )
    qfac = -1.0 if pixdim[0] < 0.0 else 1.0
    rotation[:, 0] *= abs(pixdim[1])
    rotation[:, 1] *= abs(pixdim[2])
    rotation[:, 2] *= abs(pixdim[3]) * qfac
    qoffset = struct.unpack(f"{endian}3f", header[268:280])

    return tuple(
        tuple(float(value) for value in row)
        for row in (
            (*rotation[0], qoffset[0]),
            (*rotation[1], qoffset[1]),
            (*rotation[2], qoffset[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _read_affine(
    header: bytes,
    endian: str,
    pixdim: tuple[float, ...],
) -> Tuple[Tuple[float, float, float, float], ...]:
    qform_code, sform_code = struct.unpack(f"{endian}2h", header[252:256])
    if sform_code > 0:
        return (
            struct.unpack(f"{endian}4f", header[280:296]),
            struct.unpack(f"{endian}4f", header[296:312]),
            struct.unpack(f"{endian}4f", header[312:328]),
            (0.0, 0.0, 0.0, 1.0),
        )
    if qform_code > 0:
        return _qform_affine(header=header, endian=endian, pixdim=pixdim)
    return (
        (abs(pixdim[1]), 0.0, 0.0, 0.0),
        (0.0, abs(pixdim[2]), 0.0, 0.0),
        (0.0, 0.0, abs(pixdim[3]), 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _slice_axis_direction(
    affine: Tuple[Tuple[float, float, float, float], ...],
) -> int:
    direction = _direction_sign(float(affine[2][2]))
    if direction is None:
        raise ValueError(
            "NIfTI depth axis has no resolvable superior-inferior component."
        )
    return direction


def _read_header(
    header: bytes,
) -> tuple[
    str,
    int,
    tuple[int, int, int],
    float,
    tuple[float, float, float],
    Tuple[Tuple[float, float, float, float], ...],
    int,
]:
    sizeof_hdr_le = struct.unpack("<I", header[:4])[0]
    sizeof_hdr_be = struct.unpack(">I", header[:4])[0]

    if sizeof_hdr_le == 348:
        endian = "<"
    elif sizeof_hdr_be == 348:
        endian = ">"
    else:
        raise ValueError("Invalid NIfTI header. Expected sizeof_hdr=348.")

    dims = struct.unpack(f"{endian}8h", header[40:56])
    if dims[0] < 3:
        raise ValueError(f"Expected at least 3 dimensions, got {dims[0]}.")
    shape = (int(dims[1]), int(dims[2]), int(dims[3]))

    datatype = struct.unpack(f"{endian}h", header[70:72])[0]
    if datatype not in _NIFTI_DTYPE_MAP:
        raise ValueError(f"Unsupported NIfTI datatype code: {datatype}")

    vox_offset = struct.unpack(f"{endian}f", header[108:112])[0]
    pixdim = struct.unpack(f"{endian}8f", header[76:108])
    spacing = tuple(abs(float(value)) for value in pixdim[1:4])
    if any(not np.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError(f"Invalid NIfTI voxel spacing: {spacing}")
    affine = _read_affine(header=header, endian=endian, pixdim=pixdim)
    slice_axis_direction = _slice_axis_direction(affine)
    return endian, datatype, shape, vox_offset, spacing, affine, slice_axis_direction


def read_nifti(path: str | Path) -> NiftiVolume:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            payload = handle.read()
    else:
        payload = path.read_bytes()

    header = payload[:348]
    (
        endian,
        datatype,
        shape,
        vox_offset,
        spacing,
        affine,
        slice_axis_direction,
    ) = _read_header(header)
    dtype = np.dtype(_NIFTI_DTYPE_MAP[datatype]).newbyteorder(endian)

    data_offset = int(vox_offset) if vox_offset >= 348 else 352
    num_values = shape[0] * shape[1] * shape[2]
    data = np.frombuffer(payload, dtype=dtype, count=num_values, offset=data_offset)
    data = data.reshape(shape, order="F")
    return NiftiVolume(
        data=data,
        spacing=spacing,
        affine=affine,
        slice_axis_direction=slice_axis_direction,
    )


def validate_spatial_alignment(
    reference: NiftiVolume,
    candidate: NiftiVolume,
    case_id: str,
) -> None:
    if reference.data.shape != candidate.data.shape:
        raise ValueError(
            f"CT and mask shapes differ for {case_id}: "
            f"{reference.data.shape} != {candidate.data.shape}"
        )
    if not np.allclose(
        np.asarray(reference.affine),
        np.asarray(candidate.affine),
        rtol=1.0e-5,
        atol=1.0e-4,
    ):
        raise ValueError(
            f"CT and mask NIfTI affines differ for {case_id}; "
            "resample them to one voxel grid before ROI extraction."
        )
