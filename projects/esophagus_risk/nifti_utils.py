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


def _read_header(header: bytes) -> tuple[str, int, tuple[int, int, int], float, tuple[float, float, float]]:
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
    spacing = (float(pixdim[1]), float(pixdim[2]), float(pixdim[3]))
    return endian, datatype, shape, vox_offset, spacing


def read_nifti(path: str | Path) -> NiftiVolume:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            payload = handle.read()
    else:
        payload = path.read_bytes()

    header = payload[:348]
    endian, datatype, shape, vox_offset, spacing = _read_header(header)
    dtype = np.dtype(_NIFTI_DTYPE_MAP[datatype]).newbyteorder(endian)

    data_offset = int(vox_offset) if vox_offset >= 348 else 352
    num_values = shape[0] * shape[1] * shape[2]
    data = np.frombuffer(payload, dtype=dtype, count=num_values, offset=data_offset)
    data = data.reshape(shape, order="F")
    return NiftiVolume(data=data, spacing=spacing)
