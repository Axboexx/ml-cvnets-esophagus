from __future__ import annotations

import math
from collections.abc import Iterable


def inferior_to_superior_depth_bounds(
    mask_depth_indices: Iterable[int],
    volume_depth: int,
    height_mm: float,
    slice_spacing_mm: float,
    slice_axis_direction: int,
) -> tuple[int, int]:
    """Return half-open depth bounds from the organ's inferior end toward superior."""
    indices = tuple(int(index) for index in mask_depth_indices)
    if not indices:
        raise ValueError("mask_depth_indices must not be empty")
    if volume_depth <= 0:
        raise ValueError(f"volume_depth must be positive, got {volume_depth}")
    if not math.isfinite(height_mm) or height_mm <= 0.0:
        raise ValueError(f"height_mm must be positive, got {height_mm}")
    if not math.isfinite(slice_spacing_mm) or slice_spacing_mm <= 0.0:
        raise ValueError(f"slice_spacing_mm must be positive, got {slice_spacing_mm}")
    if slice_axis_direction not in {-1, 1}:
        raise ValueError(
            f"slice_axis_direction must be -1 or 1, got {slice_axis_direction}"
        )

    num_slices = max(1, int(math.ceil(height_mm / slice_spacing_mm)))
    if slice_axis_direction > 0:
        depth_start = min(indices)
        depth_end = min(volume_depth, depth_start + num_slices)
    else:
        depth_end = max(indices) + 1
        depth_start = max(0, depth_end - num_slices)
    return depth_start, depth_end
