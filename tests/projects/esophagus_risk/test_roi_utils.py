from __future__ import annotations

import pytest

from projects.esophagus_risk.roi_utils import inferior_to_superior_depth_bounds


@pytest.mark.parametrize(
    ("slice_spacing_mm", "expected_bounds"),
    [
        (5.0, (10, 18)),
        (2.5, (10, 26)),
        (1.25, (10, 42)),
    ],
)
def test_inferior_to_superior_bounds_use_physical_height(
    slice_spacing_mm: float,
    expected_bounds: tuple[int, int],
) -> None:
    bounds = inferior_to_superior_depth_bounds(
        mask_depth_indices=range(10, 31),
        volume_depth=100,
        height_mm=40.0,
        slice_spacing_mm=slice_spacing_mm,
        slice_axis_direction=1,
    )

    assert bounds == expected_bounds


def test_inferior_to_superior_bounds_follow_negative_slice_direction() -> None:
    bounds = inferior_to_superior_depth_bounds(
        mask_depth_indices=range(10, 31),
        volume_depth=100,
        height_mm=40.0,
        slice_spacing_mm=2.5,
        slice_axis_direction=-1,
    )

    assert bounds == (15, 31)


@pytest.mark.parametrize(
    ("slice_axis_direction", "mask_depth_indices", "expected_bounds"),
    [
        (1, [95, 96], (95, 100)),
        (-1, [2, 3], (0, 4)),
    ],
)
def test_inferior_to_superior_bounds_clip_to_volume(
    slice_axis_direction: int,
    mask_depth_indices: list[int],
    expected_bounds: tuple[int, int],
) -> None:
    bounds = inferior_to_superior_depth_bounds(
        mask_depth_indices=mask_depth_indices,
        volume_depth=100,
        height_mm=40.0,
        slice_spacing_mm=2.5,
        slice_axis_direction=slice_axis_direction,
    )

    assert bounds == expected_bounds


def test_inferior_to_superior_bounds_round_up_partial_slice() -> None:
    bounds = inferior_to_superior_depth_bounds(
        mask_depth_indices=[10],
        volume_depth=100,
        height_mm=40.0,
        slice_spacing_mm=3.0,
        slice_axis_direction=1,
    )

    assert bounds == (10, 24)
