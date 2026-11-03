"""Check native crop recovery and full-plane Grad-CAM placement."""

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts/visualize_esophagus_liver_gradcam.py"
)
SPEC = importlib.util.spec_from_file_location("gradcam_display", SCRIPT)
DISPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPLAY)


@pytest.mark.parametrize("order", ["C", "F"])
def test_crop_bounds_after_depth_transpose(order: str) -> None:
    volume = np.array(np.arange(13 * 17 * 11).reshape(13, 17, 11), order=order)
    volume = volume.transpose(2, 0, 1)
    crop = volume[2:7, 3:10, 4:12]
    assert DISPLAY.locate_crop(volume, crop) == ((2, 7), (3, 10), (4, 12))
    with pytest.raises(ValueError):
        DISPLAY.locate_crop(volume, crop.copy())


@pytest.mark.parametrize("outside_only", [False, True])
def test_full_plane_overlay_stays_inside_organ_mask(
    tmp_path: Path, monkeypatch: Any, outside_only: bool
) -> None:
    volume = np.arange(8 * 20 * 24, dtype=np.float32).reshape(8, 20, 24)
    bounds = ((2, 5), (4, 10), (7, 15))
    crop = volume[2:5, 4:10, 7:15]
    mask = np.ones_like(crop)
    mask[1] *= 2  # Select a known native layer, full CT z=3.
    mask[1, 1:3, 2:5] = 0  # Non-organ tissue inside the bounding box.
    cam = np.full_like(crop, 0.6)
    if outside_only:
        cam = np.where(mask, 0.0, cam)
    args = argparse.Namespace(slices_per_organ=1, alpha=0.55, dpi=72)
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    selected = DISPLAY.export_figure(
        [(crop, mask, crop)] * 2,
        [cam, cam],
        (0.8, 0.8, 2.0),
        tmp_path / "full_plane",
        args,
        volume,
        [bounds, bounds],
        (0, 4000),
    )
    assert selected == {"esophagus": [1], "liver": [1]}
    fig = figures[-1]
    left, right = fig.axes[:2]
    np.testing.assert_allclose(left.images[0].get_array(), np.rot90(volume[3] / 4000))
    alpha = np.rot90(right.images[1].get_alpha(), k=-1)
    expected = np.zeros((20, 24), dtype=np.float32)
    if not outside_only:
        expected[4:10, 7:15] = np.where(mask[1], 0.55, 0.0)
    np.testing.assert_allclose(alpha, expected)
    heat = np.rot90(right.images[1].get_array(), k=-1)
    expected_heat = np.zeros((20, 24), dtype=np.float32)
    expected_heat[4:10, 7:15] = np.where(mask[1], cam[1], 0.0)
    np.testing.assert_allclose(heat, expected_heat)
    assert (tmp_path / "full_plane.png").is_file()
    assert (tmp_path / "full_plane.pdf").is_file()
