#!/usr/bin/env python3
"""Export full axial CT with spatially restored esophagus/liver ROI Grad-CAM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger(__name__)
DEFAULT_CASES = ("08887988", "08577671", "13646226", "02377541", "12123738")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Matching training YAML"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Internal-validation AUC-selected checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Parent of youyi-nii and youyi-mask",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Must not already exist"
    )
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--target-class", type=int, choices=(0, 1), default=1)
    parser.add_argument("--slices-per-organ", type=int, default=3)
    parser.add_argument(
        "--target-layer",
        default="backbone.down_tr512.ops",
        help="5D activation before pooling",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Exact training-time cvnets overrides, e.g. two-organ settings",
    )
    args = parser.parse_args()
    if args.slices_per_organ < 1 or args.dpi < 72 or not 0 < args.alpha <= 1:
        parser.error("Positive slice count, dpi >= 72, and 0 < alpha <= 1 required.")
    if len(set(args.case_ids)) != len(args.case_ids):
        parser.error("Duplicate requested case IDs.")
    for name in ("config", "checkpoint", "data_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_crop(volume: Any, crop: Any) -> Tuple[Tuple[int, int], ...]:
    """Recover exact bounds from the native NumPy slice used by preprocessing."""
    import numpy as np

    if not np.shares_memory(volume, crop) or any(s <= 0 for s in volume.strides):
        raise ValueError("Expected a native crop view with positive volume strides.")
    offset = crop.ctypes.data - volume.ctypes.data
    start = [0] * volume.ndim
    for axis in sorted(range(volume.ndim), key=lambda i: -volume.strides[i]):
        start[axis], offset = divmod(offset, volume.strides[axis])
    bounds = tuple((int(a), int(a + n)) for a, n in zip(start, crop.shape))
    restored = volume[tuple(slice(a, b) for a, b in bounds)]
    if offset or restored.shape != crop.shape or not np.array_equal(restored, crop):
        raise ValueError("Native ROI bounds could not be verified.")
    return bounds


def build_dataset(opts: argparse.Namespace, case_ids: List[str]) -> Any:
    import numpy as np
    from data.datasets.classification.esophagus_risk_youyi_e3a_multi_organ_roi_3d import (
        EsophagusRiskYouyiE3aMultiOrganRoi3DDataset,
    )
    from data.datasets.classification.esophagus_risk_youyi_roi_3d import (
        _discover_youyi_records,
    )

    class CaptureDataset(EsophagusRiskYouyiE3aMultiOrganRoi3DDataset):
        """Reuse inference preprocessing and capture native crops before resizing."""

        def _extract_organ_tensor(
            self,
            ct_volume: np.ndarray,
            mask_volume: np.ndarray,
            spec: Any,
            slice_spacing_mm: float,
            slice_axis_direction: int,
        ) -> Any:
            self.full_ct = ct_volume
            return super()._extract_organ_tensor(
                ct_volume, mask_volume, spec, slice_spacing_mm, slice_axis_direction
            )

        def _load_records(self) -> List[Any]:
            records = _discover_youyi_records(
                Path(getattr(self.opts, "dataset.root_test")),
                self.nii_dir_name,
                self.mask_dir_name,
            )
            selected = []
            for case_id in case_ids:
                matches = [r for r in records if Path(r.case_id).name == case_id]
                if len(matches) != 1:
                    raise ValueError(
                        f"Case {case_id}: expected exactly one match, found {len(matches)}"
                    )
                selected.append(matches[0])
            eligible = self._filter_short_esophagus_records(selected)
            if len(eligible) != len(selected):
                raise ValueError(
                    "A requested case fails the configured esophagus-height rule."
                )
            return selected

        def _normalize_roi(self, roi: np.ndarray) -> np.ndarray:
            self.native_bounds.append(locate_crop(self.full_ct, roi))
            self.current_native_ct = roi.copy()
            return super()._normalize_roi(roi)

        def _resize_ct_and_mask(self, ct_roi: np.ndarray, mask_roi: np.ndarray) -> Any:
            self.native_crops.append(
                (self.current_native_ct, mask_roi.copy(), ct_roi.copy())
            )
            return super()._resize_ct_and_mask(ct_roi, mask_roi)

    dataset = CaptureDataset(opts, is_training=False, is_evaluation=True)
    if [s.name for s in dataset.organ_specs] != ["esophagus", "liver"]:
        raise ValueError(
            "Training configuration must specify esophagus,liver in that order."
        )
    if {s.name for s in dataset.organ_specs if s.required} != {"esophagus", "liver"}:
        # A missing optional organ is allowed in training, but cannot be visualized.
        LOGGER.info("Both requested organ masks must be present for visualization.")
    return dataset


def load_model(opts: argparse.Namespace, checkpoint: Path, device: str) -> Any:
    import torch
    from cvnets.models.classification.suprem_unet_e3a_residual_context import (
        SuPreMUNetE3aResidualContext,
    )

    if getattr(opts, "model.classification.name") != "suprem_unet_e3a_residual_context":
        raise ValueError(
            "This script supports the F3 esophagus/liver residual-context model."
        )
    prefix = "model.classification.suprem_unet_e3a_residual_context"
    setattr(opts, f"{prefix}.pretrained_path", "")
    setattr(opts, f"{prefix}.require_pretrained", False)
    model = SuPreMUNetE3aResidualContext(opts)
    if model.num_organs != 2:
        raise ValueError("The selected model must use exactly two organs.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def export_figure(
    crops: List[Tuple[Any, Any, Any]],
    cams: List[Any],
    spacing: Tuple[float, ...],
    destination: Path,
    args: argparse.Namespace,
    full_ct: Any,
    bounds: List[Tuple[Tuple[int, int], ...]],
    display_window: Tuple[float, float],
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from projects.esophagus_risk.gradcam import select_slices

    fig, axes = plt.subplots(
        2,
        args.slices_per_organ * 2,
        squeeze=False,
        figsize=(args.slices_per_organ * 5.0, 6.4),
    )
    selection: Dict[str, Any] = {}
    for row, (name, crop, cam) in enumerate(zip(("Esophagus", "Liver"), crops, cams)):
        _, mask, _ = crop
        (z_start, _), (x_start, x_end), (y_start, y_end) = bounds[row]
        slices = select_slices(mask, args.slices_per_organ)
        selection[name.lower()] = slices
        for axis in axes[row]:
            axis.set_axis_off()
        for column, index in enumerate(slices):
            # NIfTI [X,Y,Z] -> dataset [Z,X,Y] -> display [Y,X].
            full_index = z_start + index
            lower, upper = display_window
            ct_slice = np.rot90(
                np.clip((full_ct[full_index] - lower) / (upper - lower), 0, 1)
            )
            heat = np.zeros(full_ct.shape[1:], dtype=np.float32)
            opacity = np.zeros_like(heat)
            heat[x_start:x_end, y_start:y_end] = cam[index]
            if np.max(cam) > 0:
                opacity[x_start:x_end, y_start:y_end] = args.alpha
            left, right = axes[row, column * 2 : column * 2 + 2]
            for axis in (left, right):
                axis.imshow(
                    ct_slice,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    aspect=spacing[1] / spacing[0],
                    interpolation="nearest",
                )
            right.imshow(
                np.rot90(heat),
                cmap="jet",
                vmin=0,
                vmax=1,
                aspect=spacing[1] / spacing[0],
                alpha=np.rot90(opacity),
                interpolation="bilinear",
            )
            left.set_title(f"{name} | CT | z={full_index}", fontsize=9)
            right.set_title("ROI Grad-CAM on full CT", fontsize=9)
    fig.subplots_adjust(
        left=0.01, right=0.93, bottom=0.04, top=0.94, wspace=0.05, hspace=0.18
    )
    color_axis = fig.add_axes((0.95, 0.2, 0.012, 0.6))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="jet"),
        cax=color_axis,
        label="Relative activation (per-organ maximum)",
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            destination.with_suffix(f".{extension}"),
            dpi=args.dpi,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)
    return selection


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    from options.opts import get_eval_arguments
    from projects.esophagus_risk.gradcam import compute_gradcam, resize_cam
    from projects.esophagus_risk.nifti_utils import read_nifti

    for path in (args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    opts = get_eval_arguments(
        args=[
            "--common.config-file",
            str(args.config),
            "--common.override-kwargs",
            *args.override,
        ]
    )
    setattr(opts, "dataset.root_test", str(args.data_root))
    dataset = build_dataset(opts, args.case_ids)
    model = load_model(opts, args.checkpoint, args.device)
    layer = model.get_submodule(args.target_layer)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config_sha256": sha256(args.config),
        "selection": "Operator-supplied internal-validation AUC-selected checkpoint; not verified locally",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_options": {key: str(value) for key, value in vars(opts).items()},
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else None
        ),
        "device": str(args.device),
        "target_score": "pre-softmax class logit",
        "slice_rule": "largest organ-mask area with spacing; independent of CAM and prediction",
        "normalization": "ReLU, per-organ 3D maximum; no cross-organ/case magnitude comparison",
        "orientation": "full native voxel axial planes, no anatomical L/R claims",
        "display": "full CT; CAM restored only inside the exact model ROI; outside ROI uncolored",
        "cases": [],
    }
    manifest = args.output_dir / "metadata.json"
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
    )
    (args.output_dir / "environment_packages.txt").write_text(
        "\n".join(packages) + "\n"
    )
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    for index, record in enumerate(dataset.records):
        dataset.native_crops = []
        dataset.native_bounds = []
        sample = dataset[index]
        if len(dataset.native_crops) != 2:
            raise ValueError(f"Both organ crops must be present: {record.case_id}")
        cams, logits = compute_gradcam(
            model,
            sample["samples"].unsqueeze(0).to(args.device),
            layer,
            args.target_class,
        )
        native_cams = []
        peaks = []
        for cam, crop in zip(cams, dataset.native_crops):
            peak = float(cam.max())
            peaks.append(peak)
            native_cams.append(
                resize_cam(cam / peak if peak > 0 else cam, crop[0].shape)
            )
        spacing = read_nifti(record.ct_path).spacing
        # Anonymous figure labels keep hospital identifiers out of manuscript panels.
        stem = args.output_dir / f"case_{index + 1:02d}"
        if dataset.use_windowing:
            lower = dataset.window_center - dataset.window_width / 2
            upper = dataset.window_center + dataset.window_width / 2
        else:
            lower, upper = np.percentile(
                dataset.full_ct,
                [dataset.clip_percentile_lower, dataset.clip_percentile_upper],
            ).tolist()
        display_window = (float(lower), float(max(upper, lower + 1)))
        slices = export_figure(
            dataset.native_crops,
            native_cams,
            spacing,
            stem,
            args,
            dataset.full_ct,
            dataset.native_bounds,
            display_window,
        )
        np.savez_compressed(
            stem.with_suffix(".npz"),
            esophagus_cam=native_cams[0],
            liver_cam=native_cams[1],
            esophagus_ct=dataset.native_crops[0][0],
            liver_ct=dataset.native_crops[1][0],
        )
        metadata["cases"].append(
            {
                "figure": stem.name,
                "case_id": record.case_id,
                "target": record.label,
                "ct_path": str(record.ct_path),
                "mask_path": str(record.mask_path),
                "hr_probability": float(logits.softmax(dim=1)[0, 1]),
                "selected_roi_local_z": slices,
                "selected_full_ct_z": {
                    name: [z + dataset.native_bounds[i][0][0] for z in indices]
                    for i, (name, indices) in enumerate(slices.items())
                },
                "roi_bounds_zxy": dataset.native_bounds,
                "display_window_hu": display_window,
                "native_spacing_xyz": list(spacing),
                "raw_cam_maxima": peaks,
                "empty_positive_cam": [peak == 0 for peak in peaks],
            }
        )
        manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        LOGGER.info("Exported %s", stem.name)
    metadata["status"] = "complete"
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
