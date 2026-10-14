"""3D Grad-CAM and anatomy-based slice selection for joint organ models."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def compute_gradcam(
    model: nn.Module, samples: Tensor, layer: nn.Module, target_class: int
) -> Tuple[Tensor, Tensor]:
    """Explain one joint prediction; preserve the flattened organ batch axis.

    The target layer must run once and return [organs, channels, D, H, W],
    or a tuple whose first element has that shape (SuPreM down transitions).
    Returns unnormalized positive CAMs and logits on CPU. No parameter update.
    """
    captured: List[Tensor] = []

    def capture(module: nn.Module, inputs: Any, output: Any) -> None:
        activation = output[0] if isinstance(output, tuple) else output
        if not isinstance(activation, Tensor) or activation.ndim != 5:
            raise ValueError("Target layer must expose a 5D feature tensor.")
        captured.append(activation)

    handle = layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            # Input gradients keep this working when encoder parameters are frozen.
            logits = model(samples.detach().requires_grad_(True))
            if logits.shape != (1, 2) or target_class not in (0, 1):
                raise ValueError("Expected one binary-classification sample.")
            if len(captured) != 1 or captured[0].shape[0] != samples.shape[1]:
                raise ValueError(
                    "Expected one target-layer call with one row per organ."
                )
            activation = captured[0]
            gradient = torch.autograd.grad(logits[0, target_class], activation)[0]
            weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
            cams = (weights * activation).sum(dim=1).relu()
            if not torch.isfinite(cams).all() or not torch.isfinite(logits).all():
                raise ValueError("Non-finite logits or Grad-CAM values.")
            return cams.detach().cpu(), logits.detach().cpu()
    finally:
        handle.remove()


def resize_cam(cam: Tensor, shape: Tuple[int, ...]) -> np.ndarray:
    """Map a coarse CAM onto a native crop with the training resize convention."""
    return F.interpolate(
        cam[None, None], size=shape, mode="trilinear", align_corners=False
    )[0, 0].numpy()


def select_slices(mask: np.ndarray, count: int) -> List[int]:
    """Select largest mask-area slices, with deterministic tie order and spacing."""
    if mask.ndim != 3 or count < 1:
        raise ValueError("Expected a 3D mask and a positive slice count.")
    area = mask.sum(axis=(1, 2))
    candidates = np.flatnonzero(area > 0)
    if not len(candidates):
        raise ValueError("Cannot select representative slices from an empty mask.")
    ranked = sorted(candidates.tolist(), key=lambda index: (-area[index], index))
    gap = max(1, len(candidates) // (count * 2))
    selected: List[int] = []
    for index in ranked:
        if all(abs(index - other) >= gap for other in selected):
            selected.append(index)
        if len(selected) == count:
            break
    return sorted(selected)
