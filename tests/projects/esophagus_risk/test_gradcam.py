"""Check organ-specific gradients, frozen encoders, and unbiased slice selection."""

import numpy as np
import torch
from projects.esophagus_risk.gradcam import compute_gradcam, resize_cam, select_slices
from torch import nn


class JointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv3d(1, 1, 1, bias=False)
        self.encoder.weight.data.fill_(1)
        self.encoder.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x.reshape(2, 1, 2, 2, 2))
        score = features[0].sum() + 2 * features[1].sum()
        return torch.stack((-score, score)).unsqueeze(0)


def test_joint_gradient_keeps_organ_axis_and_cleans_hooks() -> None:
    model = JointModel().eval()
    samples = torch.ones(1, 2, 1, 2, 2, 2)
    before = model.encoder.weight.detach().clone()
    cam, logits = compute_gradcam(model, samples, model.encoder, 1)
    torch.testing.assert_close(cam[0], torch.ones(2, 2, 2))
    torch.testing.assert_close(cam[1], torch.full((2, 2, 2), 2.0))
    assert logits.shape == (1, 2)
    assert not model.encoder._forward_hooks
    assert model.encoder.weight.grad is None
    torch.testing.assert_close(model.encoder.weight, before)
    negative_cam, _ = compute_gradcam(model, samples, model.encoder, 0)
    assert negative_cam.count_nonzero() == 0


def test_resize_and_mask_based_slices() -> None:
    resized = resize_cam(torch.ones(2, 2, 2), (5, 7, 9))
    np.testing.assert_allclose(resized, 1)
    mask = np.zeros((10, 4, 4), dtype=bool)
    mask[2:8, :2, :2] = True
    mask[5, :, :] = True
    selected = select_slices(mask, 3)
    assert 5 in selected
    assert len(selected) == 3
    assert selected == sorted(selected)
