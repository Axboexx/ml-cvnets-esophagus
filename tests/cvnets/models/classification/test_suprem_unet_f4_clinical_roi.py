#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

from unittest import TestCase

import torch
from torch import Tensor, nn

from cvnets.models.classification.suprem_unet_f4_clinical_roi import (
    SuPreMUNetF4ClinicalRoi,
    _SafeBatchNorm1d,
)


class _TinyF4Harness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_clinical_features = 15
        self.image_encoder = nn.Linear(4, 6)
        self.clinical_encoder = nn.Sequential(
            _SafeBatchNorm1d(15),
            nn.Linear(15, 3),
            nn.ReLU(),
        )
        self.fusion_classifier = nn.Linear(9, 2)

    def extract_fusion_features(self, image: Tensor) -> Tensor:
        return self.image_encoder(image)

    def forward(self, samples: dict[str, Tensor]) -> Tensor:
        return SuPreMUNetF4ClinicalRoi.forward(self, samples)


class SuPreMUNetF4ClinicalRoiTest(TestCase):
    def test_nested_forward_and_backward(self) -> None:
        model = _TinyF4Harness()
        output = model(
            {
                "image": torch.randn(2, 4),
                "clinical": torch.randn(2, 15),
            }
        )
        self.assertEqual(tuple(output.shape), (2, 2))
        output.sum().backward()
        self.assertIsNotNone(model.image_encoder.weight.grad)
        self.assertIsNotNone(model.clinical_encoder[1].weight.grad)
        self.assertIsNotNone(model.fusion_classifier.weight.grad)

    def test_one_case_training_batch_is_supported(self) -> None:
        model = _TinyF4Harness().train()
        output = model(
            {
                "image": torch.randn(1, 4),
                "clinical": torch.randn(1, 15),
            }
        )
        self.assertEqual(tuple(output.shape), (1, 2))

    def test_rejects_missing_or_invalid_clinical_input(self) -> None:
        model = _TinyF4Harness()
        with self.assertRaises(KeyError):
            model({"image": torch.randn(2, 4)})
        with self.assertRaisesRegex(ValueError, "shape"):
            model(
                {
                    "image": torch.randn(2, 4),
                    "clinical": torch.randn(2, 6),
                }
            )
        clinical = torch.randn(2, 15)
        clinical[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            model({"image": torch.randn(2, 4), "clinical": clinical})
