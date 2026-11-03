#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

from unittest import TestCase

import torch
from torch import Tensor, nn

from cvnets.models.classification.suprem_unet_f5_clinical_context import (
    SuPreMUNetF5ClinicalContext,
    _SafeBatchNorm1d,
)


class _TinyF5Harness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_organs = 2
        self.input_channels_per_organ = 1
        self.num_clinical_features = 15
        self.image_encoder = nn.Linear(8, 512)
        self.clinical_encoder = nn.Sequential(
            _SafeBatchNorm1d(15),
            nn.Linear(15, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(528, 2)

    def extract_image_features(self, image: Tensor) -> Tensor:
        return self.image_encoder(image)

    def forward(self, samples: dict[str, Tensor]) -> Tensor:
        return SuPreMUNetF5ClinicalContext.forward(self, samples)


class SuPreMUNetF5ClinicalContextTest(TestCase):
    def test_512_plus_16_forward_and_backward(self) -> None:
        model = _TinyF5Harness()
        output = model(
            {
                "image": torch.randn(2, 8),
                "clinical": torch.randn(2, 15),
            }
        )
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(model.classifier.in_features, 528)
        output.sum().backward()
        self.assertIsNotNone(model.image_encoder.weight.grad)
        self.assertIsNotNone(model.clinical_encoder[1].weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_one_case_training_batch_is_supported(self) -> None:
        model = _TinyF5Harness().train()
        output = model(
            {
                "image": torch.randn(1, 8),
                "clinical": torch.randn(1, 15),
            }
        )
        self.assertEqual(tuple(output.shape), (1, 2))

    def test_rejects_missing_or_invalid_clinical_input(self) -> None:
        model = _TinyF5Harness()
        with self.assertRaises(KeyError):
            model({"image": torch.randn(2, 8)})
        with self.assertRaisesRegex(ValueError, "shape"):
            model(
                {
                    "image": torch.randn(2, 8),
                    "clinical": torch.randn(2, 6),
                }
            )
        clinical = torch.randn(2, 15)
        clinical[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "infinite"):
            model({"image": torch.randn(2, 8), "clinical": clinical})
