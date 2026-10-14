#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

"""F4 image and clinical feature fusion for the 4 cm esophagus protocol."""

from __future__ import annotations

import argparse
from typing import Dict, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from common import is_test_env
from cvnets.layers import Dropout, LinearLayer
from cvnets.misc.init_utils import initialize_weights
from cvnets.models import MODEL_REGISTRY, BaseAnyNNModel
from cvnets.models.classification.suprem_unet_dual_mil_roi import (
    SuPreMUNetDualMILRoi,
)
from projects.esophagus_risk.clinical_data import CLINICAL_FEATURE_NAMES


class _SafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics for a one-case training batch."""

    def forward(self, input_tensor: Tensor) -> Tensor:
        use_batch_statistics = self.training and input_tensor.shape[0] > 1
        return F.batch_norm(
            input_tensor,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            use_batch_statistics,
            self.momentum,
            self.eps,
        )


@MODEL_REGISTRY.register(name="suprem_unet_f4_clinical_roi", type="classification")
class SuPreMUNetF4ClinicalRoi(SuPreMUNetDualMILRoi):
    """Fuse F2b image features with a small clinical MLP before classification."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        num_classes = int(getattr(opts, "model.classification.n_classes"))
        classifier_dropout = float(
            getattr(opts, "model.classification.classifier_dropout")
        )
        self.num_clinical_features = int(
            getattr(
                opts,
                "model.classification.suprem_unet_f4_clinical_roi.num_clinical_features",
                len(CLINICAL_FEATURE_NAMES),
            )
        )
        clinical_hidden_dim = int(
            getattr(
                opts,
                "model.classification.suprem_unet_f4_clinical_roi.clinical_hidden_dim",
                32,
            )
        )
        clinical_embedding_dim = int(
            getattr(
                opts,
                "model.classification.suprem_unet_f4_clinical_roi.clinical_embedding_dim",
                16,
            )
        )
        clinical_dropout = float(
            getattr(
                opts,
                "model.classification.suprem_unet_f4_clinical_roi.clinical_dropout",
                0.2,
            )
        )
        if self.num_clinical_features != len(CLINICAL_FEATURE_NAMES):
            raise ValueError(
                "F4 requires exactly {} clinical features, got {}.".format(
                    len(CLINICAL_FEATURE_NAMES), self.num_clinical_features
                )
            )
        if clinical_hidden_dim < 1 or clinical_embedding_dim < 1:
            raise ValueError("Clinical MLP dimensions must be positive.")

        self.clinical_encoder = nn.Sequential(
            _SafeBatchNorm1d(self.num_clinical_features),
            LinearLayer(
                in_features=self.num_clinical_features,
                out_features=clinical_hidden_dim,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            Dropout(p=clinical_dropout),
            LinearLayer(
                in_features=clinical_hidden_dim,
                out_features=clinical_embedding_dim,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        image_feature_dim = (self.feature_channels * 2) + num_classes
        fusion_layers: list[nn.Module] = []
        if 0.0 < classifier_dropout < 1.0:
            fusion_layers.append(Dropout(p=classifier_dropout))
        fusion_layers.append(
            LinearLayer(
                in_features=image_feature_dim + clinical_embedding_dim,
                out_features=num_classes,
                bias=True,
            )
        )
        self.fusion_classifier = nn.Sequential(*fusion_layers)
        initialize_weights(opts=opts, modules=self.clinical_encoder)
        initialize_weights(opts=opts, modules=self.fusion_classifier)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.suprem-unet-f4-clinical-roi.num-clinical-features",
            type=int,
            default=len(CLINICAL_FEATURE_NAMES),
        )
        group.add_argument(
            "--model.classification.suprem-unet-f4-clinical-roi.clinical-hidden-dim",
            type=int,
            default=32,
        )
        group.add_argument(
            "--model.classification.suprem-unet-f4-clinical-roi.clinical-embedding-dim",
            type=int,
            default=16,
        )
        group.add_argument(
            "--model.classification.suprem-unet-f4-clinical-roi.clinical-dropout",
            type=float,
            default=0.2,
        )
        return parser

    def forward(self, samples: Mapping[str, Tensor], *args, **kwargs) -> Tensor:
        if not isinstance(samples, Mapping):
            raise TypeError("F4 model expects samples={'image': ..., 'clinical': ...}.")
        if "image" not in samples or "clinical" not in samples:
            raise KeyError("F4 samples must contain both 'image' and 'clinical'.")

        image = samples["image"]
        clinical = samples["clinical"]
        if clinical.ndim != 2 or clinical.shape[1] != self.num_clinical_features:
            raise ValueError(
                "Clinical tensor must have shape [batch, {}], got {}.".format(
                    self.num_clinical_features,
                    tuple(clinical.shape),
                )
            )
        if image.shape[0] != clinical.shape[0]:
            raise ValueError("Image and clinical batch sizes must match.")
        if not torch.isfinite(clinical).all():
            raise ValueError("Clinical tensor contains NaN or infinite values.")

        image_features = self.extract_fusion_features(image)
        clinical_features = self.clinical_encoder(clinical.float())
        return self.fusion_classifier(
            torch.cat([image_features, clinical_features], dim=1)
        )

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        if is_test_env():
            depth = height = width = 32
        else:
            depth = int(getattr(self.opts, "dataset.esophagus_roi_3d.input_depth", 128))
            height = int(
                getattr(self.opts, "dataset.esophagus_roi_3d.input_height", 160)
            )
            width = int(getattr(self.opts, "dataset.esophagus_roi_3d.input_width", 160))
        return {
            "samples": {
                "image": torch.randn(batch_size, 1, depth, height, width),
                "clinical": torch.randn(batch_size, self.num_clinical_features),
            },
            "targets": torch.randint(low=0, high=2, size=(batch_size,)).long(),
        }

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        model = cls(opts, *args, **kwargs)
        if getattr(opts, "model.classification.freeze_batch_norm"):
            cls.freeze_norm_layers(opts=opts, model=model)
        return model
