#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

"""G4 esophagus-liver[+spleen] residual context with clinical fusion."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from common import is_test_env
from cvnets.layers import Dropout, LinearLayer
from cvnets.misc.init_utils import initialize_weights
from cvnets.models import MODEL_REGISTRY, BaseAnyNNModel
from cvnets.models.classification.suprem_unet_e3a_residual_context import (
    SuPreMUNetE3aResidualContext,
)
from projects.esophagus_risk.clinical_data import G4_CLINICAL_FEATURE_NAMES


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


@MODEL_REGISTRY.register(name="suprem_unet_g4_clinical_context", type="classification")
class SuPreMUNetG4ClinicalContext(SuPreMUNetE3aResidualContext):
    """Fuse a 512-D multi-organ representation with clinical predictors."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        if self.num_organs not in (2, 3):
            raise ValueError("G4 requires two or three organs: esophagus, liver, optionally spleen.")

        num_classes = int(getattr(opts, "model.classification.n_classes"))
        classifier_dropout = float(
            getattr(opts, "model.classification.classifier_dropout")
        )
        self.num_clinical_features = int(
            getattr(
                opts,
                "model.classification.suprem_unet_g4_clinical_context.num_clinical_features",
                len(G4_CLINICAL_FEATURE_NAMES),
            )
        )
        clinical_hidden_dim = int(
            getattr(
                opts,
                "model.classification.suprem_unet_g4_clinical_context.clinical_hidden_dim",
                32,
            )
        )
        clinical_embedding_dim = int(
            getattr(
                opts,
                "model.classification.suprem_unet_g4_clinical_context.clinical_embedding_dim",
                16,
            )
        )
        clinical_dropout = float(
            getattr(
                opts,
                "model.classification.suprem_unet_g4_clinical_context.clinical_dropout",
                0.2,
            )
        )
        if self.num_clinical_features != len(G4_CLINICAL_FEATURE_NAMES):
            raise ValueError(
                f"G4 requires exactly {len(G4_CLINICAL_FEATURE_NAMES)} clinical features, got {self.num_clinical_features}."
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

        fusion_layers: list[nn.Module] = []
        if 0.0 < classifier_dropout < 1.0:
            fusion_layers.append(Dropout(p=classifier_dropout))
        fusion_layers.append(
            LinearLayer(
                in_features=self.feature_channels + clinical_embedding_dim,
                out_features=num_classes,
                bias=True,
            )
        )
        # Replace the inherited 512 -> 2 head with the G4 528 -> 2 head.
        self.classifier = nn.Sequential(*fusion_layers)
        initialize_weights(opts=opts, modules=self.clinical_encoder)
        initialize_weights(opts=opts, modules=self.classifier)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.suprem-unet-g4-clinical-context.num-clinical-features",
            type=int,
            default=len(G4_CLINICAL_FEATURE_NAMES),
        )
        group.add_argument(
            "--model.classification.suprem-unet-g4-clinical-context.clinical-hidden-dim",
            type=int,
            default=32,
        )
        group.add_argument(
            "--model.classification.suprem-unet-g4-clinical-context.clinical-embedding-dim",
            type=int,
            default=16,
        )
        group.add_argument(
            "--model.classification.suprem-unet-g4-clinical-context.clinical-dropout",
            type=float,
            default=0.2,
        )
        return parser

    def extract_image_features(self, image: Tensor) -> Tensor:
        if image.dim() != 6:
            raise ValueError(
                "G4 image tensor must have shape [B, O, C, D, H, W], "
                f"got {tuple(image.shape)}."
            )
        batch_size, num_organs, channels, depth, height, width = image.shape
        if num_organs != self.num_organs:
            raise ValueError(f"Expected {self.num_organs} organs, got {num_organs}.")
        if channels != self.input_channels_per_organ:
            raise ValueError(
                f"Expected {self.input_channels_per_organ} channels per organ, got {channels}."
            )

        image = image.reshape(
            batch_size * num_organs,
            channels,
            depth,
            height,
            width,
        )
        image = self.input_adapter(image)
        features = self.feature_pool(self.extract_features(image))
        features = features.reshape(batch_size, num_organs, self.feature_channels)
        esophagus_features = features[:, 0]
        context_features = features[:, 1:].reshape(batch_size, -1)
        return esophagus_features + self.context_scale * self.context_projector(
            context_features
        )

    def forward(self, samples: Mapping[str, Tensor], *args, **kwargs) -> Tensor:
        if not isinstance(samples, Mapping):
            raise TypeError("G4 model expects samples={'image': ..., 'clinical': ...}.")
        if "image" not in samples or "clinical" not in samples:
            raise KeyError("G4 samples must contain both 'image' and 'clinical'.")

        image = samples["image"]
        clinical = samples["clinical"]
        if clinical.ndim != 2 or clinical.shape[1] != self.num_clinical_features:
            raise ValueError(
                f"Clinical tensor must have shape [batch, {self.num_clinical_features}], got {tuple(clinical.shape)}."
            )
        if image.shape[0] != clinical.shape[0]:
            raise ValueError("Image and clinical batch sizes must match.")
        if not torch.isfinite(clinical).all():
            raise ValueError("Clinical tensor contains NaN or infinite values.")

        image_features = self.extract_image_features(image)
        clinical_features = self.clinical_encoder(clinical.float())
        return self.classifier(
            torch.cat([image_features, clinical_features], dim=1)
        )

    def dummy_input_and_label(self, batch_size: int) -> dict:
        if is_test_env():
            depth = height = width = 32
        else:
            depth = int(
                getattr(self.opts, "dataset.e3a_multi_organ_roi_3d.input_depth", 128)
            )
            height = int(
                getattr(self.opts, "dataset.e3a_multi_organ_roi_3d.input_height", 160)
            )
            width = int(
                getattr(self.opts, "dataset.e3a_multi_organ_roi_3d.input_width", 160)
            )
        return {
            "samples": {
                "image": torch.randn(
                    batch_size,
                    self.num_organs,
                    self.input_channels_per_organ,
                    depth,
                    height,
                    width,
                ),
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
