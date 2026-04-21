from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


BackboneName = Literal["resnet18", "resnet34"]


def build_backbone(backbone_name: BackboneName, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
    elif backbone_name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)
        feature_dim = backbone.fc.in_features
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone.fc = nn.Identity()
    return backbone, feature_dim


class RoiCropCaseClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: BackboneName = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name=backbone_name, pretrained=pretrained)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: Tensor, slice_mask: Tensor) -> Tensor:
        batch_size, num_slices, channels, height, width = images.shape
        flat_images = images.view(batch_size * num_slices, channels, height, width)
        flat_features = self.backbone(flat_images)
        features = flat_features.view(batch_size, num_slices, -1)

        mask = slice_mask.unsqueeze(-1)
        masked_features = features * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = masked_features.sum(dim=1) / denom
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
