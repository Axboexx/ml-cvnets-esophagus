#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch import Tensor, nn

from common import is_test_env
from cvnets.layers import ConvLayer3d, Dropout, GlobalPool, LinearLayer
from cvnets.models import MODEL_REGISTRY, BaseAnyNNModel


def _make_stage(
    opts: argparse.Namespace,
    in_channels: int,
    out_channels: int,
    stride: int,
) -> nn.Sequential:
    return nn.Sequential(
        ConvLayer3d(
            opts=opts,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            use_norm=True,
            use_act=True,
        ),
        ConvLayer3d(
            opts=opts,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_norm=True,
            use_act=True,
        ),
    )


@MODEL_REGISTRY.register(name="esophagus_roi_3d", type="classification")
class EsophagusRoi3D(BaseAnyNNModel):
    """A compact 3D CNN classifier for esophagus ROI binary risk prediction."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        num_classes = int(getattr(opts, "model.classification.n_classes"))
        base_channels = int(
            getattr(opts, "model.classification.esophagus_roi_3d.base_channels")
        )
        classifier_dropout = float(
            getattr(opts, "model.classification.classifier_dropout")
        )
        pool_type = getattr(opts, "model.layer.global_pool")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = ConvLayer3d(
            opts=opts,
            in_channels=1,
            out_channels=c1,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )
        self.stage_1 = _make_stage(opts=opts, in_channels=c1, out_channels=c1, stride=1)
        self.stage_2 = _make_stage(opts=opts, in_channels=c1, out_channels=c2, stride=2)
        self.stage_3 = _make_stage(opts=opts, in_channels=c2, out_channels=c3, stride=2)
        self.stage_4 = _make_stage(opts=opts, in_channels=c3, out_channels=c4, stride=2)
        self.expand = ConvLayer3d(
            opts=opts,
            in_channels=c4,
            out_channels=c4,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        classifier_layers: list[nn.Module] = [
            GlobalPool(pool_type=pool_type, keep_dim=False)
        ]
        if 0.0 < classifier_dropout < 1.0:
            classifier_layers.append(Dropout(p=classifier_dropout))
        classifier_layers.append(
            LinearLayer(in_features=c4, out_features=num_classes, bias=True)
        )
        self.classifier = nn.Sequential(*classifier_layers)

        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.esophagus-roi-3d.base-channels",
            type=int,
            default=32,
            help="Base channel width for the 3D ROI classifier.",
        )
        return parser

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.expand(x)
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.extract_features(x)
        return self.classifier(x)

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        if is_test_env():
            depth = 32
            height = 32
            width = 32
        else:
            depth = int(getattr(self.opts, "dataset.esophagus_roi_3d.input_depth", 96))
            height = int(
                getattr(self.opts, "dataset.esophagus_roi_3d.input_height", 96)
            )
            width = int(getattr(self.opts, "dataset.esophagus_roi_3d.input_width", 96))

        sample_tensor = torch.randn(batch_size, 1, depth, height, width)
        target_tensor = torch.randint(low=0, high=2, size=(batch_size,)).long()
        return {"samples": sample_tensor, "targets": target_tensor}

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        model = cls(opts, *args, **kwargs)
        if getattr(opts, "model.classification.freeze_batch_norm"):
            cls.freeze_norm_layers(opts=opts, model=model)
        return model
