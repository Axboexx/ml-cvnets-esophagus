#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor, nn

from common import is_test_env
from cvnets.layers import Dropout, GlobalPool, LinearLayer
from cvnets.models import MODEL_REGISTRY, BaseAnyNNModel
from cvnets.models.classification.suprem_unet_roi import (
    UNet3D,
    _UNET_IMPORT_ERROR,
    _disable_unused_unet_decoder_gradients,
    _extract_state_dict,
    _filter_matching_weights,
)
from utils import logger


@MODEL_REGISTRY.register(name="suprem_unet_e3a_residual_context", type="classification")
class SuPreMUNetE3aResidualContext(BaseAnyNNModel):
    """E3a multi-organ model with organ context added to the esophagus feature."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        if UNet3D is None:
            logger.error(
                "Unable to import SuPreM UNet3D. Set SUPREM_ROOT to the SuPreM "
                "repository root if it is not located at ../SuPreM or ~/work/SuPreM. "
                f"Original error: {_UNET_IMPORT_ERROR}"
            )

        num_classes = int(getattr(opts, "model.classification.n_classes"))
        classifier_dropout = float(getattr(opts, "model.classification.classifier_dropout"))
        pool_type = getattr(opts, "model.layer.global_pool")

        self.num_organs = int(
            getattr(
                opts,
                "model.classification.suprem_unet_e3a_residual_context.num_organs",
                4,
            )
        )
        self.input_channels_per_organ = int(
            getattr(
                opts,
                "model.classification.suprem_unet_e3a_residual_context.input_channels_per_organ",
                1,
            )
        )
        self.context_scale = float(
            getattr(
                opts,
                "model.classification.suprem_unet_e3a_residual_context.context_scale",
                1.0,
            )
        )

        if self.num_organs < 2:
            logger.error("E3a residual context requires esophagus plus at least one context organ.")
        if self.input_channels_per_organ != 1:
            logger.error("E3a residual context expects one masked CT channel per organ.")
        if self.context_scale < 0.0:
            logger.error("context_scale must be non-negative for E3a residual context.")

        self.input_adapter = nn.Conv3d(
            in_channels=self.input_channels_per_organ,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )
        self.backbone = UNet3D()
        _disable_unused_unet_decoder_gradients(self.backbone)
        self.feature_channels = 512
        self.feature_pool = GlobalPool(pool_type=pool_type, keep_dim=False)

        context_features = self.feature_channels * (self.num_organs - 1)
        self.context_projector = LinearLayer(
            in_features=context_features,
            out_features=self.feature_channels,
            bias=True,
        )

        classifier_layers: list[nn.Module] = []
        if 0.0 < classifier_dropout < 1.0:
            classifier_layers.append(Dropout(p=classifier_dropout))
        classifier_layers.append(
            LinearLayer(
                in_features=self.feature_channels,
                out_features=num_classes,
                bias=True,
            )
        )
        self.classifier = nn.Sequential(*classifier_layers)

        self.reset_parameters(opts=opts)
        self._init_input_adapter()
        self._init_context_projector()
        self._load_suprem_checkpoint(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.suprem-unet-e3a-residual-context.pretrained-path",
            type=str,
            default="",
            help="Path to supervised_suprem_unet_2100.pth.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-e3a-residual-context.require-pretrained",
            action="store_true",
            default=False,
            help="Fail model construction when pretrained_path is empty.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-e3a-residual-context.num-organs",
            type=int,
            default=4,
            help="Number of organ-specific crops in each sample.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-e3a-residual-context.input-channels-per-organ",
            type=int,
            default=1,
            help="Channels per organ crop. E3a uses one organ-masked CT channel.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-e3a-residual-context.context-scale",
            type=float,
            default=1.0,
            help="Multiplier applied to the projected non-esophagus context residual.",
        )
        return parser

    def _init_input_adapter(self) -> None:
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            self.input_adapter.weight[:, 0:1].fill_(1.0)

    def _init_context_projector(self) -> None:
        # Start from an esophagus-only decision function; context learns a residual update.
        with torch.no_grad():
            self.context_projector.weight.zero_()
            if self.context_projector.bias is not None:
                self.context_projector.bias.zero_()

    def _load_suprem_checkpoint(self, opts: argparse.Namespace) -> None:
        pretrained_path = str(
            getattr(
                opts,
                "model.classification.suprem_unet_e3a_residual_context.pretrained_path",
                "",
            )
        ).strip()
        require_pretrained = bool(
            getattr(
                opts,
                "model.classification.suprem_unet_e3a_residual_context.require_pretrained",
                False,
            )
        )

        if not pretrained_path:
            if require_pretrained and not is_test_env():
                logger.error(
                    "SuPreM U-Net E3a residual-context fine-tuning requires "
                    "--model.classification.suprem-unet-e3a-residual-context.pretrained-path."
                )
            logger.warning("Building SuPreM U-Net E3a residual-context model without weights.")
            return

        checkpoint_path = Path(pretrained_path).expanduser()
        if not checkpoint_path.is_file():
            logger.error(f"SuPreM checkpoint does not exist: {checkpoint_path}")

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        source_state = _extract_state_dict(checkpoint)
        target_state = self.backbone.state_dict()
        matched_state = _filter_matching_weights(
            source_state=source_state,
            target_state=target_state,
        )

        if not matched_state:
            logger.error(f"No SuPreM weights matched UNet3D keys in {checkpoint_path}.")

        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            matched_state,
            strict=False,
        )
        logger.log(
            "Loaded SuPreM U-Net weights for E3a residual context from {}: matched={}, "
            "missing={}, unexpected={}".format(
                checkpoint_path,
                len(matched_state),
                len(missing_keys),
                len(unexpected_keys),
            )
        )

    def extract_features(self, x: Tensor) -> Tensor:
        out64, _ = self.backbone.down_tr64(x)
        out128, _ = self.backbone.down_tr128(out64)
        out256, _ = self.backbone.down_tr256(out128)
        out512, _ = self.backbone.down_tr512(out256)
        return out512

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if x.dim() != 6:
            logger.error(
                "E3a residual context expects samples with shape [B, O, C, D, H, W], "
                f"got {tuple(x.shape)}."
            )

        batch_size, num_organs, channels, depth, height, width = x.shape
        if num_organs != self.num_organs:
            logger.error(f"Expected {self.num_organs} organs, got {num_organs}.")
        if channels != self.input_channels_per_organ:
            logger.error(
                f"Expected {self.input_channels_per_organ} channels per organ, got {channels}."
            )

        x = x.reshape(batch_size * num_organs, channels, depth, height, width)
        x = self.input_adapter(x)
        features = self.feature_pool(self.extract_features(x))
        features = features.reshape(batch_size, num_organs, self.feature_channels)

        esophagus_features = features[:, 0]
        context_features = features[:, 1:].reshape(batch_size, -1)
        context_residual = self.context_projector(context_features)
        fused_features = esophagus_features + self.context_scale * context_residual
        return self.classifier(fused_features)

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        if is_test_env():
            depth = 32
            height = 32
            width = 32
        else:
            depth = int(getattr(self.opts, "dataset.e3a_multi_organ_roi_3d.input_depth", 128))
            height = int(getattr(self.opts, "dataset.e3a_multi_organ_roi_3d.input_height", 160))
            width = int(getattr(self.opts, "dataset.e3a_multi_organ_roi_3d.input_width", 160))

        sample_tensor = torch.randn(
            batch_size,
            self.num_organs,
            self.input_channels_per_organ,
            depth,
            height,
            width,
        )
        target_tensor = torch.randint(low=0, high=2, size=(batch_size,)).long()
        return {"samples": sample_tensor, "targets": target_tensor}

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        model = cls(opts, *args, **kwargs)
        if getattr(opts, "model.classification.freeze_batch_norm"):
            cls.freeze_norm_layers(opts=opts, model=model)
        return model
