#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

import argparse
import math
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


@MODEL_REGISTRY.register(name="suprem_unet_mil_roi", type="classification")
class SuPreMUNetMILRoi(BaseAnyNNModel):
    """SuPreM U-Net encoder with segment-level MIL pooling for 3D ROI."""

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        if UNet3D is None:
            logger.error(
                "Unable to import SuPreM UNet3D. Set SUPREM_ROOT to the SuPreM "
                "repository root if it is not located at ../SuPreM or ~/work/SuPreM. "
                f"Original error: {_UNET_IMPORT_ERROR}"
            )

        num_classes = int(getattr(opts, "model.classification.n_classes"))
        classifier_dropout = float(
            getattr(opts, "model.classification.classifier_dropout")
        )
        pool_type = getattr(opts, "model.layer.global_pool")

        self.num_segments = int(
            getattr(opts, "model.classification.suprem_unet_mil_roi.num_segments", 6)
        )
        self.segment_depth = int(
            getattr(opts, "model.classification.suprem_unet_mil_roi.segment_depth", 48)
        )
        self.segment_overlap = float(
            getattr(
                opts,
                "model.classification.suprem_unet_mil_roi.segment_overlap",
                0.25,
            )
        )
        self.mil_pooling = str(
            getattr(opts, "model.classification.suprem_unet_mil_roi.mil_pooling", "topk")
        ).lower()
        self.topk_segments = int(
            getattr(opts, "model.classification.suprem_unet_mil_roi.topk_segments", 2)
        )

        if self.num_segments < 1:
            logger.error("num_segments must be >= 1 for SuPreM U-Net MIL ROI.")
        if self.topk_segments < 1:
            logger.error("topk_segments must be >= 1 for SuPreM U-Net MIL ROI.")
        if self.mil_pooling not in {"topk", "max", "mean", "attention"}:
            logger.error(
                "mil_pooling must be one of ['topk', 'max', 'mean', 'attention']."
            )

        self.backbone = UNet3D()
        _disable_unused_unet_decoder_gradients(self.backbone)
        self.feature_channels = 512
        self.feature_pool = GlobalPool(pool_type=pool_type, keep_dim=False)

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

        self.attention: nn.Module | None = None
        if self.mil_pooling == "attention":
            attention_hidden_dim = max(32, self.feature_channels // 4)
            self.attention = nn.Sequential(
                LinearLayer(
                    in_features=self.feature_channels,
                    out_features=attention_hidden_dim,
                    bias=True,
                ),
                nn.Tanh(),
                LinearLayer(in_features=attention_hidden_dim, out_features=1, bias=True),
            )

        self.reset_parameters(opts=opts)
        self._load_suprem_checkpoint(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.suprem-unet-mil-roi.pretrained-path",
            type=str,
            default="",
            help="Path to supervised_suprem_unet_2100.pth.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-mil-roi.require-pretrained",
            action="store_true",
            default=False,
            help="Fail model construction when pretrained_path is empty.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-mil-roi.num-segments",
            type=int,
            default=6,
            help="Number of depth-axis esophagus segments per case.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-mil-roi.segment-depth",
            type=int,
            default=48,
            help="Depth of each segment. Values <= 0 derive the depth from overlap.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-mil-roi.segment-overlap",
            type=float,
            default=0.25,
            help="Approximate overlap ratio used when segment_depth <= 0.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-mil-roi.mil-pooling",
            type=str,
            default="topk",
            help="MIL pooling type: topk, max, mean, or attention.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-mil-roi.topk-segments",
            type=int,
            default=2,
            help="Number of high-risk segments averaged by top-k MIL pooling.",
        )
        return parser

    def _load_suprem_checkpoint(self, opts: argparse.Namespace) -> None:
        pretrained_path = str(
            getattr(
                opts,
                "model.classification.suprem_unet_mil_roi.pretrained_path",
                "",
            )
        ).strip()
        require_pretrained = bool(
            getattr(
                opts,
                "model.classification.suprem_unet_mil_roi.require_pretrained",
                False,
            )
        )

        if not pretrained_path:
            if require_pretrained and not is_test_env():
                logger.error(
                    "SuPreM U-Net MIL fine-tuning requires "
                    "--model.classification.suprem-unet-mil-roi.pretrained-path."
                )
            logger.warning("Building SuPreM U-Net MIL ROI model without weights.")
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
            "Loaded SuPreM U-Net MIL weights from {}: matched={}, missing={}, "
            "unexpected={}".format(
                checkpoint_path,
                len(matched_state),
                len(missing_keys),
                len(unexpected_keys),
            )
        )

    def _resolve_segment_depth(self, input_depth: int) -> int:
        input_depth = int(input_depth)
        if self.segment_depth > 0:
            return min(self.segment_depth, input_depth)

        overlap = min(max(self.segment_overlap, 0.0), 0.95)
        coverage_factor = 1.0 + (self.num_segments - 1) * (1.0 - overlap)
        depth = int(math.ceil(input_depth / coverage_factor))
        return max(1, min(depth, input_depth))

    def _segment_starts(self, input_depth: int, segment_depth: int) -> list[int]:
        input_depth = int(input_depth)
        segment_depth = int(segment_depth)
        if self.num_segments == 1 or segment_depth >= input_depth:
            return [0]

        max_start = input_depth - segment_depth
        return [
            int(round(max_start * segment_idx / (self.num_segments - 1)))
            for segment_idx in range(self.num_segments)
        ]

    def _split_segments(self, x: Tensor) -> Tensor:
        input_depth = int(x.shape[2])
        segment_depth = self._resolve_segment_depth(input_depth=input_depth)
        starts = self._segment_starts(
            input_depth=input_depth,
            segment_depth=segment_depth,
        )
        return torch.stack(
            [x[:, :, start : start + segment_depth, :, :] for start in starts],
            dim=1,
        )

    def extract_features(self, x: Tensor) -> Tensor:
        out64, _ = self.backbone.down_tr64(x)
        out128, _ = self.backbone.down_tr128(out64)
        out256, _ = self.backbone.down_tr256(out128)
        out512, _ = self.backbone.down_tr512(out256)
        return out512

    def _pool_segment_logits(self, segment_logits: Tensor) -> Tensor:
        if self.mil_pooling == "mean":
            return segment_logits.mean(dim=1)

        positive_logits = segment_logits[:, :, 1]
        if self.mil_pooling == "max":
            top_indices = positive_logits.argmax(dim=1, keepdim=True)
            gather_indices = top_indices.unsqueeze(-1).expand(
                -1,
                -1,
                segment_logits.size(-1),
            )
            return segment_logits.gather(dim=1, index=gather_indices).squeeze(1)

        topk = min(self.topk_segments, segment_logits.shape[1])
        top_indices = positive_logits.topk(k=topk, dim=1).indices
        gather_indices = top_indices.unsqueeze(-1).expand(
            -1,
            -1,
            segment_logits.size(-1),
        )
        return segment_logits.gather(dim=1, index=gather_indices).mean(dim=1)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        batch_size, channels, _, height, width = x.shape
        segments = self._split_segments(x)
        num_segments = segments.shape[1]
        segments = segments.reshape(
            batch_size * num_segments,
            channels,
            segments.shape[3],
            height,
            width,
        )

        segment_features = self.extract_features(segments)
        segment_features = self.feature_pool(segment_features)
        segment_features = segment_features.reshape(
            batch_size,
            num_segments,
            self.feature_channels,
        )
        flat_segment_features = segment_features.reshape(
            batch_size * num_segments,
            self.feature_channels,
        )

        if self.mil_pooling == "attention":
            if self.attention is None:
                logger.error("Attention pooling is selected but attention is not initialized.")
            attention_logits = self.attention(flat_segment_features).reshape(
                batch_size,
                num_segments,
            )
            attention_weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
            case_features = (segment_features * attention_weights).sum(dim=1)
            return self.classifier(case_features)

        segment_logits = self.classifier(flat_segment_features).reshape(
            batch_size,
            num_segments,
            -1,
        )
        return self._pool_segment_logits(segment_logits=segment_logits)

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        if is_test_env():
            depth = 32
            height = 32
            width = 32
        else:
            depth = int(getattr(self.opts, "dataset.esophagus_roi_3d.input_depth", 128))
            height = int(
                getattr(self.opts, "dataset.esophagus_roi_3d.input_height", 160)
            )
            width = int(getattr(self.opts, "dataset.esophagus_roi_3d.input_width", 160))

        sample_tensor = torch.randn(batch_size, 1, depth, height, width)
        target_tensor = torch.randint(low=0, high=2, size=(batch_size,)).long()
        return {"samples": sample_tensor, "targets": target_tensor}

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        model = cls(opts, *args, **kwargs)
        if getattr(opts, "model.classification.freeze_batch_norm"):
            cls.freeze_norm_layers(opts=opts, model=model)
        return model
