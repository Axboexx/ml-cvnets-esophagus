#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
from torch import Tensor, nn

from common import is_test_env
from cvnets.layers import Dropout, GlobalPool, LinearLayer
from cvnets.models import MODEL_REGISTRY, BaseAnyNNModel
from utils import logger


def _candidate_suprem_pretraining_roots(library_root: Path) -> Iterable[Path]:
    project_root = library_root.parent
    env_root = os.environ.get("SUPREM_ROOT", "").strip()
    if env_root:
        yield Path(env_root).expanduser() / "supervised_pretraining"
        yield Path(env_root).expanduser()
    yield project_root / "SuPreM" / "supervised_pretraining"
    yield Path.home() / "work" / "SuPreM" / "supervised_pretraining"
    yield Path.home() / "project" / "SuPreM" / "supervised_pretraining"


def _add_suprem_to_path() -> None:
    library_root = Path(__file__).resolve().parents[3]
    for suprem_pretraining in _candidate_suprem_pretraining_roots(library_root):
        if suprem_pretraining.exists():
            suprem_path = str(suprem_pretraining)
            if suprem_path not in sys.path:
                sys.path.insert(0, suprem_path)
            return


_add_suprem_to_path()

try:
    from backbone.unet import UNet3D
except ImportError as import_error:
    UNet3D = None
    _UNET_IMPORT_ERROR = import_error
else:
    _UNET_IMPORT_ERROR = None


def _extract_state_dict(checkpoint: object) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        logger.error("SuPreM checkpoint should be a mapping-like object.")

    for key in ("net", "state_dict", "model"):
        nested_state = checkpoint.get(key)
        if isinstance(nested_state, Mapping):
            checkpoint = nested_state
            break

    state_dict: dict[str, Tensor] = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            state_dict[str(key)] = value
    if not state_dict:
        logger.error("No tensor weights were found in the SuPreM checkpoint.")
    return state_dict


def _state_key_candidates(key: str) -> list[str]:
    clean_key = key.removeprefix("module.")
    candidates = [
        key,
        clean_key,
        clean_key.removeprefix("backbone."),
        clean_key.removeprefix("encoder."),
    ]

    parts = clean_key.split(".")
    if "backbone" in parts:
        backbone_idx = parts.index("backbone")
        candidates.append(".".join(parts[backbone_idx + 1 :]))
    if len(parts) > 2:
        candidates.append(".".join(parts[2:]))

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _filter_matching_weights(
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    matched_state: dict[str, Tensor] = {}
    for source_key, source_value in source_state.items():
        for target_key in _state_key_candidates(source_key):
            target_value = target_state.get(target_key)
            if target_value is not None and target_value.shape == source_value.shape:
                matched_state[target_key] = source_value
                break
    return matched_state


def _disable_unused_unet_decoder_gradients(backbone: nn.Module) -> None:
    encoder_prefixes = ("down_tr64.", "down_tr128.", "down_tr256.", "down_tr512.")
    frozen_count = 0
    for name, param in backbone.named_parameters():
        if not name.startswith(encoder_prefixes):
            param.requires_grad = False
            frozen_count += 1
    if frozen_count > 0:
        logger.log(
            f"Disabled gradients for {frozen_count} unused SuPreM U-Net decoder "
            "parameters in encoder-only classification mode."
        )


@MODEL_REGISTRY.register(name="suprem_unet_roi", type="classification")
class SuPreMUNetRoi(BaseAnyNNModel):
    """SuPreM U-Net backbone with a classification head for 3D esophagus ROI."""

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

        self.backbone = UNet3D()
        _disable_unused_unet_decoder_gradients(self.backbone)
        self.feature_channels = 512

        classifier_layers: list[nn.Module] = [
            GlobalPool(pool_type=pool_type, keep_dim=False)
        ]
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
        self._load_suprem_checkpoint(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.suprem-unet-roi.pretrained-path",
            type=str,
            default="",
            help="Path to supervised_suprem_unet_2100.pth.",
        )
        group.add_argument(
            "--model.classification.suprem-unet-roi.require-pretrained",
            action="store_true",
            default=False,
            help="Fail model construction when pretrained_path is empty.",
        )
        return parser

    def _load_suprem_checkpoint(self, opts: argparse.Namespace) -> None:
        pretrained_path = str(
            getattr(opts, "model.classification.suprem_unet_roi.pretrained_path", "")
        ).strip()
        require_pretrained = bool(
            getattr(opts, "model.classification.suprem_unet_roi.require_pretrained", False)
        )

        if not pretrained_path:
            if require_pretrained and not is_test_env():
                logger.error(
                    "SuPreM U-Net fine-tuning requires "
                    "--model.classification.suprem-unet-roi.pretrained-path."
                )
            logger.warning("Building SuPreM U-Net ROI model without pretrained weights.")
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
            "Loaded SuPreM U-Net weights from {}: matched={}, missing={}, unexpected={}".format(
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
        x = self.extract_features(x)
        return self.classifier(x)

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
