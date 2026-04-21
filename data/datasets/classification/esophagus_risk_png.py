#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

import argparse

from data.datasets import DATASET_REGISTRY
from data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)


@DATASET_REGISTRY.register(name="esophagus_risk_png", type="classification")
class EsophagusRiskPngDataset(BaseImageClassificationDataset):
    """ImageFolder-based dataset for esophagus risk classification."""

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        BaseImageClassificationDataset.__init__(
            self,
            opts=opts,
            *args,
            **kwargs,
        )
