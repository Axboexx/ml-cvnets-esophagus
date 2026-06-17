#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from torch import Tensor
from torch import rand

from cvnets.layers.base_layer import BaseLayer


class StochasticDepth(BaseLayer):
    """
    Drop residual branches during training without depending on torchvision.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Drop probability has to be in [0, 1], got {p}")
        if mode not in {"batch", "row"}:
            raise ValueError(f"Mode has to be 'batch' or 'row', got {mode}")
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        survival_rate = 1.0 - self.p
        if survival_rate == 0.0:
            return x.mul(0.0)

        if self.mode == "batch":
            mask_shape = [1] * x.ndim
        else:
            mask_shape = [x.shape[0]] + [1] * (x.ndim - 1)

        noise = rand(mask_shape, dtype=x.dtype, device=x.device)
        binary_mask = (noise < survival_rate).to(dtype=x.dtype)
        return x * binary_mask / survival_rate
