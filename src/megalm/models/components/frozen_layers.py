from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.models.components.parameters import frozen_parameter


class FrozenLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.weight = frozen_parameter(
            (out_features, in_features),
            device=device,
            dtype=dtype,
        )
        self.bias = (
            frozen_parameter(
                (out_features,),
                device=device,
                dtype=dtype,
            )
            if bias
            else None
        )

    def forward(self, inputs: tensorplay.Tensor) -> tensorplay.Tensor:
        return F.linear(inputs, self.weight, self.bias)


class FrozenConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = frozen_parameter(
            (out_channels, in_channels, kernel_size),
            device=device,
            dtype=dtype,
        )
        self.bias = frozen_parameter(
            (out_channels,),
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs: tensorplay.Tensor) -> tensorplay.Tensor:
        return F.conv1d(
            inputs,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class FrozenLayerNorm(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        eps: float,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.normalized_shape = (width,)
        self.eps = eps
        self.weight = frozen_parameter(
            (width,),
            device=device,
            dtype=dtype,
        )
        self.bias = frozen_parameter(
            (width,),
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs: tensorplay.Tensor) -> tensorplay.Tensor:
        return F.layer_norm(
            inputs,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )


__all__ = ["FrozenConv1d", "FrozenLayerNorm", "FrozenLinear"]
