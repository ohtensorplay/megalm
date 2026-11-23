from __future__ import annotations

import tensorplay
import tensorplay.nn as nn


def frozen_parameter(
    shape: tuple[int, ...],
    *,
    device: tensorplay.device,
    dtype: tensorplay.dtype,
) -> nn.Parameter:
    return nn.Parameter(
        tensorplay.empty(shape, device=device, dtype=dtype),
        requires_grad=False,
    )


__all__ = ["frozen_parameter"]
