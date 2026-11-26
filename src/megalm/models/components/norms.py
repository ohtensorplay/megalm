"""Model-agnostic RMSNorm variants shared by towers and connectors."""

from __future__ import annotations

import tensorplay
from tensorplay import nn

from megalm.kernels.bindings import KernelBindings
from megalm.models.components.parameters import frozen_parameter


class RMSNorm(nn.Module):
    """RMS normalisation with an optional learnable per-channel scale."""

    def __init__(
        self,
        width: int,
        eps: float,
        *,
        with_scale: bool = True,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.weight = (
            frozen_parameter((width,), device=device, dtype=dtype)
            if with_scale
            else None
        )
        self.eps = eps
        self._rms_norm = kernel_bindings.core.rms_norm
        self._unit_weight = tensorplay.ones((width,), dtype=dtype, device=device)

    def forward(self, value: tensorplay.Tensor) -> tensorplay.Tensor:
        weight = self.weight if self.weight is not None else self._unit_weight
        return self._rms_norm(value, weight, self.eps)


__all__ = ["RMSNorm"]
