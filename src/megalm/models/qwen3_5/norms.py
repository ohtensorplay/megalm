from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings


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


class OffsetRMSNorm(nn.Module):
    """Gemma-style RMSNorm used by the Qwen3.5 text backbone."""

    def __init__(
        self,
        width: int,
        eps: float,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.weight = frozen_parameter(
            (width,),
            device=device,
            dtype=dtype,
        )
        self.eps = eps
        self._offset_rms_norm = kernel_bindings.core.offset_rms_norm
        self._fused_add_offset_rms_norm = (
            kernel_bindings.core.fused_add_offset_rms_norm
        )

    def forward(self, hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
        return self._offset_rms_norm(
            hidden_states,
            self.weight,
            self.eps,
        )

    def with_residual(
        self,
        hidden_states: tensorplay.Tensor,
        residual: tensorplay.Tensor,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        return self._fused_add_offset_rms_norm(
            hidden_states,
            residual,
            self.weight,
            self.eps,
        )


class GatedRMSNorm(nn.Module):
    """Per-value-head norm-before-SiLU gate used after the delta rule."""

    def __init__(
        self,
        width: int,
        eps: float,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.weight = frozen_parameter(
            (width,),
            device=device,
            dtype=tensorplay.float32,
        )
        self.eps = eps

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        gate: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        normalized = hidden_states.float()
        normalized = normalized * tensorplay.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized * self.weight.float()
        return (normalized * F.silu(gate.float())).to(hidden_states.dtype)
