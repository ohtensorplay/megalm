from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings, KernelCallable
from megalm.models.components.parameters import frozen_parameter


class PackedSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        bias: bool,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernels: KernelBindings,
        linear: KernelCallable | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_weight = frozen_parameter(
            (2 * intermediate_size, hidden_size),
            device=device,
            dtype=dtype,
        )
        self.gate_up_bias = (
            frozen_parameter(
                (2 * intermediate_size,),
                device=device,
                dtype=dtype,
            )
            if bias
            else None
        )
        self.down_weight = frozen_parameter(
            (hidden_size, intermediate_size),
            device=device,
            dtype=dtype,
        )
        self.down_bias = (
            frozen_parameter(
                (hidden_size,),
                device=device,
                dtype=dtype,
            )
            if bias
            else None
        )
        self._silu_and_mul = kernels.core.silu_and_mul
        self._linear = F.linear if linear is None else linear

    def forward(self, hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
        gate_up = self._linear(
            hidden_states,
            self.gate_up_weight,
            self.gate_up_bias,
        )
        return self._linear(
            self._silu_and_mul(gate_up),
            self.down_weight,
            self.down_bias,
        )


__all__ = ["PackedSwiGLU"]
