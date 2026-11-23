from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.models.qwen3 import SparseMoE, WeightOnlyLinear
from megalm.models.qwen3_5.config import Qwen35TextConfig
from megalm.models.qwen3_5.norms import frozen_parameter


class Qwen35SharedExpertMLP(nn.Module):
    """Dense SiLU MLP used as the always-on expert of Qwen3.5-MoE layers."""

    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        intermediate = config.shared_expert_intermediate_size
        if intermediate <= 0:
            raise ValueError("Qwen3.5 shared expert requires a positive width")
        self.gate_up_weight = frozen_parameter(
            (2 * intermediate, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.down_proj_weight = frozen_parameter(
            (config.hidden_size, intermediate),
            device=device,
            dtype=dtype,
        )
        self._silu_and_mul = kernel_bindings.core.silu_and_mul

    def forward(self, hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
        gate_up = F.linear(hidden_states, self.gate_up_weight)
        return F.linear(
            self._silu_and_mul(gate_up),
            self.down_proj_weight,
        )


class Qwen35SparseMoeBlock(SparseMoE):
    """Qwen3.5 routed experts plus the sigmoid-gated shared expert."""

    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__(
            config,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen35SharedExpertMLP(
                config,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
            self.shared_expert_gate = WeightOnlyLinear(
                1,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        decode: bool,
    ) -> tensorplay.Tensor:
        routed = super().forward(hidden_states, decode=decode)
        if self.shared_expert is None or self.shared_expert_gate is None:
            return routed
        shared_gate = tensorplay.sigmoid(
            self.shared_expert_gate(hidden_states).float()
        ).to(routed.dtype)
        return routed + shared_gate * self.shared_expert(hidden_states)


__all__ = ["Qwen35SharedExpertMLP", "Qwen35SparseMoeBlock"]
