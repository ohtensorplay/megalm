from __future__ import annotations

import tensorplay
import tensorplay.nn as nn

from megalm.models.qwen2_5_omni.config import OmniTextConfig
from megalm.models.qwen2_5_omni.text_errors import OmniTextError


class OmniTextPositionError(OmniTextError):
    """Raised when Omni text positions violate the three-axis contract."""


class OmniMRoPE(nn.Module):
    def __init__(
        self,
        config: OmniTextConfig,
        *,
        device: tensorplay.device,
    ) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            config.rope_theta
            ** (
                tensorplay.arange(
                    0,
                    config.head_dim,
                    2,
                    dtype=tensorplay.float32,
                    device=device,
                )
                / config.head_dim
            )
        )
        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )
        self.sections = tuple(
            section * 2 for section in config.mrope_sections
        )

    def forward(
        self,
        positions: tensorplay.Tensor,
        *,
        dtype: tensorplay.dtype,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.shape[0] != 3:
            raise OmniTextPositionError(
                "Omni MRoPE requires three position axes"
            )
        frequencies = (
            positions.float().unsqueeze(-1)
            * self.inverse_frequency[None, None, :]
        )
        embeddings = tensorplay.cat((frequencies, frequencies), dim=-1)
        chunks = embeddings.split(self.sections, dim=-1)
        selected = tensorplay.cat(
            [chunk[index % 3] for index, chunk in enumerate(chunks)],
            dim=-1,
        )
        return selected.cos().to(dtype), selected.sin().to(dtype)

    @staticmethod
    def apply(
        query: tensorplay.Tensor,
        key: tensorplay.Tensor,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        cos, sin = position_embeddings
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        return (
            query * cos + _rotate_half(query) * sin,
            key * cos + _rotate_half(key) * sin,
        )


def _rotate_half(hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return tensorplay.cat((-second, first), dim=-1)


__all__ = ["OmniMRoPE", "OmniTextPositionError"]
