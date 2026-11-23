from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.kernels.vision_attention import (
    build_grouped_varlen_attention_layout,
)
from megalm.models.components import (
    frame_cu_seqlens,
    frozen_parameter,
    merged_vision_rotary,
    merged_window_index,
)
from megalm.models.qwen2_5_omni.config import OmniVisionConfig
from megalm.models.qwen2_5_omni.vision_layers import (
    OmniVisionBlock,
    OmniVisionMerger,
)


class Qwen25OmniVisionEncoder(nn.Module):
    """Qwen2.5-Omni ViT composition over reusable native components."""

    def __init__(
        self,
        config: OmniVisionConfig,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernels: KernelBindings,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.spatial_merge_size = config.spatial_merge_size
        patch_width = (
            config.in_channels
            * config.temporal_patch_size
            * config.patch_size**2
        )
        self.patch_embed_weight = frozen_parameter(
            (config.hidden_size, patch_width),
            device=self.device,
            dtype=dtype,
        )
        self.blocks = nn.ModuleList(
            [
                OmniVisionBlock(
                    config,
                    device=self.device,
                    dtype=dtype,
                    kernels=kernels,
                )
                for _ in range(config.depth)
            ]
        )
        self.merger = OmniVisionMerger(
            config,
            device=self.device,
            dtype=dtype,
            kernels=kernels,
        )
        rotary_dim = config.hidden_size // config.num_heads // 2
        inverse_frequency = 1.0 / (
            10_000.0
            ** (
                tensorplay.arange(
                    0,
                    rotary_dim,
                    2,
                    dtype=tensorplay.float32,
                    device=self.device,
                )
                / rotary_dim
            )
        )
        self.register_buffer(
            "rotary_inverse_frequency",
            inverse_frequency,
            persistent=False,
        )

    def forward(
        self,
        pixel_values: tensorplay.Tensor,
        grid_thw: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        _validate_grid(grid_thw, merge=self.spatial_merge_size)
        grid_thw = grid_thw.to(device=self.device)
        hidden_states = F.linear(
            pixel_values.to(device=self.device, dtype=self.dtype),
            self.patch_embed_weight,
        )
        rotary = merged_vision_rotary(
            grid_thw,
            self.rotary_inverse_frequency,
            spatial_merge_size=self.spatial_merge_size,
            device=self.device,
        )
        window_index, cu_window = merged_window_index(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            patch_size=self.config.patch_size,
            window_size=self.config.window_size,
            device=self.device,
        )
        merge_unit = self.spatial_merge_size**2
        token_count = hidden_states.shape[0]
        hidden_states = (
            hidden_states.view(-1, merge_unit, self.config.hidden_size)
            .index_select(0, window_index)
            .reshape(token_count, self.config.hidden_size)
        )
        rotary = (
            rotary.view(-1, merge_unit, rotary.shape[-1])
            .index_select(0, window_index)
            .reshape(token_count, rotary.shape[-1])
        )
        cu_full = frame_cu_seqlens(grid_thw)
        full_attention_layout = build_grouped_varlen_attention_layout(
            cu_full
        )
        window_attention_layout = build_grouped_varlen_attention_layout(
            cu_window
        )
        full_attention = set(self.config.full_attention_layers)
        for index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                attention_layout=(
                    full_attention_layout
                    if index in full_attention
                    else window_attention_layout
                ),
                rotary=rotary,
            )
        merged = self.merger(hidden_states)
        return merged.index_select(0, tensorplay.argsort(window_index))


def _validate_grid(grid_thw: tensorplay.Tensor, *, merge: int) -> None:
    if (
        grid_thw.ndim != 2
        or grid_thw.shape[1] != 3
        or grid_thw.dtype not in {tensorplay.int32, tensorplay.int64}
        or bool((grid_thw <= 0).any().item())
        or bool((grid_thw[:, 1:] % merge != 0).any().item())
    ):
        raise ValueError(
            "grid_thw must be positive merge-aligned integer [N, 3]"
        )


__all__ = ["Qwen25OmniVisionEncoder"]
