from __future__ import annotations

import math

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.vision_attention import (
    GroupedVarlenAttentionLayout,
    build_grouped_varlen_attention_layout,
)
from megalm.models.components import (
    PackedVisionAttention,
    frame_cu_seqlens,
    merged_vision_rotary,
)
from megalm.models.qwen3_vl.config import Qwen3VLVisionConfig


def _freeze(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    return module


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        kernel = (
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.proj = _freeze(
            nn.Conv3d(
                config.in_channels,
                config.hidden_size,
                kernel_size=kernel,
                stride=kernel,
                bias=True,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, pixel_values: tensorplay.Tensor) -> tensorplay.Tensor:
        pixel_values = pixel_values.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(pixel_values).view(-1, self.hidden_size)


class Qwen3VLVisionMLP(nn.Module):
    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.linear_fc1 = _freeze(
            nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=True,
                device=device,
                dtype=dtype,
            )
        )
        self.linear_fc2 = _freeze(
            nn.Linear(
                config.intermediate_size,
                config.hidden_size,
                bias=True,
                device=device,
                dtype=dtype,
            )
        )
        self.activation = config.hidden_act

    def forward(self, hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
        activated = (
            F.silu(self.linear_fc1(hidden_states))
            if self.activation == "silu"
            else F.gelu(
                self.linear_fc1(hidden_states),
                approximate="tanh",
            )
        )
        return self.linear_fc2(activated)


class Qwen3VLVisionBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.norm1 = _freeze(
            nn.LayerNorm(
                config.hidden_size,
                eps=1e-6,
                device=device,
                dtype=dtype,
            )
        )
        self.norm2 = _freeze(
            nn.LayerNorm(
                config.hidden_size,
                eps=1e-6,
                device=device,
                dtype=dtype,
            )
        )
        self.attn = PackedVisionAttention(
            config.hidden_size,
            config.num_heads,
            device=device,
            dtype=dtype,
        )
        self.mlp = Qwen3VLVisionMLP(
            config,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        attention_layout: GroupedVarlenAttentionLayout,
        rotary: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_layout=attention_layout,
            rotary=rotary,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3VLVisionMerger(nn.Module):
    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        use_postshuffle_norm: bool = False,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = (
            config.hidden_size * config.spatial_merge_size**2
        )
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_width = (
            self.hidden_size if use_postshuffle_norm else config.hidden_size
        )
        self.norm = _freeze(
            nn.LayerNorm(
                norm_width,
                eps=1e-6,
                device=device,
                dtype=dtype,
            )
        )
        self.linear_fc1 = _freeze(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                device=device,
                dtype=dtype,
            )
        )
        self.linear_fc2 = _freeze(
            nn.Linear(
                self.hidden_size,
                config.out_hidden_size,
                bias=True,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
        if self.use_postshuffle_norm:
            merged = hidden_states.view(-1, self.hidden_size)
            merged = self.norm(merged)
        else:
            merged = self.norm(hidden_states).view(-1, self.hidden_size)
        return self.linear_fc2(
            F.gelu(self.linear_fc1(merged), approximate="none")
        )


class Qwen3VLVisionTransformer(nn.Module):
    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.num_grid_per_side = math.isqrt(
            config.num_position_embeddings
        )
        if self.num_grid_per_side**2 != config.num_position_embeddings:
            raise ValueError(
                "vision position embedding count must be a square"
            )
        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config,
            device=device,
            dtype=dtype,
        )
        self.pos_embed = _freeze(
            nn.Embedding(
                config.num_position_embeddings,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(config, device=device, dtype=dtype)
                for _ in range(config.depth)
            ]
        )
        self.merger = Qwen3VLVisionMerger(
            config,
            device=device,
            dtype=dtype,
        )
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionMerger(
                    config,
                    use_postshuffle_norm=True,
                    device=device,
                    dtype=dtype,
                )
                for _ in config.deepstack_visual_indexes
            ]
        )
        head_dim = config.hidden_size // config.num_heads
        inverse_frequency = 1.0 / (
            10_000.0
            ** (
                tensorplay.arange(
                    0,
                    head_dim // 2,
                    2,
                    dtype=tensorplay.float32,
                    device=device,
                )
                / (head_dim // 2)
            )
        )
        self.register_buffer(
            "rotary_inverse_frequency",
            inverse_frequency,
            persistent=False,
        )

    @property
    def dtype(self) -> tensorplay.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> tensorplay.device:
        return self.patch_embed.proj.weight.device

    @property
    def deepstack_depth(self) -> int:
        return len(self.deepstack_merger_list)

    def forward(
        self,
        pixel_values: tensorplay.Tensor,
        grid_thw: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("grid_thw must have shape [N, 3]")
        hidden_states = self.patch_embed(
            pixel_values.to(device=self.device, dtype=self.dtype)
        )
        hidden_states = hidden_states + self._position_embedding(grid_thw)
        rotary = merged_vision_rotary(
            grid_thw,
            self.rotary_inverse_frequency,
            spatial_merge_size=self.spatial_merge_size,
            device=self.device,
        )
        cu_seqlens = frame_cu_seqlens(
            grid_thw.to(device=self.device)
        )
        attention_layout = build_grouped_varlen_attention_layout(
            cu_seqlens
        )
        deepstack_features: list[tensorplay.Tensor] = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                attention_layout=attention_layout,
                rotary=rotary,
            )
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(
                    layer_index
                )
                deepstack_features.append(
                    self.deepstack_merger_list[merger_index](hidden_states)
                )
        outputs = [self.merger(hidden_states), *deepstack_features]
        return tensorplay.cat(outputs, dim=1)

    def _position_embedding(
        self,
        grid_thw: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        outputs: list[tensorplay.Tensor] = []
        side = self.num_grid_per_side
        merge = self.spatial_merge_size
        for temporal, height, width in grid_thw.tolist():
            h = tensorplay.linspace(0, side - 1, height, device=self.device)
            w = tensorplay.linspace(0, side - 1, width, device=self.device)
            h_floor = h.long()
            w_floor = w.long()
            h_ceil = (h_floor + 1).clamp_max(side - 1)
            w_ceil = (w_floor + 1).clamp_max(side - 1)
            dh = h - h_floor
            dw = w - w_floor
            indices = (
                h_floor[:, None] * side + w_floor[None, :],
                h_floor[:, None] * side + w_ceil[None, :],
                h_ceil[:, None] * side + w_floor[None, :],
                h_ceil[:, None] * side + w_ceil[None, :],
            )
            weights = (
                (1 - dh)[:, None] * (1 - dw)[None, :],
                (1 - dh)[:, None] * dw[None, :],
                dh[:, None] * (1 - dw)[None, :],
                dh[:, None] * dw[None, :],
            )
            position = sum(
                self.pos_embed(index) * weight[..., None].to(self.dtype)
                for index, weight in zip(indices, weights)
            )
            position = (
                position.view(
                    height // merge,
                    merge,
                    width // merge,
                    merge,
                    -1,
                )
                .permute(0, 2, 1, 3, 4)
                .reshape(height * width, -1)
                .repeat(temporal, 1)
            )
            outputs.append(position)
        return tensorplay.cat(outputs, dim=0)
