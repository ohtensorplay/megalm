from __future__ import annotations

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.kernels.vision_attention import (
    GroupedVarlenAttentionLayout,
    build_grouped_varlen_attention_layout,
    grouped_varlen_attention,
)
from megalm.models.oxr.config import OXRVisionConfig


def _frozen(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    return module


def _rotate_half(x: tensorplay.Tensor) -> tensorplay.Tensor:
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return tensorplay.cat((-second, first), dim=-1)


class OXRPatchEmbeddings(nn.Module):
    """SigLIP2 NAViT patch embedding plus resized 2-D position embeddings."""

    def __init__(
        self,
        config: OXRVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = _frozen(
            nn.Conv2d(
                config.num_channels * config.temporal_patch_size,
                config.hidden_size,
                kernel_size=config.patch_size,
                stride=config.patch_size,
                padding="valid",
                bias=True,
                device=device,
                dtype=dtype,
            )
        )
        self.position_embedding = _frozen(
            nn.Embedding(
                config.num_position_embeddings,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        self.position_size = config.position_embedding_size
        self.stride = config.hidden_stride

    def forward(
        self,
        pixel_values: tensorplay.Tensor,
        grid_thws: list[tuple[int, int, int]],
    ) -> tensorplay.Tensor:
        channels = self.config.num_channels * self.config.temporal_patch_size
        patches = pixel_values.reshape(-1, channels, self.config.patch_size, self.config.patch_size)
        patch_embeds = self.patch_embedding(patches).reshape(-1, self.config.hidden_size)

        source = (
            self.position_embedding.weight.reshape(
                self.position_size,
                self.position_size,
                self.config.hidden_size,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        # Bicubic resize is computed in fp32 for numerical parity with the
        # PyTorch reference; the result is cast back after interpolation.
        source = source.float()
        result = tensorplay.zeros_like(patch_embeds)
        offset = 0
        for temporal, height, width in grid_thws:
            count = temporal * height * width
            resized = F.interpolate(
                source,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
            resized = (
                resized.permute(0, 2, 3, 1)
                .reshape(height * width, self.config.hidden_size)
                .to(patch_embeds.dtype)
            )
            if temporal > 1:
                resized = resized.repeat(temporal, 1)
            resized = (
                resized.reshape(
                    temporal,
                    height // self.stride,
                    self.stride,
                    width // self.stride,
                    self.stride,
                    self.config.hidden_size,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(count, self.config.hidden_size)
            )
            result[offset : offset + count] = resized
            offset += count
        return patch_embeds + result


class OXRVisionAttention(nn.Module):
    def __init__(
        self,
        config: OXRVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.use_rope = config.use_rope
        self.q_proj = _frozen(
            nn.Linear(width, width, bias=True, device=device, dtype=dtype)
        )
        self.k_proj = _frozen(
            nn.Linear(width, width, bias=True, device=device, dtype=dtype)
        )
        self.v_proj = _frozen(
            nn.Linear(width, width, bias=True, device=device, dtype=dtype)
        )
        self.out_proj = _frozen(
            nn.Linear(width, width, bias=True, device=device, dtype=dtype)
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        layout: GroupedVarlenAttentionLayout,
        cos: tensorplay.Tensor,
        sin: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        tokens = hidden_states.shape[0]
        query = self.q_proj(hidden_states).reshape(tokens, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(tokens, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(tokens, self.num_heads, self.head_dim)
        if self.use_rope:
            broadcast_cos = cos.unsqueeze(1).to(query.dtype)
            broadcast_sin = sin.unsqueeze(1).to(query.dtype)
            query = query * broadcast_cos + _rotate_half(query) * broadcast_sin
            key = key * broadcast_cos + _rotate_half(key) * broadcast_sin
        attended = grouped_varlen_attention(
            query,
            key,
            value,
            layout,
            scale=self.scale,
        )
        return self.out_proj(attended.reshape(tokens, -1))


class OXRVisionMLP(nn.Module):
    def __init__(
        self,
        config: OXRVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.fc1 = _frozen(
            nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=True,
                device=device,
                dtype=dtype,
            )
        )
        self.fc2 = _frozen(
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
        activated = self.fc1(hidden_states)
        if self.activation == "gelu_pytorch_tanh":
            activated = F.gelu(activated, approximate="tanh")
        elif self.activation == "gelu":
            activated = F.gelu(activated)
        else:
            raise ValueError(f"unsupported OXR ViT activation {self.activation!r}")
        return self.fc2(activated)


class OXRVisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: OXRVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.layer_norm1 = _frozen(
            nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
                device=device,
                dtype=dtype,
            )
        )
        self.layer_norm2 = _frozen(
            nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
                device=device,
                dtype=dtype,
            )
        )
        self.self_attn = OXRVisionAttention(config, device=device, dtype=dtype)
        self.mlp = OXRVisionMLP(config, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        layout: GroupedVarlenAttentionLayout,
        cos: tensorplay.Tensor,
        sin: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states),
            layout=layout,
            cos=cos,
            sin=sin,
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class OXRVisualTokenizer(nn.Module):
    """ViT encoder + soft visual tokenizer + visual-token embedding."""

    def __init__(
        self,
        config: OXRVisionConfig,
        *,
        visual_vocab_size: int,
        hidden_size: int,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.embeddings = OXRPatchEmbeddings(config, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                OXRVisionEncoderLayer(config, device=device, dtype=dtype)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.post_layernorm = _frozen(
            nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
                device=device,
                dtype=dtype,
            )
        )
        head_dim = visual_vocab_size - 4
        self.head = _frozen(
            nn.Linear(
                config.hidden_size * config.vision_tokens_per_merge,
                head_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )
        )
        self.head_norm = _frozen(
            nn.LayerNorm(head_dim, device=device, dtype=dtype)
        )
        self.visual_embedding = _frozen(
            nn.Embedding(visual_vocab_size, hidden_size, device=device, dtype=dtype)
        )
        self.visual_vocab_size = visual_vocab_size
        self.head_dim = head_dim

    def _rotary(
        self,
        grid_thws: list[tuple[int, int, int]],
        *,
        device: tensorplay.device,
        tokens: int,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        stride = self.config.hidden_stride
        positions: list[tensorplay.Tensor] = []
        for _, height, width in grid_thws:
            height_ids = (
                tensorplay.arange(height, device=device)
                .unsqueeze(1)
                .expand(height, width)
                .reshape(height // stride, stride, width // stride, stride)
                .permute(0, 2, 1, 3)
                .reshape(-1)
            )
            width_ids = (
                tensorplay.arange(width, device=device)
                .unsqueeze(0)
                .expand(height, width)
                .reshape(height // stride, stride, width // stride, stride)
                .permute(0, 2, 1, 3)
                .reshape(-1)
            )
            positions.append(tensorplay.stack((height_ids, width_ids), dim=-1))
        position_ids = tensorplay.cat(positions, dim=0)
        rotary_dim = self.config.rotary_dim
        inv_freq = 1.0 / (
            10000.0
            ** (
                tensorplay.arange(
                    0,
                    rotary_dim,
                    2,
                    dtype=tensorplay.float32,
                    device=device,
                )
                / rotary_dim
            )
        )
        max_grid_size = max(max(height, width) for _, height, width in grid_thws)
        sequence = tensorplay.arange(
            max_grid_size, dtype=tensorplay.float32, device=device
        )
        frequencies = tensorplay.outer(sequence, inv_freq)
        gathered = frequencies.index_select(
            0, position_ids.reshape(-1).long()
        ).reshape(tokens, 2, rotary_dim // 2)
        rotary = gathered.reshape(tokens, rotary_dim)
        doubled = tensorplay.cat((rotary, rotary), dim=-1)
        return doubled.cos(), doubled.sin()

    def _window_index(
        self,
        grid_thws: list[tuple[int, int, int]],
        *,
        device: tensorplay.device,
    ) -> tensorplay.Tensor:
        stride = self.config.hidden_stride
        window = self.config.window_size // stride // self.config.patch_size
        if window <= 0:
            raise ValueError("OXR ViT window size is too small")
        chunks: list[tensorplay.Tensor] = []
        offset = 0
        for temporal, height, width in grid_thws:
            merged_h, merged_w = height // stride, width // stride
            index = tensorplay.arange(
                temporal * merged_h * merged_w, device=device
            ).reshape(temporal, merged_h, merged_w)
            pad_h = window - merged_h % window
            pad_w = window - merged_w % window
            padded = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
            padded = padded.reshape(
                temporal,
                (merged_h + pad_h) // window,
                window,
                (merged_w + pad_w) // window,
                window,
            ).permute(0, 1, 3, 2, 4)
            flat = padded.reshape(-1)
            kept = flat.index_select(0, (flat != -100).nonzero().reshape(-1))
            chunks.append(kept + offset)
            offset += temporal * merged_h * merged_w
        return tensorplay.cat(chunks, dim=0)

    @tensorplay.inference_mode()
    def forward(
        self,
        pixel_values: tensorplay.Tensor,
        grid_thws: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        pixel_values = pixel_values.to(
            device=self.visual_embedding.weight.device,
            dtype=self.visual_embedding.weight.dtype,
        )
        grid_thws = grid_thws.to(device=pixel_values.device)
        grids = [
            (int(row[0]), int(row[1]), int(row[2]))
            for row in grid_thws.tolist()
        ]
        if not grids:
            raise ValueError("OXR vision tower requires at least one grid")
        patch_count = int(pixel_values.shape[0])
        merge = self.config.vision_tokens_per_merge
        if patch_count % merge:
            raise ValueError("OXR patch count is not divisible by the merge stride")

        hidden = self.embeddings(pixel_values, grids)
        merged_count = patch_count // merge
        window_index = self._window_index(grids, device=hidden.device)
        reverse_index = window_index.argsort()
        hidden = hidden.reshape(merged_count, merge, -1)
        hidden = hidden[window_index.long()].reshape(patch_count, -1)

        cos, sin = self._rotary(grids, device=hidden.device, tokens=patch_count)
        # The rotary vectors are built at patch granularity; the window
        # permutation must be applied to them exactly as to the tokens.
        rotary_cos = cos.reshape(merged_count, merge, -1)[
            window_index.long()
        ].reshape(patch_count, -1)
        rotary_sin = sin.reshape(merged_count, merge, -1)[
            window_index.long()
        ].reshape(patch_count, -1)

        boundaries = [0]
        for temporal, height, width in grids:
            boundaries.append(boundaries[-1] + temporal * height * width)
        cu_seqlens = tensorplay.tensor(
            boundaries, dtype=tensorplay.int32, device=hidden.device
        )
        layout = build_grouped_varlen_attention_layout(cu_seqlens)
        for layer in self.layers:
            hidden = layer(
                hidden,
                layout=layout,
                cos=rotary_cos,
                sin=rotary_sin,
            )
        hidden = hidden.reshape(merged_count, merge, -1)
        hidden = hidden[reverse_index.long()].reshape(patch_count, -1)
        # The reference tokenizer consumes the last encoder hidden state
        # *before* the vision transformer's post-LayerNorm, so the head is
        # deliberately applied to the unnormalized output here.
        merged = hidden.reshape(merged_count, -1)
        logits = self.head_norm(self.head(merged))
        tokens = F.softmax(logits.float(), dim=-1).to(logits.dtype)
        padding = tensorplay.zeros(
            (tokens.shape[0], 4), dtype=tokens.dtype, device=tokens.device
        )
        tokens = tensorplay.cat((tokens, padding), dim=-1)
        return tensorplay.matmul(tokens, self.visual_embedding.weight)


__all__ = ["OXRVisionEncoderLayer", "OXRVisualTokenizer"]
