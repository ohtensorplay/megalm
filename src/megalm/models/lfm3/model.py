from __future__ import annotations

import math
from pathlib import Path

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings, bind_kernel_plan
from megalm.kernels.ops import KernelPolicy
from megalm.loading import SafetensorsSource, WeightLoadReport
from megalm.models.lfm3.config import Lfm3Config
from megalm.models.qwen3 import RMSNorm, TokenEmbedding
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class Lfm3Error(RuntimeError):
    """Raised when an LFM3 checkpoint violates its native contract."""


def _parameter(
    shape: tuple[int, ...],
    *,
    device: tensorplay.device,
    dtype: tensorplay.dtype,
) -> nn.Parameter:
    return nn.Parameter(
        tensorplay.empty(shape, device=device, dtype=dtype),
        requires_grad=False,
    )


class Lfm3RotaryCache(nn.Module):
    """Precomputed cos/sin tables for LFM3's two rotary flavours.

    Full-attention layers use partial rotation with a zero tail (the tail
    channels are simply not rotated); sliding-attention layers rotate the
    whole head. Both tables share the neox channel layout expected by the
    ``rope_cache_write`` kernel.
    """

    def __init__(
        self,
        config: Lfm3Config,
        *,
        kind: str,
        device: tensorplay.device,
    ) -> None:
        super().__init__()
        head_dim = config.head_dim
        half = head_dim // 2
        if kind == "full_attention":
            theta = config.rope_theta_full
            partial = config.partial_rotary_factor
        else:
            theta = config.rope_theta_sliding
            partial = 1.0
        angles = int(partial * head_dim) // 2
        exponents = tensorplay.arange(
            0,
            2 * angles,
            2,
            dtype=tensorplay.float32,
            device=device,
        )
        inverse = 1.0 / (theta ** (exponents / head_dim))
        if angles < half:
            zeros = tensorplay.zeros(
                (half - angles,), dtype=tensorplay.float32, device=device
            )
            inverse = tensorplay.cat((inverse, zeros), dim=0)
        positions = tensorplay.arange(
            config.max_position_embeddings,
            dtype=tensorplay.float32,
            device=device,
        )
        frequencies = tensorplay.outer(positions, inverse)
        self.register_buffer("cos", frequencies.cos().contiguous(), persistent=False)
        self.register_buffer("sin", frequencies.sin().contiguous(), persistent=False)


class Lfm3Attention(nn.Module):
    """LFM3 attention: QK-RMSNorm, RoPE, sigmoid head gate, optional sliding."""

    def __init__(
        self,
        config: Lfm3Config,
        *,
        layer_index: int,
        kv_layer_index: int,
        rotary: Lfm3RotaryCache,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.kv_layer_index = kv_layer_index
        self.query_width = config.num_attention_heads * config.head_dim
        self.kv_width = config.num_key_value_heads * config.head_dim
        self.num_query_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_per_kv = self.num_query_heads // self.num_kv_heads
        self.sliding_window = (
            config.sliding_window if config.layer_types[layer_index] == "sliding_attention" else None
        )
        self.qkv_proj = _parameter(
            (self.query_width + 2 * self.kv_width, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.o_proj = _parameter(
            (config.hidden_size, self.query_width), device=device, dtype=dtype
        )
        self.q_norm = RMSNorm(
            config.head_dim,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.k_norm = RMSNorm(
            config.head_dim,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.gate_proj = _parameter(
            (config.num_attention_heads, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.rotary = rotary
        self.rope_cache_write = kernel_bindings.core.rope_cache_write

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        token_count = hidden_states.shape[0]
        qkv = F.linear(hidden_states, self.qkv_proj)
        query, key, value = qkv.split(
            (self.query_width, self.kv_width, self.kv_width), dim=-1
        )
        query = query.view(-1, self.num_query_heads, self.head_dim)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        key_cache, value_cache = cache.layer(self.kv_layer_index)
        query = self.rope_cache_write(
            query,
            key,
            value,
            batch.positions,
            batch.slot_mapping,
            self.rotary.cos,
            self.rotary.sin,
            key_cache,
            value_cache,
            batch.token_active,
        )
        attended = self._attention(query, key_cache, value_cache, batch)
        gate = tensorplay.sigmoid(
            F.linear(hidden_states, self.gate_proj)
        ).unsqueeze(-1)
        attended = attended * gate.expand(-1, -1, self.head_dim)
        return F.linear(attended.reshape(token_count, -1), self.o_proj)

    def _attention(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        output = tensorplay.empty_like(query)
        page_size = batch.page_size
        for sequence_index in range(batch.num_sequences):
            start = batch.query_start_loc[sequence_index]
            end = batch.query_start_loc[sequence_index + 1]
            query_length = end - start
            context_length = batch.context_lengths[sequence_index]
            sequence_length = context_length + query_length
            num_pages = math.ceil(sequence_length / page_size)
            page_ids = batch.block_tables[sequence_index, :num_pages].long()
            keys = key_cache.index_select(0, page_ids).reshape(
                -1, self.num_kv_heads, self.head_dim
            )[:sequence_length]
            values = value_cache.index_select(0, page_ids).reshape(
                -1, self.num_kv_heads, self.head_dim
            )[:sequence_length]
            if self.sliding_window is not None:
                first = max(0, sequence_length - self.sliding_window)
                keys = keys[first:]
                values = values[first:]
            else:
                first = 0
            keys = keys.repeat_interleave(self.query_per_kv, dim=1)
            values = values.repeat_interleave(self.query_per_kv, dim=1)
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = keys.transpose(0, 1).unsqueeze(0)
            v = values.transpose(0, 1).unsqueeze(0)
            if query_length > 1:
                q_positions = tensorplay.arange(
                    sequence_length - query_length,
                    sequence_length,
                    device=query.device,
                )
                k_positions = tensorplay.arange(
                    first, sequence_length, device=query.device
                )
                causal = (
                    k_positions[None, :] <= q_positions[:, None]
                )[None, None, :, :]
            else:
                causal = None
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal,
                dropout_p=0.0,
                scale=self.head_dim**-0.5,
            )
            output[start:end] = attended.squeeze(0).transpose(0, 1)
        return output


class Lfm3ShortConv(nn.Module):
    """LFM2-style gated depthwise short convolution with per-request state."""

    def __init__(
        self,
        config: Lfm3Config,
        *,
        layer_index: int,
        state_slots: int,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.kernel_size = config.conv_L_cache
        self.hidden_size = config.hidden_size
        self.in_proj = _parameter(
            (3 * config.hidden_size, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.out_proj = _parameter(
            (config.hidden_size, config.hidden_size), device=device, dtype=dtype
        )
        self.conv_weight = _parameter(
            (config.hidden_size, self.kernel_size), device=device, dtype=dtype
        )
        self.conv_bias = (
            _parameter((config.hidden_size,), device=device, dtype=dtype)
            if config.conv_bias
            else None
        )
        self.state = tensorplay.zeros(
            (state_slots, config.hidden_size, self.kernel_size),
            dtype=dtype,
            device=device,
        )
        self.rms_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        projected = F.linear(hidden_states, self.in_proj)
        b, c, x = projected.chunk(3, dim=-1)
        activated = b * x
        output = tensorplay.empty_like(activated)
        slots = batch.state_slots
        if slots is None:
            raise Lfm3Error("LFM3 short convolution requires state slots")
        if batch.all_decode:
            previous = self.state.index_select(0, slots)
            window = tensorplay.cat(
                (previous, activated.unsqueeze(-1)), dim=-1
            )
            values = window[..., 1:]
            result = (values * self.conv_weight.unsqueeze(0)).sum(dim=-1)
            if self.conv_bias is not None:
                result = result + self.conv_bias
            new_state = values
            self.state.index_copy_(0, slots, new_state)
            output.copy_(result)
        else:
            offset = 0
            for sequence_index in range(batch.num_sequences):
                length = batch.query_lengths[sequence_index]
                segment = activated[offset : offset + length]
                slot = slots[sequence_index : sequence_index + 1]
                previous = self.state.index_select(0, slot)
                window = tensorplay.cat(
                    (previous, segment.transpose(0, 1).unsqueeze(0)), dim=-1
                )
                convolved = F.conv1d(
                    window,
                    self.conv_weight.unsqueeze(1),
                    self.conv_bias,
                    groups=self.hidden_size,
                )
                result = convolved[..., -length:].squeeze(0).transpose(0, 1)
                output[offset : offset + length] = result
                self.state.index_copy_(0, slot, window[..., -self.kernel_size :])
                offset += length
        return F.linear(c * output, self.out_proj)


class Lfm3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Lfm3Config,
        *,
        layer_index: int,
        kv_layer_index: int | None,
        state_slots: int,
        rotary_full: Lfm3RotaryCache,
        rotary_sliding: Lfm3RotaryCache,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.is_attention = config.layer_types[layer_index] != "conv"
        self.use_sandwich_norm = config.use_sandwich_norm
        self.operator_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        if self.is_attention:
            assert kv_layer_index is not None
            rotary = (
                rotary_full
                if config.layer_types[layer_index] == "full_attention"
                else rotary_sliding
            )
            self.operator = Lfm3Attention(
                config,
                layer_index=layer_index,
                kv_layer_index=kv_layer_index,
                rotary=rotary,
                kernel_bindings=kernel_bindings,
                device=device,
                dtype=dtype,
            )
        else:
            self.operator = Lfm3ShortConv(
                config,
                layer_index=layer_index,
                state_slots=state_slots,
                kernel_bindings=kernel_bindings,
                device=device,
                dtype=dtype,
            )
        self.w1 = _parameter(
            (config.intermediate_size, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.w3 = _parameter(
            (config.intermediate_size, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.w2 = _parameter(
            (config.hidden_size, config.intermediate_size),
            device=device,
            dtype=dtype,
        )
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
            self.post_ffn_norm = RMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        residual = hidden_states
        hidden_states = self.operator_norm(hidden_states)
        if self.is_attention:
            hidden_states = self.operator(hidden_states, batch=batch, cache=cache)
        else:
            hidden_states = self.operator(hidden_states, batch=batch)
        if self.use_sandwich_norm:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        gate = F.linear(hidden_states, self.w1)
        up = F.linear(hidden_states, self.w3)
        activated = F.silu(gate) * up
        feed_forward = F.linear(activated, self.w2)
        if self.use_sandwich_norm:
            feed_forward = self.post_ffn_norm(feed_forward)
        return residual + feed_forward


class Lfm3ForCausalLM(nn.Module):
    """Native MegaLM executor for LiquidAI's LFM3 tri-hybrid architecture."""

    graph_capable = False

    def __init__(
        self,
        config: Lfm3Config,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
        max_cache_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.kernel_bindings = kernel_bindings or bind_kernel_plan(
            kernel_policy, device=self.device, dtype=dtype, model_type="lfm3"
        )
        self.embed_tokens = TokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            device=self.device,
            dtype=dtype,
        )
        self.rotary_full = Lfm3RotaryCache(
            config, kind="full_attention", device=self.device
        )
        self.rotary_sliding = Lfm3RotaryCache(
            config, kind="sliding_attention", device=self.device
        )
        self.state_slots = max(32, max_cache_tokens)
        self.layers = nn.ModuleList()
        for layer_index in range(config.num_hidden_layers):
            self.layers.append(
                Lfm3DecoderLayer(
                    config,
                    layer_index=layer_index,
                    kv_layer_index=(
                        layer_index
                        if config.layer_types[layer_index] != "conv"
                        else None
                    ),
                    state_slots=self.state_slots,
                    rotary_full=self.rotary_full,
                    rotary_sliding=self.rotary_sliding,
                    kernel_bindings=self.kernel_bindings,
                    device=self.device,
                    dtype=dtype,
                )
            )
        self.embedding_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=self.device,
            dtype=dtype,
            kernel_bindings=self.kernel_bindings,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
        max_cache_tokens: int = 512,
    ) -> tuple[Lfm3ForCausalLM, WeightLoadReport]:
        config = Lfm3Config.from_path(model_path)
        model = cls(
            config,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
            max_cache_tokens=max_cache_tokens,
        )
        report = model.load_weights(model_path)
        model.eval()
        return model, report

    def load_weights(self, model_path: str | Path) -> WeightLoadReport:
        source = SafetensorsSource(model_path)
        loaded: set[str] = set()

        def consume(name: str, tensor: tensorplay.Tensor) -> None:
            local = tensor.to(device=self.device, dtype=self.dtype)
            target: tensorplay.Tensor
            if name == "model.embed_tokens.weight":
                target = self.embed_tokens.weight
            elif name == "model.embedding_norm.weight":
                target = self.embedding_norm.weight
            else:
                parts = name.split(".")
                if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
                    return
                layer_index = int(parts[2])
                leaf = ".".join(parts[3:])
                layer = self.layers[layer_index]
                if leaf == "operator_norm.weight":
                    target = layer.operator_norm.weight
                elif leaf == "ffn_norm.weight":
                    target = layer.ffn_norm.weight
                elif leaf == "post_operator_norm.weight":
                    target = layer.post_operator_norm.weight
                elif leaf == "post_ffn_norm.weight":
                    target = layer.post_ffn_norm.weight
                elif leaf == "feed_forward.w1.weight":
                    target = layer.w1
                elif leaf == "feed_forward.w2.weight":
                    target = layer.w2
                elif leaf == "feed_forward.w3.weight":
                    target = layer.w3
                elif layer.is_attention:
                    attention = layer.operator
                    assert isinstance(attention, Lfm3Attention)
                    if leaf == "self_attn.q_proj.weight":
                        target = attention.qkv_proj[: attention.query_width]
                    elif leaf == "self_attn.k_proj.weight":
                        target = attention.qkv_proj[
                            attention.query_width : attention.query_width
                            + attention.kv_width
                        ]
                    elif leaf == "self_attn.v_proj.weight":
                        target = attention.qkv_proj[
                            attention.query_width + attention.kv_width :
                        ]
                    elif leaf == "self_attn.o_proj.weight":
                        target = attention.o_proj
                    elif leaf == "self_attn.q_norm.weight":
                        target = attention.q_norm.weight
                    elif leaf == "self_attn.k_norm.weight":
                        target = attention.k_norm.weight
                    elif leaf == "self_attn.gate_proj.weight":
                        target = attention.gate_proj
                    else:
                        return
                else:
                    conv = layer.operator
                    assert isinstance(conv, Lfm3ShortConv)
                    if leaf == "conv.in_proj.weight":
                        target = conv.in_proj
                    elif leaf == "conv.out_proj.weight":
                        target = conv.out_proj
                    elif leaf == "conv.conv.weight":
                        target = conv.conv_weight
                        local = local.view(conv.hidden_size, conv.kernel_size)
                    elif leaf == "conv.conv.bias":
                        assert conv.conv_bias is not None
                        target = conv.conv_bias
                    else:
                        return
            if tuple(target.shape) != tuple(local.shape):
                raise Lfm3Error(
                    f"shape mismatch for {name}: {tuple(local.shape)} vs "
                    f"{tuple(target.shape)}"
                )
            target.copy_(local)
            loaded.add(name)

        report = source.stream(consume, device="cpu")
        expected = {info.name for info in source.manifest}
        missing = expected - loaded
        if missing:
            raise Lfm3Error(
                f"LFM3 checkpoint did not load {len(missing)} tensors: "
                f"{sorted(missing)[:5]}"
            )
        return report

    def warmup_prefill_kernels(
        self,
        *,
        max_batched_tokens: int,
    ) -> tuple[int, ...]:
        """LFM3 has no MoE prefill kernels to warm up."""

        if max_batched_tokens <= 0:
            raise Lfm3Error("prefill warmup token budget must be positive")
        return ()

    def configure_moe_workspace(self, max_batched_tokens: int) -> int:
        """LFM3 is dense; there is no MoE scratch arena to allocate."""

        if max_batched_tokens <= 0:
            raise Lfm3Error("workspace token budget must be positive")
        return 0

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        hidden_states = self.embed_tokens(batch.input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, batch=batch, cache=cache)
        hidden_states = self.embedding_norm(hidden_states)
        indices = batch.sample_indices
        if indices is not None:
            hidden_states = hidden_states.index_select(0, indices)
        return F.linear(hidden_states, self.embed_tokens.weight)
