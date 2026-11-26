"""Native DiffusionGemma block-diffusion executor.

DiffusionGemma is a Gemma-4 backbone trained as a *block diffusion* model:

* an **encoder** pass runs causally over the committed prefix and fills a KV
  cache, exactly like an autoregressive prefill;
* a **decoder** pass denoises a fixed ``canvas_length`` block with
  bidirectional attention over ``[prefix KV, canvas KV]``; it reads the
  encoder cache but never writes it;
* the decoder is conditioned on the previous step's logits through a small
  soft-embedding MLP (self-conditioning), which is what makes iterative
  denoising converge.

The checkpoint stores a single weight stack under ``model.decoder.*`` and the
encoder reuses it.  This module resolves that stack into native TensorPlay
kernels, including the ``compressed-tensors`` W4A16 layout of the 26B release
(int4 attention projections and int4 experts, bf16 dense MLP/router).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.kernels.bindings import KernelBindings, bind_kernel_plan
from megalm.kernels.ops import KernelPolicy
from megalm.kernels.silu_mul import gelu_tanh_and_mul
from megalm.loading import SafetensorsSource, WeightLoadReport
from megalm.models.components.int4_linear import Int4GroupLinear
from megalm.models.diffusiongemma.config import (
    DECODER_PREFIX,
    DiffusionGemmaConfig,
)
from megalm.models.gemma4.attention import gelu_tanh
from megalm.models.gemma4.config import FULL_ATTENTION
from megalm.models.gemma4.model import Gemma4RMSNorm

_NEG_INF = float("-inf")
# Encoder prefill is chunked so a long prompt never materialises a
# ``[heads, prompt, prompt]`` score tensor (1 GB+ at 4k on the 26B geometry).
_ENCODER_QUERY_BLOCK = 256


class DiffusionGemmaError(RuntimeError):
    """Raised when a checkpoint or a generation request violates the contract."""


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


def _rms_norm(
    value: tensorplay.Tensor,
    weight: tensorplay.Tensor | None,
    eps: float,
) -> tensorplay.Tensor:
    """Gemma RMSNorm: unit-RMS in fp32, optional learned scale."""

    flat = value.float()
    mean = flat.pow(2).mean(-1, keepdim=True) + eps
    normalized = flat * mean.rsqrt()
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized.to(value.dtype)


class _RotaryEmbedding(nn.Module):
    """RoPE frequencies for one layer flavour, materialised per position.

    Gemma-4 uses neox-style split-half rotation.  Global layers use
    *proportional* RoPE: only ``partial_rotary_factor`` of the head rotates
    and the remaining inverse frequencies are zero (so cos=1, sin=0).
    Storing only ``inv_freq`` keeps the 262k-position table out of GPU memory.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        partial: float,
        theta: float,
        rope_type: str,
        device: tensorplay.device,
    ) -> None:
        super().__init__()
        half = head_dim // 2
        angles = int(head_dim * partial // 2)
        if rope_type == "proportional":
            exponents = tensorplay.arange(
                0, 2 * angles, 2, dtype=tensorplay.float32, device=device
            ) / float(head_dim)
        else:
            exponents = tensorplay.arange(
                0, 2 * half, 2, dtype=tensorplay.float32, device=device
            ) / float(head_dim)
        inverse = 1.0 / (theta**exponents)
        if angles < half:
            inverse = tensorplay.cat(
                (
                    inverse,
                    tensorplay.zeros(
                        (half - angles,), dtype=tensorplay.float32, device=device
                    ),
                ),
                dim=0,
            )
        self.register_buffer("inv_freq", inverse.contiguous(), persistent=False)
        self.half = half

    def forward(
        self, positions: tensorplay.Tensor
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        freqs = tensorplay.outer(positions.float(), self.inv_freq)
        return freqs.cos(), freqs.sin()


def _apply_rope(
    value: tensorplay.Tensor,
    cos: tensorplay.Tensor,
    sin: tensorplay.Tensor,
) -> tensorplay.Tensor:
    """Rotate ``[tokens, heads, head_dim]`` with ``[tokens, head_dim // 2]`` tables."""

    half = value.shape[-1] // 2
    first, second = value[..., :half], value[..., half:]
    cos = cos.to(value.dtype).unsqueeze(1)
    sin = sin.to(value.dtype).unsqueeze(1)
    return tensorplay.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    )


class DiffusionKVCache:
    """Contiguous per-layer encoder cache for one sequence.

    The decoder reads it and appends nothing; only :meth:`DiffusionGemmaModel.encode`
    writes into it.  ``capacity`` covers the prompt plus every canvas that the
    request may append.
    """

    __slots__ = ("keys", "values", "length", "capacity")

    def __init__(
        self,
        keys: list[tensorplay.Tensor],
        values: list[tensorplay.Tensor],
        capacity: int,
    ) -> None:
        self.keys = keys
        self.values = values
        self.length = 0
        self.capacity = capacity

    @classmethod
    def allocate(
        cls,
        model: "DiffusionGemmaModel",
        capacity: int,
    ) -> "DiffusionKVCache":
        if capacity <= 0:
            raise DiffusionGemmaError("KV cache capacity must be positive")
        config = model.config.text
        keys: list[tensorplay.Tensor] = []
        values: list[tensorplay.Tensor] = []
        for layer in range(config.num_hidden_layers):
            heads = config.layer_num_key_value_heads[layer]
            head_dim = config.layer_head_dims[layer]
            keys.append(
                tensorplay.empty(
                    (capacity, heads, head_dim),
                    dtype=model.dtype,
                    device=model.device,
                )
            )
            values.append(
                tensorplay.empty(
                    (capacity, heads, head_dim),
                    dtype=model.dtype,
                    device=model.device,
                )
            )
        return cls(keys, values, capacity)

    def nbytes(self) -> int:
        return sum(
            int(tensor.numel()) * tensor.element_size()
            for tensor in (*self.keys, *self.values)
        )

    def reset(self) -> None:
        self.length = 0


class _Projection(nn.Module):
    """Dense bf16 or weight-only int4 projection with one call signature."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        quant: Int4GroupLinear | None,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        marlin: bool = False,
    ) -> None:
        super().__init__()
        self.int4 = quant
        self.marlin = False
        if marlin and quant is not None:
            from megalm.kernels.marlin_moe import supports_marlin_moe

            self.marlin = supports_marlin_moe(
                device, dtype, out_features, in_features, 1, quant.group_size
            )
        if quant is None:
            self.weight = _parameter(
                (out_features, in_features), device=device, dtype=dtype
            )
        else:
            self.weight = None

    def load_compressed(self, packed: tensorplay.Tensor, scale: tensorplay.Tensor) -> None:
        assert self.int4 is not None
        self.int4.load_compressed(packed, scale)
        if not self.marlin or self.int4.marlin_q is not None:
            return
        from megalm.kernels.marlin_moe import pack_experts

        scale_by_group = self.int4.scale.t()
        metadata = tensorplay.stack((scale_by_group, scale_by_group), dim=-1)
        self.int4.marlin_q, self.int4.marlin_s = pack_experts(
            self.int4.packed.unsqueeze(0), metadata.contiguous().unsqueeze(0)
        )
        self.int4.packed = None
        self.int4.scale = None

    def forward(self, hidden: tensorplay.Tensor) -> tensorplay.Tensor:
        if self.marlin and self.int4.marlin_q is not None:
            from megalm.kernels.marlin_moe import marlin_project

            return marlin_project(hidden, self.int4.marlin_q, self.int4.marlin_s)
        if self.int4 is not None:
            return self.int4(hidden)
        assert self.weight is not None
        return F.linear(hidden, self.weight)


class DiffusionAttention(nn.Module):
    """Gemma-4 attention with explicit encoder/canvas phases.

    The projections are shared by both phases, as in the checkpoint.  Full
    attention layers have no ``v_proj``: their value is the raw key projection
    passed through the scale-free value norm, then the key alone is rotated.
    """

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        layer_index: int,
        *,
        rotary: _RotaryEmbedding,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        marlin: bool = False,
    ) -> None:
        super().__init__()
        text = config.text
        self.layer_index = layer_index
        self.kind = text.layer_types[layer_index]
        self.head_dim = text.layer_head_dims[layer_index]
        self.num_query_heads = text.num_attention_heads
        self.num_kv_heads = text.layer_num_key_value_heads[layer_index]
        self.query_per_kv = self.num_query_heads // self.num_kv_heads
        self.query_width = self.num_query_heads * self.head_dim
        self.kv_width = self.num_kv_heads * self.head_dim
        self.sliding_window = (
            text.sliding_window if self.kind != FULL_ATTENTION else None
        )
        self.softcap = text.attn_logit_softcapping or 0.0
        self.eps = text.rms_norm_eps
        self.rotary = rotary
        self._use_triton = kernel_bindings.policy.use_triton
        self.has_value_projection = self.kind != FULL_ATTENTION
        self.quant_group = (
            config.quant.group_size if config.quant is not None else 0
        )

        def projection(in_features: int, out_features: int, suffix: str) -> _Projection:
            quant = None
            if config.module_is_quantized(f"layers.{layer_index}.{suffix}"):
                quant = Int4GroupLinear(
                    in_features,
                    out_features,
                    group_size=self.quant_group,
                    device=device,
                    dtype=dtype,
                )
            return _Projection(
                in_features,
                out_features,
                quant=quant,
                device=device,
                dtype=dtype,
                marlin=marlin,
            )

        hidden = text.hidden_size
        self.q_proj = projection(hidden, self.query_width, "self_attn.q_proj")
        self.k_proj = projection(hidden, self.kv_width, "self_attn.k_proj")
        self.v_proj = (
            projection(hidden, self.kv_width, "self_attn.v_proj")
            if self.has_value_projection
            else None
        )
        self.o_proj = projection(self.query_width, hidden, "self_attn.o_proj")
        self.q_norm = _parameter((self.head_dim,), device=device, dtype=dtype)
        self.k_norm = _parameter((self.head_dim,), device=device, dtype=dtype)
        self.register_buffer("qkv_marlin_q", None)
        self.register_buffer("qkv_marlin_s", None)

    def finalize_qkv(self) -> None:
        projections = [self.q_proj, self.k_proj]
        if self.v_proj is not None:
            projections.append(self.v_proj)
        if not all(proj.marlin and proj.int4.marlin_q is not None for proj in projections):
            return
        # Both Marlin permutations repeat on 64-channel boundaries. Joining
        # packed channel segments is identical to repacking their concatenation.
        self.qkv_marlin_q = tensorplay.cat(
            [proj.int4.marlin_q.reshape(self.q_proj.int4.in_features // 16, -1)
             for proj in projections], dim=-1
        )
        self.qkv_marlin_s = tensorplay.cat(
            [proj.int4.marlin_s.reshape(self.q_proj.int4.in_features // 32, -1)
             for proj in projections], dim=-1
        )
        for proj in projections:
            proj.int4.marlin_q = None
            proj.int4.marlin_s = None

    def _project_qkv(
        self, hidden: tensorplay.Tensor
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
        tokens = hidden.shape[0]
        if self.qkv_marlin_q is not None:
            from megalm.kernels.marlin_moe import marlin_project

            qkv = marlin_project(hidden, self.qkv_marlin_q, self.qkv_marlin_s)
            query = qkv[:, :self.query_width].view(tokens, self.num_query_heads, self.head_dim)
            key_end = self.query_width + self.kv_width
            key = qkv[:, self.query_width:key_end].view(tokens, self.num_kv_heads, self.head_dim)
            value = (
                qkv[:, key_end:].view(tokens, self.num_kv_heads, self.head_dim)
                if self.has_value_projection else key
            )
            return query, key, value
        query = self.q_proj(hidden).view(
            tokens, self.num_query_heads, self.head_dim
        )
        key = self.k_proj(hidden).view(tokens, self.num_kv_heads, self.head_dim)
        raw_value = (
            self.v_proj(hidden).view(tokens, self.num_kv_heads, self.head_dim)
            if self.v_proj is not None
            else key
        )
        return query, key, raw_value

    def _prepare_qkv(
        self,
        hidden: tensorplay.Tensor,
        positions: tensorplay.Tensor,
        rotation: tuple[tensorplay.Tensor, tensorplay.Tensor] | None,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
        query, key, value = self._project_qkv(hidden)
        cos, sin = rotation if rotation is not None else self.rotary(positions)
        if self._use_triton and query.is_cuda and self.head_dim in (256, 512):
            from megalm.kernels.diffusion_attention import prepare_diffusion_qkv

            return prepare_diffusion_qkv(
                query, key, value, self.q_norm, self.k_norm, cos, sin, self.eps
            )
        return (
            _apply_rope(_rms_norm(query, self.q_norm, self.eps), cos, sin),
            _apply_rope(_rms_norm(key, self.k_norm, self.eps), cos, sin),
            _rms_norm(value, None, self.eps),
        )

    def encode(
        self,
        hidden: tensorplay.Tensor,
        *,
        positions: tensorplay.Tensor,
        cache: DiffusionKVCache,
        rotation: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
    ) -> tensorplay.Tensor:
        """Causal pass that appends ``[start, start + tokens)`` to the cache."""

        tokens = hidden.shape[0]
        start = cache.length
        end = start + tokens
        if end > cache.capacity:
            raise DiffusionGemmaError(
                f"encoder overflow: {end} tokens exceed cache capacity {cache.capacity}"
            )
        query, key, value = self._prepare_qkv(hidden, positions, rotation)

        key_cache = cache.keys[self.layer_index]
        value_cache = cache.values[self.layer_index]
        key_cache[start:end].copy_(key)
        value_cache[start:end].copy_(value)

        attended = self._attend(
            query,
            key_cache[:end],
            value_cache[:end],
            query_positions=positions,
            key_positions=tensorplay.arange(
                end, device=hidden.device, dtype=tensorplay.int64
            ),
            causal=True,
        )
        return self.o_proj(attended)

    def denoise(
        self,
        hidden: tensorplay.Tensor,
        *,
        positions: tensorplay.Tensor,
        caches: Sequence[DiffusionKVCache],
        canvas_length: int,
        rotation: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
    ) -> tensorplay.Tensor:
        """Bidirectional canvas pass over ``[prefix KV, canvas KV]``.

        ``hidden`` is ``[batch * canvas_length, width]``; ``positions`` is
        ``[batch, canvas_length]``.  Canvas keys are never cached: every
        denoising step rebuilds them from the current noisy canvas.
        """

        batch = len(caches)
        tokens = hidden.shape[0]
        if tokens != batch * canvas_length:
            raise DiffusionGemmaError("canvas hidden state does not match the batch")
        query, key, value = self._prepare_qkv(hidden, positions.reshape(-1), rotation)
        query = query.view(batch, canvas_length, self.num_query_heads, self.head_dim)
        key = key.view(batch, canvas_length, self.num_kv_heads, self.head_dim)
        value = value.view(batch, canvas_length, self.num_kv_heads, self.head_dim)

        outputs = []
        for index, cache in enumerate(caches):
            prefix = cache.length
            if prefix == 0:
                keys = key[index]
                values = value[index]
                key_positions = positions[index]
            else:
                keys = tensorplay.cat(
                    (cache.keys[self.layer_index][:prefix], key[index]), dim=0
                )
                values = tensorplay.cat(
                    (cache.values[self.layer_index][:prefix], value[index]), dim=0
                )
                key_positions = tensorplay.cat(
                    (
                        tensorplay.arange(
                            prefix,
                            device=hidden.device,
                            dtype=tensorplay.int64,
                        ),
                        positions[index],
                    ),
                    dim=0,
                )
            outputs.append(
                self._attend(
                    query[index],
                    keys,
                    values,
                    query_positions=positions[index],
                    key_positions=key_positions,
                    causal=False,
                )
            )
        attended = tensorplay.cat(outputs, dim=0)
        return self.o_proj(attended)

    def _attend(
        self,
        query: tensorplay.Tensor,
        keys: tensorplay.Tensor,
        values: tensorplay.Tensor,
        *,
        query_positions: tensorplay.Tensor,
        key_positions: tensorplay.Tensor,
        causal: bool,
    ) -> tensorplay.Tensor:
        """GQA attention with Gemma-4 scale (1.0), soft-cap and window."""

        if (
            self._use_triton and query.is_cuda
            and query.dtype in (tensorplay.bfloat16, tensorplay.float16)
            and self.head_dim in (256, 512) and not self.softcap
        ):
            from megalm.kernels.diffusion_attention import diffusion_attention

            return diffusion_attention(
                query.contiguous(), keys.contiguous(), values.contiguous(),
                causal=causal, sliding_window=self.sliding_window or 0,
            )

        tokens = query.shape[0]
        if tokens > _ENCODER_QUERY_BLOCK and causal:
            chunks = []
            for start in range(0, tokens, _ENCODER_QUERY_BLOCK):
                end = min(start + _ENCODER_QUERY_BLOCK, tokens)
                limit = self._key_limit(
                    query_positions[end - 1], key_positions
                )
                chunks.append(
                    self._attend(
                        query[start:end],
                        keys[:limit],
                        values[:limit],
                        query_positions=query_positions[start:end],
                        key_positions=key_positions[:limit],
                        causal=True,
                    )
                )
            return tensorplay.cat(chunks, dim=0)

        repeated_keys = keys.repeat_interleave(self.query_per_kv, dim=1)
        repeated_values = values.repeat_interleave(self.query_per_kv, dim=1)
        q = query.transpose(0, 1).unsqueeze(0)
        k = repeated_keys.transpose(0, 1).unsqueeze(0)
        v = repeated_values.transpose(0, 1).unsqueeze(0)
        scores = tensorplay.matmul(q, k.transpose(-1, -2))
        if self.softcap:
            scores = self.softcap * tensorplay.tanh(scores / self.softcap)
        mask = tensorplay.ones(
            (tokens, keys.shape[0]), dtype=tensorplay.bool, device=query.device
        )
        if causal:
            mask = key_positions[None, :] <= query_positions[:, None]
        if self.sliding_window is not None:
            mask = mask & (
                key_positions[None, :]
                > query_positions[:, None] - self.sliding_window
            )
        scores = scores.masked_fill(~mask[None, None, :, :], _NEG_INF)
        weights = tensorplay.softmax(scores.float(), dim=-1).to(query.dtype)
        attended = tensorplay.matmul(weights, v)
        return attended.squeeze(0).transpose(0, 1).reshape(
            tokens, self.query_width
        )

    @staticmethod
    def _key_limit(
        last_query_position: tensorplay.Tensor,
        key_positions: tensorplay.Tensor,
    ) -> int:
        """Number of keys a causal chunk can see.

        Encoder positions are contiguous from zero, so the causal bound is the
        last query position + 1.
        """

        last = int(last_query_position)
        return min(int(key_positions.shape[0]), last + 1)


class DiffusionSparseMoE(nn.Module):
    """Gemma-4 sparse experts: softmax router plus int4 grouped GEMM.

    Routing follows the reference exactly: scale-free RMS norm, learned router
    scale, fp32 softmax over *all* experts, top-k renormalisation, then the
    per-expert multiplier.  On CUDA with int4 weights the expert GEMMs run as
    two grouped Triton launches per layer instead of a 128-iteration Python
    loop.
    """

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        layer_index: int,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        marlin: bool = False,
    ) -> None:
        super().__init__()
        text = config.text
        self.layer_index = layer_index
        self.hidden_size = text.hidden_size
        self.intermediate_size = text.moe_intermediate_size
        self.num_experts = text.num_experts
        self.top_k = text.top_k_experts
        self.eps = text.rms_norm_eps
        self.router_scale = _parameter(
            (text.hidden_size,), device=device, dtype=dtype
        )
        self.router_proj = _parameter(
            (text.num_experts, text.hidden_size), device=device, dtype=dtype
        )
        self.per_expert_scale = _parameter(
            (text.num_experts,), device=device, dtype=dtype
        )
        self.quant_group = (
            config.quant.group_size if config.quant is not None else 0
        )
        self.int4 = config.quant is not None
        self.marlin = False
        if marlin and self.int4:
            from megalm.kernels.marlin_moe import supports_marlin_moe

            self.marlin = supports_marlin_moe(
                device, dtype, self.hidden_size, self.intermediate_size,
                self.num_experts, self.quant_group,
            )
        if self.int4:
            self.register_buffer("w13_packed", None)
            self.register_buffer("w13_scale_zero", None)
            self.register_buffer("w2_packed", None)
            self.register_buffer("w2_scale_zero", None)
            self.register_buffer("marlin_q13", None)
            self.register_buffer("marlin_s13", None)
            self.register_buffer("marlin_q2", None)
            self.register_buffer("marlin_s2", None)
            self.gate_up_weight = None
            self.down_weight = None
        else:
            self.gate_up_weight = _parameter(
                (text.num_experts, 2 * self.intermediate_size, text.hidden_size),
                device=device,
                dtype=dtype,
            )
            self.down_weight = _parameter(
                (text.num_experts, text.hidden_size, self.intermediate_size),
                device=device,
                dtype=dtype,
            )
        self._workspace = None

    def configure_workspace(self, max_tokens: int) -> int:
        if not self.int4 or self.w13_packed is None or self.marlin:
            return 0
        from megalm.kernels.moe_grouped_int4 import Int4MoEWorkspace

        self._workspace = Int4MoEWorkspace.allocate(
            max_tokens=max_tokens,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            out_width13=2 * self.intermediate_size,
            top_k=self.top_k,
            dtype=self.router_proj.dtype,
            device=self.w13_packed.device,
        )
        element_size = self.router_proj.element_size()
        return int(self._workspace.gate_up.numel() * element_size) + int(
            self._workspace.contribution.numel() * element_size
        )

    def route(
        self,
        routing_states: tensorplay.Tensor,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        """Return ``(top_k_weights, top_k_ids)`` for ``[tokens, hidden]``."""

        normed = _rms_norm(routing_states, None, self.eps)
        routed = (
            normed
            * (self.hidden_size**-0.5)
            * self.router_scale
        )
        logits = F.linear(routed, self.router_proj)
        probabilities = tensorplay.softmax(logits.float(), dim=-1)
        top_weights, top_ids = tensorplay.topk(
            probabilities, self.top_k, dim=-1
        )
        top_weights = top_weights / top_weights.sum(-1, keepdim=True)
        top_weights = top_weights * self.per_expert_scale.index_select(
            0, top_ids.reshape(-1)
        ).reshape(top_ids.shape)
        return top_weights, top_ids

    def forward(
        self,
        expert_states: tensorplay.Tensor,
        *,
        routing_states: tensorplay.Tensor | None = None,
    ) -> tensorplay.Tensor:
        if routing_states is None:
            routing_states = expert_states
        top_weights, top_ids = self.route(routing_states)
        if self.int4:
            if self.marlin:
                if self.marlin_q13 is None:
                    raise DiffusionGemmaError(
                        f"layer {self.layer_index}: Marlin experts were not loaded"
                    )
                return self._forward_marlin(expert_states, top_weights, top_ids)
            return self._forward_int4(expert_states, top_weights, top_ids)
        return self._forward_dense(expert_states, top_weights, top_ids)

    def _forward_int4(
        self,
        expert_states: tensorplay.Tensor,
        top_weights: tensorplay.Tensor,
        top_ids: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        from megalm.kernels.moe_grouped_int4 import routed_moe_grouped_int4

        if self.w13_packed is None or self.w2_packed is None:
            raise DiffusionGemmaError(
                f"layer {self.layer_index}: int4 experts were not loaded"
            )
        if self._workspace is None or self._workspace.max_tokens < expert_states.shape[0]:
            self.configure_workspace(int(expert_states.shape[0]))
        return routed_moe_grouped_int4(
            expert_states,
            self.w13_packed,
            self.w13_scale_zero,
            self.w2_packed,
            self.w2_scale_zero,
            top_ids,
            top_weights,
            group_size=self.quant_group,
            num_experts=self.num_experts,
            gate_softcap=0.0,
            up_softcap=0.0,
            interleaved=False,
            cap_fn=_gemma_swiglu,
            workspace=self._workspace,
        )

    def _forward_marlin(
        self,
        expert_states: tensorplay.Tensor,
        top_weights: tensorplay.Tensor,
        top_ids: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        from megalm.kernels.marlin_moe import grouped_marlin_moe

        return grouped_marlin_moe(
            expert_states, top_ids, top_weights,
            self.marlin_q13, self.marlin_s13, self.marlin_q2, self.marlin_s2,
        )

    def _forward_dense(
        self,
        expert_states: tensorplay.Tensor,
        top_weights: tensorplay.Tensor,
        top_ids: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        result = tensorplay.zeros_like(expert_states)
        for expert in range(self.num_experts):
            mask = top_ids == expert
            if not bool(mask.any()):
                continue
            weight = (top_weights * mask).sum(-1)
            active = weight > 0
            tokens = expert_states[active]
            if tokens.shape[0] == 0:
                continue
            gate_up = F.linear(tokens, self.gate_up_weight[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(
                gelu_tanh(gate) * up, self.down_weight[expert]
            )
            update = tensorplay.zeros_like(expert_states)
            update[active] = expert_output * weight[active].unsqueeze(-1).to(
                expert_output.dtype
            )
            result = result + update
        return result


def _gemma_swiglu(
    gate: tensorplay.Tensor,
    up: tensorplay.Tensor,
    gate_softcap: float,
    up_softcap: float,
) -> tensorplay.Tensor:
    """Activation callback for the grouped int4 MoE kernel."""

    return gelu_tanh(gate) * up


class DiffusionSelfConditioning(nn.Module):
    """Soft-embedding MLP that carries the previous step's logits forward."""

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        text = config.text
        hidden = text.hidden_size
        intermediate = text.intermediate_size
        self.eps = text.rms_norm_eps
        self.pre_norm = _parameter((hidden,), device=device, dtype=dtype)
        self.gate_proj = _parameter(
            (intermediate, hidden), device=device, dtype=dtype
        )
        self.up_proj = _parameter(
            (intermediate, hidden), device=device, dtype=dtype
        )
        self.down_proj = _parameter(
            (hidden, intermediate), device=device, dtype=dtype
        )

    def forward(
        self,
        inputs_embeds: tensorplay.Tensor,
        signal: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        normed = _rms_norm(signal, self.pre_norm, self.eps)
        gate = F.linear(normed, self.gate_proj)
        up = F.linear(normed, self.up_proj)
        update = F.linear(gelu_tanh(gate) * up, self.down_proj)
        # ``post_norm`` is scale-free in the checkpoint.
        return _rms_norm(inputs_embeds + update, None, self.eps)


class DiffusionLayer(nn.Module):
    """One Gemma-4 layer with a phase-switchable attention path."""

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        layer_index: int,
        *,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        rotary: dict[str, _RotaryEmbedding],
        marlin: bool = False,
    ) -> None:
        super().__init__()
        text = config.text
        self.layer_index = layer_index
        self.config = config
        self._use_triton = kernel_bindings.policy.use_triton
        self.self_attn = DiffusionAttention(
            config,
            layer_index,
            rotary=rotary[text.layer_types[layer_index]],
            kernel_bindings=kernel_bindings,
            device=device,
            dtype=dtype,
            marlin=marlin,
        )
        self.input_layernorm = Gemma4RMSNorm(
            text.hidden_size,
            text.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            text.hidden_size,
            text.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            text.hidden_size,
            text.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            text.hidden_size,
            text.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
            text.hidden_size,
            text.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
            text.hidden_size,
            text.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
            text.hidden_size,
            text.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.gate_up_weight = _parameter(
            (2 * text.intermediate_size, text.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.down_weight = _parameter(
            (text.hidden_size, text.intermediate_size),
            device=device,
            dtype=dtype,
        )
        self.moe = DiffusionSparseMoE(
            config, layer_index, device=device, dtype=dtype, marlin=marlin
        )
        self.register_buffer(
            "layer_scalar", tensorplay.ones((1,), dtype=dtype, device=device)
        )

    def encode(
        self,
        hidden: tensorplay.Tensor,
        *,
        positions: tensorplay.Tensor,
        cache: DiffusionKVCache,
        rotation: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
    ) -> tensorplay.Tensor:
        return self._forward(
            hidden,
            attention=lambda value: self.self_attn.encode(
                value, positions=positions, cache=cache, rotation=rotation,
            ),
        )

    def denoise(
        self,
        hidden: tensorplay.Tensor,
        *,
        positions: tensorplay.Tensor,
        caches: Sequence[DiffusionKVCache],
        canvas_length: int,
        rotation: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
    ) -> tensorplay.Tensor:
        return self._forward(
            hidden,
            attention=lambda value: self.self_attn.denoise(
                value,
                positions=positions,
                caches=caches,
                canvas_length=canvas_length,
                rotation=rotation,
            ),
        )

    def _forward(
        self,
        hidden: tensorplay.Tensor,
        *,
        attention: Callable[[tensorplay.Tensor], tensorplay.Tensor],
    ) -> tensorplay.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = attention(hidden)
        hidden = self.post_attention_layernorm(hidden)
        hidden = residual + hidden

        residual = hidden
        hidden = self.pre_feedforward_layernorm(hidden)
        gate_up = F.linear(hidden, self.gate_up_weight)
        if self._use_triton and gate_up.is_cuda:
            activated = gelu_tanh_and_mul(gate_up)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            activated = gelu_tanh(gate) * up
        hidden = F.linear(activated, self.down_weight)
        hidden = self.post_feedforward_layernorm_1(hidden)

        moe_input = self.pre_feedforward_layernorm_2(residual)
        moe_output = self.moe(moe_input, routing_states=residual)
        moe_output = self.post_feedforward_layernorm_2(moe_output)
        hidden = hidden + moe_output

        hidden = self.post_feedforward_layernorm(hidden)
        hidden = residual + hidden
        return hidden * self.layer_scalar


class DiffusionGemmaModel(nn.Module):
    """The encoder/decoder weight stack shared by every diffusion phase."""

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        *,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        marlin: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        text = config.text
        self.embed_tokens = _parameter(
            (text.vocab_size, text.hidden_size), device=device, dtype=dtype
        )
        self.embed_scale_value = math.sqrt(text.hidden_size)
        self.embed_scale = tensorplay.tensor(
            self.embed_scale_value, dtype=dtype, device=device
        )
        self.rotary: dict[str, _RotaryEmbedding] = {}
        for kind in dict.fromkeys(text.layer_types):
            head_dim = next(
                dim
                for dim, layer_kind in zip(text.layer_head_dims, text.layer_types)
                if layer_kind == kind
            )
            self.rotary[kind] = _RotaryEmbedding(
                head_dim=head_dim,
                partial=text.rope_partial_factor[kind],
                theta=text.rope_theta[kind],
                rope_type=text.rope_type[kind],
                device=device,
            )
        self.layers = nn.ModuleList(
            [
                DiffusionLayer(
                    config,
                    index,
                    kernel_bindings=kernel_bindings,
                    device=device,
                    dtype=dtype,
                    rotary=self.rotary,
                    marlin=marlin,
                )
                for index in range(text.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(
            text.hidden_size,
            text.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.self_conditioning = DiffusionSelfConditioning(
            config, device=device, dtype=dtype
        )
        if text.tie_word_embeddings:
            self.lm_head_weight = None
        else:
            self.lm_head_weight = _parameter(
                (text.vocab_size, text.hidden_size), device=device, dtype=dtype
            )

    # -- phases ---------------------------------------------------------

    def embed(self, token_ids: tensorplay.Tensor) -> tensorplay.Tensor:
        hidden = F.embedding(token_ids, self.embed_tokens)
        if self.embed_scale_value != 1.0:
            hidden = hidden * self.embed_scale
        return hidden

    def encode(
        self,
        token_ids: tensorplay.Tensor,
        cache: DiffusionKVCache,
    ) -> None:
        """Causally encode ``token_ids`` and append them to ``cache``."""

        tokens = int(token_ids.shape[0])
        if tokens == 0:
            return
        if cache.length + tokens > cache.capacity:
            raise DiffusionGemmaError(
                "prompt and generated canvases exceed the KV cache capacity"
            )
        start = cache.length
        positions = tensorplay.arange(
            start, start + tokens, device=self.device, dtype=tensorplay.int64
        )
        hidden = self.embed(token_ids)
        rotations = {kind: rope(positions) for kind, rope in self.rotary.items()}
        for layer in self.layers:
            hidden = layer.encode(
                hidden, positions=positions, cache=cache,
                rotation=rotations[layer.self_attn.kind],
            )
        cache.length += tokens

    def denoise(
        self,
        canvas: tensorplay.Tensor,
        caches: Sequence[DiffusionKVCache],
        *,
        self_conditioning_logits: tensorplay.Tensor | None = None,
        self_conditioning_probs: tensorplay.Tensor | None = None,
        return_raw_logits: bool = False,
    ) -> tensorplay.Tensor:
        """Run one decoder pass over ``[batch, canvas_length]`` noisy tokens."""

        batch, canvas_length = canvas.shape
        if len(caches) != batch:
            raise DiffusionGemmaError("one KV cache per canvas is required")
        positions = tensorplay.stack(
            [
                tensorplay.arange(
                    cache.length,
                    cache.length + canvas_length,
                    device=self.device,
                    dtype=tensorplay.int64,
                )
                for cache in caches
            ],
            dim=0,
        )
        hidden = self.embed(canvas)
        soft = self_conditioning_probs
        if self_conditioning_logits is not None:
            if soft is not None:
                raise DiffusionGemmaError("pass either conditioning logits or probabilities")
            soft = tensorplay.softmax(
                self_conditioning_logits.float(), dim=-1
            ).to(self.dtype)
        if soft is not None:
            signal = (
                tensorplay.matmul(soft, self.embed_tokens)
                * self.embed_scale
            )
        else:
            signal = tensorplay.zeros_like(hidden)
        hidden = self.self_conditioning(
            hidden.reshape(batch * canvas_length, -1),
            signal.reshape(batch * canvas_length, -1),
        )
        flat_positions = positions.reshape(-1)
        rotations = {kind: rope(flat_positions) for kind, rope in self.rotary.items()}
        for layer in self.layers:
            hidden = layer.denoise(
                hidden,
                positions=positions,
                caches=caches,
                canvas_length=canvas_length,
                rotation=rotations[layer.self_attn.kind],
            )
        hidden = self.norm(hidden)
        logits = F.linear(hidden, self.lm_head()).view(batch, canvas_length, -1)
        if return_raw_logits:
            return logits
        logits = logits.float()
        cap = self.config.text.final_logit_softcapping
        if cap is not None:
            logits = cap * tensorplay.tanh(logits / cap)
        return logits

    def lm_head(self) -> tensorplay.Tensor:
        if self.config.text.tie_word_embeddings:
            return self.embed_tokens
        assert self.lm_head_weight is not None
        return self.lm_head_weight


class DiffusionGemmaForBlockDiffusion(nn.Module):
    """Checkpoint-level entry point: loading plus the two diffusion phases."""

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        *,
        device: tensorplay.device | str = "cuda",
        dtype: tensorplay.dtype = tensorplay.bfloat16,
        kernel_policy: KernelPolicy | None = None,
        kernel_bindings: KernelBindings | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.kernel_policy = kernel_policy or KernelPolicy()
        if kernel_bindings is None:
            kernel_bindings = bind_kernel_plan(
                self.kernel_policy,
                device=self.device,
                dtype=dtype,
                model_type="gemma4",
            )
        self.kernel_bindings = kernel_bindings
        marlin_mode = self.kernel_policy.diffusion_marlin_moe
        marlin = False
        if marlin_mode is not False and config.quant is not None:
            from megalm.kernels.marlin_moe import prepare_marlin_moe, supports_marlin_moe

            text = config.text
            marlin = supports_marlin_moe(
                self.device, dtype, text.hidden_size, text.moe_intermediate_size,
                text.num_experts, config.quant.group_size,
            )
            if marlin:
                try:
                    prepare_marlin_moe(self.device)
                except (OSError, RuntimeError):
                    if marlin_mode is True:
                        raise
                    marlin = False
        self.model = DiffusionGemmaModel(
            config,
            kernel_bindings=kernel_bindings,
            device=self.device,
            dtype=dtype,
            marlin=marlin,
        )

    # -- convenience ----------------------------------------------------

    @property
    def canvas_length(self) -> int:
        return self.config.canvas_length

    @property
    def text_config(self):
        return self.config.text

    def allocate_cache(self, capacity: int) -> DiffusionKVCache:
        return DiffusionKVCache.allocate(self.model, capacity)

    def encode(
        self,
        token_ids: tensorplay.Tensor,
        cache: DiffusionKVCache,
    ) -> None:
        self.model.encode(token_ids, cache)

    def denoise(
        self,
        canvas: tensorplay.Tensor,
        caches: Sequence[DiffusionKVCache],
        *,
        self_conditioning_logits: tensorplay.Tensor | None = None,
        self_conditioning_probs: tensorplay.Tensor | None = None,
        return_raw_logits: bool = False,
    ) -> tensorplay.Tensor:
        return self.model.denoise(
            canvas,
            caches,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_probs=self_conditioning_probs,
            return_raw_logits=return_raw_logits,
        )

    def configure_moe_workspace(self, max_tokens: int) -> int:
        total = 0
        for layer in self.model.layers:
            total += layer.moe.configure_workspace(max_tokens)
        return total

    def eval(self) -> "DiffusionGemmaForBlockDiffusion":
        super().eval()
        return self

    # -- loading --------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: tensorplay.device | str = "cuda",
        dtype: tensorplay.dtype = tensorplay.bfloat16,
        kernel_policy: KernelPolicy | None = None,
        kernel_bindings: KernelBindings | None = None,
        strict: bool = True,
    ) -> tuple["DiffusionGemmaForBlockDiffusion", WeightLoadReport]:
        config = DiffusionGemmaConfig.from_path(model_path)
        model = cls(
            config,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
        report = model.load_weights(model_path, strict=strict)
        model.eval()
        return model, report

    def load_weights(
        self,
        model_path: str | Path,
        *,
        source: SafetensorsSource | None = None,
        strict: bool = True,
    ) -> WeightLoadReport:
        """Stream the ``model.decoder.*`` stack into native parameters.

        Experts are folded straight into the grouped-kernel layout as they
        arrive: the per-expert source tensors are dropped immediately, so the
        resident footprint stays at the packed size instead of a transient
        copy of every expert.
        """

        text = self.config.text
        parameters = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded: set[str] = set()
        pending_attn: dict[str, dict[str, tensorplay.Tensor]] = {}
        pending_mlp: dict[int, set[str]] = {}
        expert_state: dict[int, dict[str, object]] = {}

        def target_of(name: str):
            target = parameters.get(name)
            if target is None:
                target = buffers.get(name)
            if target is None:
                raise DiffusionGemmaError(
                    f"checkpoint tensor has no native target: {name}"
                )
            return target

        @tensorplay.no_grad()
        def copy(name: str, tensor: tensorplay.Tensor) -> None:
            target = target_of(name)
            if tuple(target.shape) != tuple(tensor.shape):
                raise DiffusionGemmaError(
                    f"shape mismatch for {name}: expected {tuple(target.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            target.copy_(tensor.to(device=self.device, dtype=self.dtype))
            loaded.add(name)

        def layer_state(layer: int) -> dict[str, object]:
            state = expert_state.get(layer)
            if state is None:
                moe = self.model.layers[layer].moe
                if not moe.int4:
                    raise DiffusionGemmaError(
                        f"layer {layer}: packed experts in a dense checkpoint"
                    )
                experts = text.num_experts
                hidden = text.hidden_size
                intermediate = text.moe_intermediate_size
                group = self.config.quant.group_size  # type: ignore[union-attr]
                state = {
                    "w13_packed": tensorplay.empty(
                        (experts, 2 * intermediate, hidden // 2),
                        dtype=tensorplay.uint8,
                        device=self.device,
                    ),
                    "w13_scale_zero": tensorplay.empty(
                        (experts, hidden // group, 2 * intermediate, 2),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    "w2_packed": tensorplay.empty(
                        (experts, hidden, intermediate // 2),
                        dtype=tensorplay.uint8,
                        device=self.device,
                    ),
                    "w2_scale_zero": tensorplay.empty(
                        (experts, intermediate // group, hidden, 2),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    # Every expert contributes gate, up and down slabs.  The
                    # layer is committed as soon as the last slab arrives so
                    # streaming a shard never keeps several layers of packed
                    # experts resident at once.
                    "pending": 3 * experts,
                }
                expert_state[layer] = state
            return state

        @tensorplay.no_grad()
        def install_expert_packed(
            layer: int,
            expert: int,
            kind: str,
            packed: tensorplay.Tensor,
            scale: tensorplay.Tensor,
        ) -> None:
            state = layer_state(layer)
            hidden = text.hidden_size
            intermediate = text.moe_intermediate_size
            group = self.config.quant.group_size  # type: ignore[union-attr]
            packed_u8 = packed.to(device=self.device).view(tensorplay.uint8)
            scale_dev = scale.to(device=self.device, dtype=self.dtype)
            zero = tensorplay.full(
                scale_dev.shape, 8.0, dtype=self.dtype, device=self.device
            )
            if kind in {"gate", "up"}:
                rows = slice(0, intermediate) if kind == "gate" else slice(
                    intermediate, 2 * intermediate
                )
                state["w13_packed"][expert, rows].copy_(packed_u8)
                state["w13_scale_zero"][expert, :, rows].copy_(
                    tensorplay.stack((scale_dev.t(), zero.t()), dim=-1)
                )
            else:
                state["w2_packed"][expert].copy_(packed_u8)
                state["w2_scale_zero"][expert].copy_(
                    tensorplay.stack((scale_dev.t(), zero.t()), dim=-1)
                )
            state["pending"] -= 1  # type: ignore[operator]
            if state["pending"] == 0:
                finish_expert_layer(layer)

        @tensorplay.no_grad()
        def finish_expert_layer(layer: int) -> None:
            state = expert_state.pop(layer, None)
            if state is None:
                return
            moe = self.model.layers[layer].moe
            if moe.marlin:
                from megalm.kernels.marlin_moe import pack_experts

                moe.marlin_q13, moe.marlin_s13 = pack_experts(
                    state["w13_packed"], state["w13_scale_zero"]
                )
                moe.marlin_q2, moe.marlin_s2 = pack_experts(
                    state["w2_packed"], state["w2_scale_zero"]
                )
            else:
                moe.w13_packed = state["w13_packed"]
                moe.w13_scale_zero = state["w13_scale_zero"]
                moe.w2_packed = state["w2_packed"]
                moe.w2_scale_zero = state["w2_scale_zero"]
            loaded.add(f"model.layers.{layer}.moe.w13_packed")
            loaded.add(f"model.layers.{layer}.moe.w2_packed")

        @tensorplay.no_grad()
        def consume(name: str, tensor: tensorplay.Tensor) -> None:
            if name.startswith("model.encoder."):
                # The encoder shares every parameter with the decoder; only
                # layer scalars are stored twice.
                return
            if not name.startswith(DECODER_PREFIX + "."):
                return
            leaf = name[len(DECODER_PREFIX) + 1 :]
            if leaf == "embed_tokens.weight":
                copy("model.embed_tokens", tensor)
                return
            if leaf == "norm.weight":
                copy("model.norm.weight", tensor)
                return
            if leaf.startswith("self_conditioning."):
                field = leaf[len("self_conditioning.") :]
                if field.endswith(".weight"):
                    copy(
                        f"model.self_conditioning.{field[:-len('.weight')]}",
                        tensor,
                    )
                return
            parts = leaf.split(".")
            if len(parts) < 3 or parts[0] != "layers":
                return
            try:
                layer = int(parts[1])
            except ValueError:
                return
            rest = ".".join(parts[2:])
            prefix = f"model.layers.{layer}"
            native = self.model.layers[layer]

            if rest.startswith("experts."):
                # Dense checkpoints store stacked expert slabs.
                if rest == "experts.gate_up_proj":
                    copy(f"{prefix}.moe.gate_up_weight", tensor)
                    return
                if rest == "experts.down_proj":
                    copy(f"{prefix}.moe.down_weight", tensor)
                    return
                fields = rest[len("experts.") :].split(".")
                if len(fields) < 3:
                    return
                expert = int(fields[0])
                kind = fields[1].replace("_proj", "")
                tensor_kind = fields[2]
                if tensor_kind not in {"weight_packed", "weight_scale"}:
                    return
                slot = pending_attn.setdefault(
                    f"expert:{layer}:{expert}:{kind}", {}
                )
                slot[tensor_kind] = tensor
                if len(slot) == 2:
                    install_expert_packed(
                        layer,
                        expert,
                        kind,
                        slot["weight_packed"],
                        slot["weight_scale"],
                    )
                    pending_attn.pop(f"expert:{layer}:{expert}:{kind}", None)
                return

            if rest.startswith("self_attn."):
                field = rest[len("self_attn.") :]
                if field in {"q_norm.weight", "k_norm.weight"}:
                    copy(f"{prefix}.self_attn.{field[:-len('.weight')]}", tensor)
                    return
                if field.endswith((".weight_packed", ".weight_scale")):
                    module, kind = field.rsplit(".", 1)
                    slot = pending_attn.setdefault(
                        f"attn:{layer}:{module}", {}
                    )
                    slot[kind] = tensor
                    if len(slot) == 2:
                        target = getattr(native.self_attn, module)
                        if target.int4 is None:
                            raise DiffusionGemmaError(
                                f"layer {layer}: packed {module} in a dense config"
                            )
                        target.load_compressed(
                            slot["weight_packed"], slot["weight_scale"]
                        )
                        loaded.add(f"{prefix}.self_attn.{module}.weight")
                        pending_attn.pop(f"attn:{layer}:{module}", None)
                    return
                if field.endswith(".weight"):
                    module = field[: -len(".weight")]
                    target = getattr(native.self_attn, module, None)
                    if target is None:
                        return
                    copy(f"{prefix}.self_attn.{module}.weight", tensor)
                    return
                return

            if rest.startswith("mlp."):
                field = rest[len("mlp.") :]
                if field in {"gate_proj.weight", "up_proj.weight"}:
                    # The native layer stores gate and up concatenated.
                    target = target_of(f"{prefix}.gate_up_weight")
                    intermediate = text.intermediate_size
                    offset = 0 if field.startswith("gate") else intermediate
                    target[offset : offset + intermediate].copy_(
                        tensor.to(device=self.device, dtype=self.dtype)
                    )
                    parts_seen = pending_mlp.setdefault(layer, set())
                    parts_seen.add(field.split(".")[0])
                    if len(parts_seen) == 2:
                        loaded.add(f"{prefix}.gate_up_weight")
                        pending_mlp.pop(layer, None)
                    return
                if field == "down_proj.weight":
                    copy(f"{prefix}.down_weight", tensor)
                    return
                return

            if rest.startswith("router."):
                field = rest[len("router.") :]
                if field == "proj.weight":
                    copy(f"{prefix}.moe.router_proj", tensor)
                    return
                if field == "scale":
                    copy(f"{prefix}.moe.router_scale", tensor)
                    return
                if field == "per_expert_scale":
                    copy(f"{prefix}.moe.per_expert_scale", tensor)
                    return
                return

            if rest in {
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "pre_feedforward_layernorm_2.weight",
                "layer_scalar",
            }:
                copy(f"{prefix}.{rest}", tensor)
                return

        if source is None:
            source = SafetensorsSource(model_path)
        report = source.stream(consume)

        if strict:
            if pending_attn:
                raise DiffusionGemmaError(
                    "checkpoint ended with incomplete packed tensors: "
                    f"{sorted(pending_attn)[:4]}"
                )
            if pending_mlp:
                raise DiffusionGemmaError(
                    "checkpoint ended with half a dense gate/up pair: "
                    f"{sorted(pending_mlp)}"
                )
            if expert_state:
                raise DiffusionGemmaError(
                    "checkpoint ended with incomplete expert layers: "
                    f"{sorted(expert_state)}"
                )
            persistent = set(self.state_dict())
            missing = sorted(set(parameters) - loaded)
            missing.extend(
                f"model.layers.{index}.moe.w13_packed"
                for index, layer in enumerate(self.model.layers)
                if layer.moe.int4 and f"model.layers.{index}.moe.w13_packed" not in loaded
            )
            missing_buffers = sorted(
                name
                for name in set(buffers) & persistent
                if name not in loaded and name != "model.lm_head_weight"
            )
            # Expert buffers are registered as None and replaced wholesale;
            # their names are tracked through the layer marker above.
            missing_buffers = [
                name
                for name in missing_buffers
                if not name.endswith(
                    (
                        "w13_packed", "w13_scale_zero", "w2_packed",
                        "w2_scale_zero", "marlin_q13", "marlin_s13",
                        "marlin_q2", "marlin_s2",
                    )
                )
            ]
            if missing or missing_buffers:
                raise DiffusionGemmaError(
                    "checkpoint did not initialize parameters: "
                    f"{missing + missing_buffers}"
                )
        for layer in self.model.layers:
            layer.self_attn.finalize_qkv()
        return report


__all__ = [
    "DiffusionAttention",
    "DiffusionGemmaError",
    "DiffusionGemmaForBlockDiffusion",
    "DiffusionGemmaModel",
    "DiffusionKVCache",
    "DiffusionLayer",
    "DiffusionSparseMoE",
]
