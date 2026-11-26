"""Gemma-4 attention: per-layer geometry, dual RoPE flavours and soft-capping."""

from __future__ import annotations

import math

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.kernels.bindings import KernelBindings
from megalm.kernels.decode_gemv import decode_gemv
from megalm.kernels.fallback import dequantize_cache_to_dtype
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.cache import PagedKVCache
from megalm.runtime.kv_quant import kv_quant_of


class Gemma4RotaryCache(nn.Module):
    """RoPE tables for one layer type.

    Sliding layers use the standard (neox) layout over the full head.  Full
    attention layers use Gemma-4's *proportional* RoPE: frequencies divide by
    ``head_dim`` rather than the rotated width, and non-rotated channel pairs
    are zero-padded so the rotation is the identity for them.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        partial: float,
        theta: float,
        rope_type: str,
        max_positions: int,
        device: tensorplay.device,
    ) -> None:
        super().__init__()
        half = head_dim // 2
        angles = int(head_dim * partial) // 2
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
        positions = tensorplay.arange(
            max_positions, dtype=tensorplay.float32, device=device
        )
        frequencies = tensorplay.outer(positions, inverse)
        cos = tensorplay.cat(
            (
                frequencies.cos(),
                tensorplay.ones(
                    (max_positions, 0), dtype=tensorplay.float32, device=device
                ),
            ),
            dim=-1,
        )
        sin = tensorplay.cat(
            (
                frequencies.sin(),
                tensorplay.zeros(
                    (max_positions, 0), dtype=tensorplay.float32, device=device
                ),
            ),
            dim=-1,
        )
        self.register_buffer("cos", cos.contiguous(), persistent=False)
        self.register_buffer("sin", sin.contiguous(), persistent=False)


def gelu_tanh(x: tensorplay.Tensor) -> tensorplay.Tensor:
    """The ``gelu_pytorch_tanh`` activation used by Gemma-4 MLP and PLE gates."""

    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + tensorplay.tanh(inner))


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        *,
        layer_index: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        owns_kv: bool,
        use_k_eq_v: bool,
        sliding_window: int | None,
        softcap: float | None,
        hidden_size: int,
        eps: float,
        rotary: Gemma4RotaryCache,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        fuse_qkv: bool = True,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.head_dim = head_dim
        self.num_query_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.query_per_kv = num_heads // num_kv_heads
        self.query_width = num_heads * head_dim
        self.kv_width = num_kv_heads * head_dim
        self.owns_kv = owns_kv
        self.use_k_eq_v = use_k_eq_v
        self.sliding_window = sliding_window
        self.softcap = softcap or 0.0
        self.rotary = rotary
        self.rope_cache_write = kernel_bindings.core.rope_cache_write
        self._paged_decode = kernel_bindings.core.paged_decode_attention
        self._paged_prefill = kernel_bindings.core.paged_prefill_attention
        # Gemma-4 uses unit-normalised Q/K, so no 1/sqrt(d) attention scale.
        # Sliding-window masking is now handled inside the fast Triton kernels
        # via ``sliding_window``; only a soft-cap (unused by E2B) or a non-Triton
        # fallback needs the explicit score path.
        self._custom_scores = bool(
            softcap or not kernel_bindings.policy.use_triton
        )

        def parameter(shape: tuple[int, ...]) -> nn.Parameter:
            return nn.Parameter(
                tensorplay.empty(shape, device=device, dtype=dtype),
                requires_grad=False,
            )

        self._fused = (
            kernel_bindings.core.fused_qkv_norm_rope_cache_write
            if fuse_qkv
            and kernel_bindings.policy.use_triton
            and not softcap
            else None
        )
        self.qkv_proj = None
        self.q_proj = None
        self.k_proj = None
        self.k_norm = None
        self.v_proj = None
        self.o_proj = parameter((hidden_size, self.query_width))
        self.q_norm = parameter((head_dim,))
        self.eps = eps
        if self._fused is not None and owns_kv:
            # One packed projection: [Q heads, K heads, V heads] (V omitted for
            # k_eq_v layers).  Norm+RoPE+cache-write are fused in one kernel.
            sections = 1 if use_k_eq_v else 2
            width = (
                self.num_query_heads + sections * self.num_kv_heads
            ) * head_dim
            self.qkv_proj = parameter((width, hidden_size))
            self.k_norm = parameter((head_dim,))
        elif self._fused is not None:
            # Shared (YOCO) layers only own a Q projection.
            self.q_proj = parameter((self.query_width, hidden_size))
        else:
            self.q_proj = parameter((self.query_width, hidden_size))
            if owns_kv:
                self.k_proj = parameter((self.kv_width, hidden_size))
                self.k_norm = parameter((head_dim,))
                self.v_proj = (
                    None if use_k_eq_v
                    else parameter((self.kv_width, hidden_size))
                )
        # V norm has no learnable scale.

    def _norm(self, value: tensorplay.Tensor, weight: tensorplay.Tensor | None) -> tensorplay.Tensor:
        shape = (*value.shape[:-1], -1, self.head_dim)
        flat = value.view(*shape)
        mean = flat.float().pow(2).mean(-1, keepdim=True) + self.eps
        normalized = flat.float() * mean.rsqrt()
        if weight is not None:
            normalized = normalized * weight.float().view(1, 1, -1, self.head_dim)
        return normalized.to(value.dtype).view(value.shape)

    def _layer_cache(
        self,
        cache: PagedKVCache,
        kv_slot: int,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        """Narrow the shared arena to this layer's KV geometry."""

        key_cache, value_cache = cache.layer(kv_slot)
        if key_cache.shape[-2] != self.num_kv_heads:
            key_cache = key_cache[..., : self.num_kv_heads, :]
            value_cache = value_cache[..., : self.num_kv_heads, :]
        if key_cache.shape[-1] != self.head_dim:
            key_cache = key_cache[..., : self.head_dim]
            value_cache = value_cache[..., : self.head_dim]
        return key_cache, value_cache

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        kv_slot: int,
    ) -> tensorplay.Tensor:
        token_count = hidden_states.shape[0]
        decode = (
            batch.all_decode
            and self._fused is not None
            and token_count == 1
        )
        if self._fused is not None:
            key_cache, value_cache = self._layer_cache(cache, kv_slot)
            if self.owns_kv:
                assert self.qkv_proj is not None and self.k_norm is not None
                if decode:
                    n = self.qkv_proj.shape[0]
                    bn, bk = (64, 128) if n > 4096 else (8, 128)
                    qkv = decode_gemv(
                        hidden_states[0],
                        self.qkv_proj,
                        block_n=bn,
                        block_k=bk,
                        num_warps=8,
                    ).unsqueeze(0)
                else:
                    qkv = F.linear(hidden_states, self.qkv_proj)
                query = self._fused(
                    qkv,
                    self.q_norm,
                    self.k_norm,
                    self.eps,
                    batch.positions,
                    batch.slot_mapping,
                    self.rotary.cos,
                    self.rotary.sin,
                    key_cache,
                    value_cache,
                    batch.token_active,
                    num_query_heads=self.num_query_heads,
                    num_kv_heads=self.num_kv_heads,
                    v_norm=True,
                    k_eq_v=self.use_k_eq_v,
                    has_kv=True,
                )
            else:
                assert self.q_proj is not None
                if decode:
                    q = decode_gemv(
                        hidden_states[0],
                        self.q_proj,
                        block_n=8,
                        block_k=64,
                        num_warps=8,
                    ).unsqueeze(0)
                else:
                    q = F.linear(hidden_states, self.q_proj)
                q = q.view(token_count, self.num_query_heads, self.head_dim)
                query = self._fused(
                    q.reshape(token_count, -1),
                    self.q_norm,
                    self.q_norm,
                    self.eps,
                    batch.positions,
                    batch.slot_mapping,
                    self.rotary.cos,
                    self.rotary.sin,
                    key_cache,
                    value_cache,
                    batch.token_active,
                    num_query_heads=self.num_query_heads,
                    num_kv_heads=0,
                    v_norm=False,
                    k_eq_v=False,
                    has_kv=False,
                )
        else:
            query = F.linear(hidden_states, self.q_proj).view(
                token_count, self.num_query_heads, self.head_dim
            )
            query = self._norm(query, self.q_norm)
            if self.owns_kv:
                key = F.linear(hidden_states, self.k_proj).view(
                    token_count, self.num_kv_heads, self.head_dim
                )
                key = self._norm(key, self.k_norm)
                if self.use_k_eq_v:
                    value = key
                else:
                    assert self.v_proj is not None
                    value = F.linear(hidden_states, self.v_proj).view(
                        token_count, self.num_kv_heads, self.head_dim
                    )
                value = self._norm(value, None)
                key_cache, value_cache = self._layer_cache(cache, kv_slot)
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
            else:
                key_cache, value_cache = self._layer_cache(cache, kv_slot)
                # Shared layers only rotate Q; K/V come from the owning layer.
                cos = self.rotary.cos.index_select(
                    0, batch.positions.long()
                ).unsqueeze(1).to(query.dtype)
                sin = self.rotary.sin.index_select(
                    0, batch.positions.long()
                ).unsqueeze(1).to(query.dtype)
                half = self.head_dim // 2
                first, second = query[..., :half], query[..., half:]
                query = tensorplay.cat(
                    (first * cos - second * sin, second * cos + first * sin),
                    dim=-1,
                )

        attended = self._attention(query, key_cache, value_cache, batch)
        return F.linear(attended.reshape(token_count, -1), self.o_proj)

    def _attention(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        if self._custom_scores:
            return self._tensorplay_attention(query, key_cache, value_cache, batch)
        if batch.all_decode:
            return self._paged_decode(
                query,
                key_cache,
                value_cache,
                batch.block_tables,
                batch.sequence_lengths,
                batch.page_size,
                scale=1.0,
                sliding_window=self.sliding_window or 0,
            )
        return self._mixed_attention(query, key_cache, value_cache, batch)

    def _mixed_attention(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        query_start_loc = batch.query_start_loc_device
        assert query_start_loc is not None
        decode_count = 0
        for phase in batch.phases:
            if phase is not BatchPhase.DECODE:
                break
            decode_count += 1
        if any(
            phase is BatchPhase.DECODE for phase in batch.phases[decode_count:]
        ):
            raise RuntimeError("mixed Gemma-4 batch must keep decode rows first")
        output = tensorplay.empty_like(query)
        if decode_count:
            self._paged_decode(
                query[:decode_count],
                key_cache,
                value_cache,
                batch.block_tables[:decode_count],
                batch.sequence_lengths[:decode_count],
                batch.page_size,
                output=output[:decode_count],
                scale=1.0,
                sliding_window=self.sliding_window or 0,
            )
        if decode_count < batch.num_sequences:
            assert self._paged_prefill is not None
            self._paged_prefill(
                query,
                key_cache,
                value_cache,
                batch.block_tables[decode_count:],
                query_start_loc[decode_count:],
                batch.positions,
                query_lengths=batch.query_lengths[decode_count:],
                context_lengths=batch.context_lengths[decode_count:],
                page_size=batch.page_size,
                output=output,
                scale=1.0,
                sliding_window=self.sliding_window or 0,
            )
        return output

    def _tensorplay_attention(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        """Reference-correct attention with Gemma-4's scale and soft-cap.

        Used for soft-capped and windowed layers, and for the CPU fallback.
        Keys are gathered from paged slots per sequence so the semantics match
        the Triton prefill/decode kernels exactly.
        """
        output = tensorplay.empty_like(query)
        quant = kv_quant_of(key_cache)
        keys_all = dequantize_cache_to_dtype(
            key_cache, quant, "key", query.dtype
        ).reshape(-1, self.num_kv_heads, self.head_dim)
        values_all = dequantize_cache_to_dtype(
            value_cache, quant, "value", query.dtype
        ).reshape(-1, self.num_kv_heads, self.head_dim)
        for sequence_index in range(batch.num_sequences):
            start = batch.query_start_loc[sequence_index]
            end = batch.query_start_loc[sequence_index + 1]
            if start == end:
                continue
            query_length = end - start
            context_length = batch.context_lengths[sequence_index]
            sequence_length = context_length + query_length
            num_pages = math.ceil(sequence_length / batch.page_size)
            page_ids = batch.block_tables[
                sequence_index, :num_pages
            ].long()
            slots = (
                page_ids.unsqueeze(-1) * batch.page_size
                + tensorplay.arange(
                    batch.page_size, device=query.device
                ).unsqueeze(0)
            ).reshape(-1)[:sequence_length]
            first = 0
            keys = keys_all.index_select(0, slots).repeat_interleave(
                self.query_per_kv, dim=1
            )
            values = values_all.index_select(0, slots).repeat_interleave(
                self.query_per_kv, dim=1
            )
            span = int(keys.shape[0])
            q = query[start:end].float().transpose(0, 1).unsqueeze(0)
            k = keys.float().transpose(0, 1).unsqueeze(0)
            v = values.to(query.dtype).transpose(0, 1).unsqueeze(0)
            scores = tensorplay.matmul(q, k.transpose(-1, -2))
            if self.softcap:
                scores = self.softcap * tensorplay.tanh(scores / self.softcap)
            if query_length > 1:
                q_positions = tensorplay.arange(
                    sequence_length - query_length,
                    sequence_length,
                    device=query.device,
                )
                k_positions = tensorplay.arange(span, device=query.device)
                causal = k_positions[None, :] <= q_positions[:, None]
                if self.sliding_window is not None:
                    causal = causal & (
                        k_positions[None, :]
                        > q_positions[:, None] - self.sliding_window
                    )
                causal = causal[None, None, :, :].expand(scores.shape)
                scores = scores.masked_fill(~causal, float("-inf"))
            weights = tensorplay.softmax(scores, dim=-1).to(query.dtype)
            attended = tensorplay.matmul(weights, v)
            output[start:end] = attended.squeeze(0).transpose(0, 1)
        return output
