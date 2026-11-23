from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.models.qwen3_5.config import Qwen35TextConfig
from megalm.models.qwen3_5.norms import GatedRMSNorm, frozen_parameter
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.hybrid_cache import HybridCache


class Qwen35GatedDeltaNet(nn.Module):
    """First-party GDN with packed Triton prefill and decode paths."""

    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        layer_index: int,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_width = self.num_key_heads * self.key_head_dim
        self.value_width = self.num_value_heads * self.value_head_dim
        self.conv_width = 2 * self.key_width + self.value_width
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_repeat = self.num_value_heads // self.num_key_heads
        self.scale = self.key_head_dim**-0.5
        self._causal_conv_update = (
            kernel_bindings.core.gdn_causal_conv_update
        )
        self._causal_conv_prefill = (
            kernel_bindings.core.gdn_causal_conv_prefill
        )
        self._packed_recurrent_decode = (
            kernel_bindings.core.gdn_packed_recurrent_decode
        )
        self._packed_recurrent_prefill = (
            kernel_bindings.core.gdn_packed_recurrent_prefill
        )
        self._gated_rms_norm = (
            kernel_bindings.core.gdn_gated_rms_norm
        )

        qkvz_width = self.conv_width + self.value_width
        projection_weight = tensorplay.empty(
            (
                qkvz_width + 2 * self.num_value_heads,
                self.hidden_size,
            ),
            device=device,
            dtype=dtype,
        )
        # Keep the checkpoint-facing parameters as views, but retain their
        # shared contiguous storage for the latency-critical decode projection.
        # Prefill preserves the original two-GEMM geometry.
        self.in_proj_qkvz_weight = nn.Parameter(
            projection_weight[:qkvz_width],
            requires_grad=False,
        )
        self.in_proj_ba_weight = nn.Parameter(
            projection_weight[qkvz_width:],
            requires_grad=False,
        )
        self._decode_projection_weight = projection_weight
        self.conv1d_weight = frozen_parameter(
            (self.conv_width, 1, self.conv_kernel_dim),
            device=device,
            dtype=dtype,
        )
        self.A_log = frozen_parameter(
            (self.num_value_heads,),
            device=device,
            dtype=tensorplay.float32,
        )
        self.dt_bias = frozen_parameter(
            (self.num_value_heads,),
            device=device,
            dtype=tensorplay.float32,
        )
        self.norm = GatedRMSNorm(
            self.value_head_dim,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.out_proj_weight = frozen_parameter(
            (self.hidden_size, self.value_width),
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: HybridCache,
    ) -> tensorplay.Tensor:
        state_indices = batch.gdn_state_indices
        if state_indices is None:
            raise RuntimeError("Qwen3.5 batch has no GDN state indices")
        if batch.all_decode and hidden_states.shape[0] <= 16:
            projected = F.linear(
                hidden_states,
                self._decode_projection_weight,
            )
            mixed_qkv, z, b, a = projected.split(
                (
                    self.conv_width,
                    self.value_width,
                    self.num_value_heads,
                    self.num_value_heads,
                ),
                dim=-1,
            )
        else:
            qkvz = F.linear(hidden_states, self.in_proj_qkvz_weight)
            ba = F.linear(hidden_states, self.in_proj_ba_weight)
            mixed_qkv, z = qkvz.split(
                (self.conv_width, self.value_width),
                dim=-1,
            )
            b, a = ba.chunk(2, dim=-1)
        conv_cache, recurrent_cache = cache.gdn.layer(self.layer_index)
        triton_decode = (
            batch.all_decode
            and self._causal_conv_update is not None
            and self._packed_recurrent_decode is not None
        )
        triton_prefill = (
            not batch.all_decode
            and self._causal_conv_prefill is not None
            and self._packed_recurrent_prefill is not None
        )
        if triton_decode:
            core = self._decode_triton(
                mixed_qkv,
                b,
                a,
                state_indices,
                batch.token_active,
                conv_cache,
                recurrent_cache,
            )
        elif triton_prefill:
            core = self._prefill_triton(
                mixed_qkv,
                b,
                a,
                batch,
                state_indices,
                conv_cache,
                recurrent_cache,
            )
        elif batch.all_decode:
            core = self._decode(
                mixed_qkv,
                b,
                a,
                state_indices,
                conv_cache,
                recurrent_cache,
            )
        else:
            core = self._prefill_or_mixed(
                mixed_qkv,
                b,
                a,
                batch,
                state_indices,
                conv_cache,
                recurrent_cache,
            )
        triton_norm = (
            (triton_decode or triton_prefill)
            and self._gated_rms_norm is not None
        )
        if triton_norm:
            assert self._gated_rms_norm is not None
            gated = self._gated_rms_norm(
                core,
                z.view(
                    hidden_states.shape[0],
                    self.num_value_heads,
                    self.value_head_dim,
                ),
                self.norm.weight,
                self.norm.eps,
            ).reshape(hidden_states.shape[0], self.value_width)
        else:
            core = core.reshape(-1, self.value_head_dim)
            gated = self.norm(
                core,
                z.reshape(-1, self.value_head_dim),
            ).reshape(hidden_states.shape[0], self.value_width)
        return F.linear(gated, self.out_proj_weight)

    def _decode_triton(
        self,
        mixed_qkv: tensorplay.Tensor,
        b: tensorplay.Tensor,
        a: tensorplay.Tensor,
        state_indices: tensorplay.Tensor,
        active: tensorplay.Tensor,
        conv_cache: tensorplay.Tensor,
        recurrent_cache: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        assert self._causal_conv_update is not None
        assert self._packed_recurrent_decode is not None
        convolved = self._causal_conv_update(
            mixed_qkv,
            conv_cache,
            self.conv1d_weight,
            state_indices[:, 0],
            active,
        )
        return self._packed_recurrent_decode(
            convolved,
            a,
            b,
            self.A_log,
            self.dt_bias,
            recurrent_cache,
            state_indices[:, 1],
            active,
            num_key_heads=self.num_key_heads,
            scale=self.scale,
        )

    def _prefill_triton(
        self,
        mixed_qkv: tensorplay.Tensor,
        b: tensorplay.Tensor,
        a: tensorplay.Tensor,
        batch: ExecutionBatch,
        state_indices: tensorplay.Tensor,
        conv_cache: tensorplay.Tensor,
        recurrent_cache: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        assert self._causal_conv_prefill is not None
        assert self._packed_recurrent_prefill is not None
        query_starts = batch.query_start_loc_device
        if query_starts is None:
            raise RuntimeError("Qwen3.5 prefill has no device query offsets")
        max_query_tokens = max(batch.query_lengths)
        convolved = self._causal_conv_prefill(
            mixed_qkv,
            conv_cache,
            self.conv1d_weight,
            query_starts,
            batch.positions,
            state_indices,
            max_query_tokens=max_query_tokens,
        )
        return self._packed_recurrent_prefill(
            convolved,
            a,
            b,
            self.A_log,
            self.dt_bias,
            recurrent_cache,
            query_starts,
            batch.positions,
            state_indices,
            num_key_heads=self.num_key_heads,
            scale=self.scale,
            max_query_tokens=max_query_tokens,
        )

    def _decode(
        self,
        mixed_qkv: tensorplay.Tensor,
        b: tensorplay.Tensor,
        a: tensorplay.Tensor,
        state_indices: tensorplay.Tensor,
        conv_cache: tensorplay.Tensor,
        recurrent_cache: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        conv_indices = state_indices[:, 0].long()
        recurrent_indices = state_indices[:, 1].long()
        if tensorplay.any(conv_indices < 0) or tensorplay.any(recurrent_indices < 0):
            raise RuntimeError("decode references an invalid GDN state slot")
        prior = conv_cache.index_select(0, conv_indices)
        combined = tensorplay.cat(
            (prior, mixed_qkv.unsqueeze(-1)),
            dim=-1,
        )
        conv_output = F.silu(
            (
                combined
                * self.conv1d_weight[:, 0, :].unsqueeze(0)
            ).sum(dim=-1)
        )
        conv_cache.index_copy_(0, conv_indices, combined[:, :, 1:])
        query, key, value = self._split_qkv(conv_output)
        state = recurrent_cache.index_select(
            0,
            recurrent_indices,
        )
        output, state = self._delta_step(
            query,
            key,
            value,
            a,
            b,
            state,
        )
        recurrent_cache.index_copy_(0, recurrent_indices, state)
        return output

    def _prefill_or_mixed(
        self,
        mixed_qkv: tensorplay.Tensor,
        b: tensorplay.Tensor,
        a: tensorplay.Tensor,
        batch: ExecutionBatch,
        state_indices: tensorplay.Tensor,
        conv_cache: tensorplay.Tensor,
        recurrent_cache: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        output = tensorplay.empty(
            (
                mixed_qkv.shape[0],
                self.num_value_heads,
                self.value_head_dim,
            ),
            device=mixed_qkv.device,
            dtype=mixed_qkv.dtype,
        )
        for sequence_index, (
            start,
            end,
            context_length,
        ) in enumerate(
            zip(
                batch.query_start_loc[:-1],
                batch.query_start_loc[1:],
                batch.context_lengths,
            )
        ):
            conv_index = int(state_indices[sequence_index, 0].item())
            recurrent_index = int(
                state_indices[sequence_index, 1].item()
            )
            if min(conv_index, recurrent_index) < 0:
                raise RuntimeError("prefill references an invalid GDN state slot")
            prior_conv = conv_cache[conv_index]
            state = recurrent_cache[recurrent_index]
            if context_length == 0:
                prior_conv.zero_()
                state.zero_()
            raw = mixed_qkv[start:end].transpose(0, 1)
            combined = tensorplay.cat((prior_conv, raw), dim=-1)
            convolved = F.conv1d(
                combined.unsqueeze(0),
                self.conv1d_weight,
                groups=self.conv_width,
            ).squeeze(0).transpose(0, 1)
            convolved = F.silu(convolved)
            prior_conv.copy_(
                combined[:, -(self.conv_kernel_dim - 1) :]
            )
            query, key, value = self._split_qkv(convolved)
            for token_index in range(end - start):
                token_output, state = self._delta_step(
                    query[token_index : token_index + 1],
                    key[token_index : token_index + 1],
                    value[token_index : token_index + 1],
                    a[start + token_index : start + token_index + 1],
                    b[start + token_index : start + token_index + 1],
                    state.unsqueeze(0),
                )
                output[start + token_index] = token_output[0]
                state = state[0]
            recurrent_cache[recurrent_index].copy_(state)
        return output

    def _split_qkv(
        self,
        mixed_qkv: tensorplay.Tensor,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
        query, key, value = mixed_qkv.split(
            (self.key_width, self.key_width, self.value_width),
            dim=-1,
        )
        query = query.view(-1, self.num_key_heads, self.key_head_dim)
        key = key.view(-1, self.num_key_heads, self.key_head_dim)
        value = value.view(
            -1,
            self.num_value_heads,
            self.value_head_dim,
        )
        if self.key_repeat > 1:
            query = query.repeat_interleave(self.key_repeat, dim=1)
            key = key.repeat_interleave(self.key_repeat, dim=1)
        return query, key, value

    def _delta_step(
        self,
        query: tensorplay.Tensor,
        key: tensorplay.Tensor,
        value: tensorplay.Tensor,
        a: tensorplay.Tensor,
        b: tensorplay.Tensor,
        state: tensorplay.Tensor,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        query_float = query.float()
        query_float = query_float * tensorplay.rsqrt(
            query_float.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        key_float = key.float()
        key_float = key_float * tensorplay.rsqrt(
            key_float.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        value_float = value.float()
        decay = tensorplay.exp(
            -tensorplay.exp(self.A_log)
            * F.softplus(a.float() + self.dt_bias)
        )
        beta = tensorplay.sigmoid(b.float())
        state = state.float() * decay[:, :, None, None]
        memory_value = tensorplay.einsum(
            "bhkv,bhk->bhv",
            state,
            key_float,
        )
        delta = (value_float - memory_value) * beta[:, :, None]
        state = state + key_float.unsqueeze(-1) * delta.unsqueeze(-2)
        output = tensorplay.einsum(
            "bhkv,bhk->bhv",
            state,
            query_float * self.scale,
        )
        return output.to(value.dtype), state

    @property
    def recurrent_state_elements(self) -> int:
        return (
            self.num_value_heads
            * self.key_head_dim
            * self.value_head_dim
        )

    @property
    def conv_state_elements(self) -> int:
        return self.conv_width * (self.conv_kernel_dim - 1)
