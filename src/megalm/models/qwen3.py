from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings, bind_kernel_plan
from megalm.kernels.decode_gemv import decode_gemv, gate_up_silu_gemv
from megalm.kernels.ops import (
    GroupedMoEWorkspace,
    KernelPolicy,
    routed_moe_tensorcore,
)
from megalm.loading import SafetensorsSource, WeightLoadReport
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class ModelImplementationError(RuntimeError):
    """Raised when a model fact or checkpoint violates the implemented contract."""


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    hidden_act: str
    attention_bias: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...]

    @classmethod
    def from_path(cls, model_path: str | Path) -> Qwen3Config:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelImplementationError(
                f"cannot load Qwen3 config from {config_path}"
            ) from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Qwen3Config:
        """Build a Qwen3 config from an already-parsed JSON object.

        OXR embeds a Qwen3 config in ``llm_config`` and needs the same validation
        without writing a temporary config file.
        """

        model_type = str(payload.get("model_type", ""))
        if model_type not in {"qwen3", "qwen3_moe"}:
            raise ModelImplementationError(
                f"unsupported first-party model_type {model_type!r}"
            )
        if payload.get("rope_scaling") not in (None, {}):
            raise ModelImplementationError(
                "Qwen3 rope_scaling requires an explicit first-party implementation"
            )
        hidden_size = int(payload["hidden_size"])
        num_attention_heads = int(payload["num_attention_heads"])
        head_dim = int(
            payload.get("head_dim", hidden_size // num_attention_heads)
        )
        config = cls(
            model_type=model_type,
            vocab_size=int(payload["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(payload["intermediate_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(
                payload.get("num_key_value_heads", num_attention_heads)
            ),
            head_dim=head_dim,
            max_position_embeddings=int(payload["max_position_embeddings"]),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-6)),
            rope_theta=float(payload.get("rope_theta", 1_000_000.0)),
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", False)),
            hidden_act=str(payload.get("hidden_act", "silu")),
            attention_bias=bool(payload.get("attention_bias", False)),
            num_experts=int(payload.get("num_experts", 0)),
            num_experts_per_tok=int(payload.get("num_experts_per_tok", 0)),
            moe_intermediate_size=int(payload.get("moe_intermediate_size", 0)),
            norm_topk_prob=bool(payload.get("norm_topk_prob", False)),
            decoder_sparse_step=int(payload.get("decoder_sparse_step", 1)),
            mlp_only_layers=tuple(
                int(index) for index in payload.get("mlp_only_layers", ())
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
            self.decoder_sparse_step,
        )
        if any(value <= 0 for value in positive):
            raise ModelImplementationError("Qwen3 dimensions must be positive")
        # Qwen3 decouples head_dim from hidden_size / heads (e.g. the 0.6B
        # checkpoint uses 16 heads x 128 dims over a 1024-wide residual). The
        # attention projections already use the query width explicitly.
        if self.num_attention_heads % self.num_key_value_heads:
            raise ModelImplementationError("GQA head counts are not divisible")
        if self.head_dim % 2:
            raise ModelImplementationError("RoPE head_dim must be even")
        if self.hidden_act != "silu":
            raise ModelImplementationError("only Qwen3 SiLU is supported")
        if self.attention_bias:
            raise ModelImplementationError("attention bias is not implemented")
        if any(
            index < 0 or index >= self.num_hidden_layers
            for index in self.mlp_only_layers
        ):
            raise ModelImplementationError("mlp_only_layers contains an invalid index")
        if self.model_type == "qwen3_moe":
            if min(
                self.num_experts,
                self.num_experts_per_tok,
                self.moe_intermediate_size,
            ) <= 0:
                raise ModelImplementationError("Qwen3-MoE expert facts are missing")
            if self.num_experts_per_tok > self.num_experts:
                raise ModelImplementationError("top-k exceeds expert count")

    def is_moe_layer(self, layer_index: int) -> bool:
        return (
            self.model_type == "qwen3_moe"
            and self.num_experts > 0
            and layer_index not in self.mlp_only_layers
            and (layer_index + 1) % self.decoder_sparse_step == 0
        )


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


class WeightOnlyLinear(nn.Module):
    def __init__(
        self,
        output_features: int,
        input_features: int,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        use_decode_gemv: bool = False,
    ) -> None:
        super().__init__()
        self.weight = _parameter(
            (output_features, input_features),
            device=device,
            dtype=dtype,
        )
        self.use_decode_gemv = use_decode_gemv

    def forward(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
        if self.use_decode_gemv and x.shape[0] == 1:
            return decode_gemv(x[0], self.weight).unsqueeze(0)
        return F.linear(x, self.weight)


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.weight = _parameter(
            (vocab_size, hidden_size),
            device=device,
            dtype=dtype,
        )

    def forward(self, input_ids: tensorplay.Tensor) -> tensorplay.Tensor:
        return F.embedding(input_ids, self.weight)


class RMSNorm(nn.Module):
    def __init__(
        self,
        width: int,
        eps: float,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.weight = _parameter((width,), device=device, dtype=dtype)
        self.eps = eps
        self._rms_norm = kernel_bindings.core.rms_norm
        self._fused_add_rms_norm = (
            kernel_bindings.core.fused_add_rms_norm
        )

    def forward(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
        return self._rms_norm(
            x,
            self.weight,
            self.eps,
        )

    def with_residual(
        self,
        x: tensorplay.Tensor,
        residual: tensorplay.Tensor,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        return self._fused_add_rms_norm(
            x,
            residual,
            self.weight,
            self.eps,
        )


class DenseMLP(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
        use_decode_gemv: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_weight = _parameter(
            (2 * config.intermediate_size, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.down_proj = WeightOnlyLinear(
            config.hidden_size,
            config.intermediate_size,
            device=device,
            dtype=dtype,
        )
        self._silu_and_mul = kernel_bindings.core.silu_and_mul
        self.use_decode_gemv = use_decode_gemv

    def forward(self, hidden_states: tensorplay.Tensor, *, decode: bool) -> tensorplay.Tensor:
        if self.use_decode_gemv and decode and hidden_states.shape[0] == 1:
            activated = gate_up_silu_gemv(hidden_states[0], self.gate_up_weight).unsqueeze(0)
        else:
            gate_up = F.linear(hidden_states, self.gate_up_weight)
            activated = self._silu_and_mul(gate_up)
        return self.down_proj(activated)


class SparseMoE(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.gate = WeightOnlyLinear(
            config.num_experts,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.gate_up_weight = _parameter(
            (
                config.num_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            ),
            device=device,
            dtype=dtype,
        )
        self.down_weight = _parameter(
            (
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            ),
            device=device,
            dtype=dtype,
        )
        self.top_k = config.num_experts_per_tok
        self.normalize_top_k = config.norm_topk_prob
        self.kernel_policy = kernel_bindings.policy
        self.kernels = kernel_bindings.moe
        self.workspace: GroupedMoEWorkspace | None = None

    def forward(self, hidden_states: tensorplay.Tensor, *, decode: bool) -> tensorplay.Tensor:
        if (
            self.top_k == 1
            and decode
            and self.workspace is not None
            and hidden_states.shape[0]
            >= self.kernel_policy.grouped_moe_min_tokens
            and hidden_states.shape[0] < self.workspace.compact_min_tokens
            and self.gate.weight.shape[0] <= 16
            and (
                hidden_states.shape[0] <= 32
                or (
                    hidden_states.shape[1] * self.gate.weight.shape[0]
                    <= 8192
                )
            )
        ):
            return self.kernels.routed_grouped_from_hidden(
                hidden_states,
                self.gate.weight,
                self.gate_up_weight,
                self.down_weight,
                normalize=self.normalize_top_k,
                workspace=self.workspace,
            )
        if self.top_k == 1:
            route_output = (
                None
                if self.workspace is None
                else (
                    self.workspace.topk_weights[: hidden_states.shape[0]],
                    self.workspace.topk_ids[: hidden_states.shape[0]],
                )
            )
            if (
                decode
                and self.gate.weight.shape[0] <= 16
                and (
                    hidden_states.shape[0] <= 32
                    or (
                        hidden_states.shape[1]
                        * self.gate.weight.shape[0]
                        <= 8192
                    )
                )
            ):
                topk_weights, topk_ids = self.kernels.top1_from_hidden(
                    hidden_states,
                    self.gate.weight,
                    normalize=self.normalize_top_k,
                    output=route_output,
                )
            else:
                router_logits = self.gate(hidden_states)
                topk_weights, topk_ids = self.kernels.top1(
                    router_logits,
                    normalize=self.normalize_top_k,
                    output=route_output,
                )
        else:
            router_logits = self.gate(hidden_states)
            topk_weights, topk_ids = self.kernels.topk(
                router_logits,
                self.top_k,
                normalize=self.normalize_top_k,
            )
        if (
            hidden_states.shape[0]
            >= self.kernel_policy.grouped_moe_min_tokens
            and (
                self.top_k == 1
                or (
                    self.workspace is not None
                    and self.top_k <= self.workspace.max_top_k
                )
            )
        ):
            return self.kernels.routed_grouped(
                hidden_states,
                self.gate_up_weight,
                self.down_weight,
                topk_ids,
                topk_weights,
                workspace=self.workspace,
            )
        if (
            decode
            and self.top_k == 1
            and hidden_states.shape[0]
            >= self.kernel_policy.moe_tensorcore_min_tokens
        ):
            return routed_moe_tensorcore(
                hidden_states,
                self.gate_up_weight,
                self.down_weight,
                topk_ids,
                topk_weights,
            )
        routed_impl = (
            self.kernels.routed_small
            if (
                decode
                or hidden_states.shape[0] <= self.kernel_policy.moe_max_tokens
            )
            else self.kernels.routed_fallback
        )
        return routed_impl(
            hidden_states,
            self.gate_up_weight,
            self.down_weight,
            topk_ids,
            topk_weights,
        )


class RotaryCache(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
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
        positions = tensorplay.arange(
            config.max_position_embeddings,
            dtype=tensorplay.float32,
            device=device,
        )
        frequencies = tensorplay.outer(positions, inverse_frequency)
        self.register_buffer(
            "cos",
            frequencies.cos().contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "sin",
            frequencies.sin().contiguous(),
            persistent=False,
        )


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        rotary: RotaryCache,
        kernel_bindings: KernelBindings,
        use_decode_gemv: bool = False,
    ) -> None:
        super().__init__()
        self.query_width = config.num_attention_heads * config.head_dim
        self.kv_width = config.num_key_value_heads * config.head_dim
        self.qkv_proj = WeightOnlyLinear(
            self.query_width + 2 * self.kv_width,
            config.hidden_size,
            device=device,
            dtype=dtype,
            use_decode_gemv=use_decode_gemv,
        )
        self.o_proj = WeightOnlyLinear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            device=device,
            dtype=dtype,
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
        self.num_query_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_per_kv = self.num_query_heads // self.num_kv_heads
        self.rotary = rotary
        self._fused_qkv = (
            kernel_bindings.core.fused_qkv_norm_rope_cache_write
        )
        self._paged_decode = kernel_bindings.core.paged_decode_attention
        self._paged_prefill = (
            kernel_bindings.core.paged_prefill_attention
        )
        self._mixed_attention_impl = (
            self._mixed_attention_triton
            if self._paged_prefill is not None
            else self._mixed_attention_tensorplay
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        layer_index: int,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        token_count = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        key_cache, value_cache = cache.layer(layer_index)
        query = self._fused_qkv(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.eps,
            batch.positions,
            batch.slot_mapping,
            self.rotary.cos,
            self.rotary.sin,
            key_cache,
            value_cache,
            batch.token_active,
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
        )
        if batch.all_decode:
            attended = self._paged_decode(
                query,
                key_cache,
                value_cache,
                batch.block_tables,
                batch.sequence_lengths,
                batch.page_size,
            )
        else:
            attended = self._mixed_attention_impl(
                query,
                key_cache,
                value_cache,
                batch,
            )
        return self.o_proj(attended.reshape(token_count, -1))

    def _mixed_attention_triton(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        query_start_loc_device = batch.query_start_loc_device
        assert query_start_loc_device is not None
        assert self._paged_prefill is not None
        decode_count = 0
        for phase in batch.phases:
            if phase is not BatchPhase.DECODE:
                break
            decode_count += 1
        if any(
            phase is BatchPhase.DECODE
            for phase in batch.phases[decode_count:]
        ):
            raise ModelImplementationError(
                "mixed batch must keep decode rows contiguous"
            )
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
            )
        if decode_count < batch.num_sequences:
            self._paged_prefill(
                query,
                key_cache,
                value_cache,
                batch.block_tables[decode_count:],
                query_start_loc_device[decode_count:],
                batch.positions,
                query_lengths=batch.query_lengths[decode_count:],
                context_lengths=batch.context_lengths[decode_count:],
                page_size=batch.page_size,
                output=output,
            )
        return output

    def _mixed_attention_tensorplay(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        output = tensorplay.empty_like(query)
        decode_rows: list[int] = []
        decode_tokens: list[int] = []
        for sequence_index, phase in enumerate(batch.phases):
            if phase is BatchPhase.DECODE:
                decode_rows.append(sequence_index)
                decode_tokens.append(batch.query_start_loc[sequence_index])
        if decode_rows:
            row_indices = tensorplay.tensor(
                decode_rows,
                dtype=tensorplay.long,
                device=query.device,
            )
            token_indices = tensorplay.tensor(
                decode_tokens,
                dtype=tensorplay.long,
                device=query.device,
            )
            decoded = self._paged_decode(
                query.index_select(0, token_indices),
                key_cache,
                value_cache,
                batch.block_tables.index_select(0, row_indices),
                batch.sequence_lengths.index_select(0, row_indices),
                batch.page_size,
            )
            output.index_copy_(0, token_indices, decoded)

        for sequence_index, phase in enumerate(batch.phases):
            if phase is BatchPhase.DECODE:
                continue
            start = batch.query_start_loc[sequence_index]
            end = batch.query_start_loc[sequence_index + 1]
            query_length = end - start
            context_length = batch.context_lengths[sequence_index]
            sequence_length = context_length + query_length
            num_pages = math.ceil(sequence_length / batch.page_size)
            page_ids = batch.block_tables[
                sequence_index,
                :num_pages,
            ].long()
            keys = (
                key_cache.index_select(0, page_ids)
                .reshape(-1, self.num_kv_heads, self.head_dim)[:sequence_length]
            )
            values = (
                value_cache.index_select(0, page_ids)
                .reshape(-1, self.num_kv_heads, self.head_dim)[:sequence_length]
            )
            keys = keys.repeat_interleave(self.query_per_kv, dim=1)
            values = values.repeat_interleave(self.query_per_kv, dim=1)
            query_states = query[start:end].transpose(0, 1).unsqueeze(0)
            key_states = keys.transpose(0, 1).unsqueeze(0)
            value_states = values.transpose(0, 1).unsqueeze(0)
            query_positions = tensorplay.arange(
                context_length,
                sequence_length,
                device=query.device,
            )
            key_positions = tensorplay.arange(sequence_length, device=query.device)
            causal_mask = (
                key_positions[None, :] <= query_positions[:, None]
            )[None, None, :, :]
            attended = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=0.0,
                scale=self.head_dim**-0.5,
            )
            output[start:end] = attended.squeeze(0).transpose(0, 1)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_index: int,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        rotary: RotaryCache,
        kernel_bindings: KernelBindings,
        use_decode_gemv: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            config,
            device=device,
            dtype=dtype,
            rotary=rotary,
            kernel_bindings=kernel_bindings,
            use_decode_gemv=use_decode_gemv,
        )
        self.mlp: DenseMLP | SparseMoE
        if config.is_moe_layer(layer_index):
            self.mlp = SparseMoE(
                config,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
        else:
            self.mlp = DenseMLP(
                config,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
                use_decode_gemv=use_decode_gemv,
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.layer_index = layer_index

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        residual: tensorplay.Tensor | None,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm.with_residual(
                hidden_states,
                residual,
            )
        hidden_states = self.self_attn(
            hidden_states,
            layer_index=self.layer_index,
            batch=batch,
            cache=cache,
        )
        hidden_states, residual = self.post_attention_layernorm.with_residual(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states, decode=batch.all_decode)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
        use_decode_gemv: bool = False,
    ) -> None:
        super().__init__()
        self.embed_tokens = TokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.rotary = RotaryCache(config, device=device)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    layer_index,
                    device=device,
                    dtype=dtype,
                    rotary=self.rotary,
                    kernel_bindings=kernel_bindings,
                    use_decode_gemv=use_decode_gemv,
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        hidden_states = (
            self.embed_tokens(batch.input_ids)
            if batch.input_embeddings is None
            else batch.input_embeddings
        )
        residual: tensorplay.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                batch=batch,
                cache=cache,
            )
        assert residual is not None
        hidden_states, _ = self.norm.with_residual(
            hidden_states,
            residual,
        )
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
        use_decode_gemv: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.kernel_policy = kernel_policy
        if kernel_bindings is None:
            kernel_bindings = bind_kernel_plan(
                kernel_policy,
                device=self.device,
                dtype=dtype,
                model_type="qwen3",
            )
        if kernel_bindings.policy != kernel_policy:
            raise ModelImplementationError(
                "kernel bindings do not match the model kernel policy"
            )
        self.kernel_bindings = kernel_bindings
        self.moe_workspace: GroupedMoEWorkspace | None = None
        self.model = Qwen3Model(
            config,
            device=self.device,
            dtype=dtype,
            kernel_bindings=self.kernel_bindings,
            use_decode_gemv=use_decode_gemv,
        )
        self.lm_head_weight: nn.Parameter | None
        if config.tie_word_embeddings:
            self.register_parameter("lm_head_weight", None)
            self.lm_head_weight = None
        else:
            self.lm_head_weight = _parameter(
                (config.vocab_size, config.hidden_size),
                device=self.device,
                dtype=dtype,
            )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: tensorplay.device | str = "cuda",
        dtype: tensorplay.dtype = tensorplay.float16,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
    ) -> tuple[Qwen3ForCausalLM, WeightLoadReport]:
        config = Qwen3Config.from_path(model_path)
        model = cls(
            config,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
        report = model.load_weights(model_path)
        model.eval()
        return model, report

    def configure_moe_workspace(self, max_batched_tokens: int) -> int:
        """Allocate the one MoE scratch arena owned by this Replica."""

        if max_batched_tokens <= 0:
            raise ModelImplementationError(
                "MoE workspace token budget must be positive"
            )
        moe_layers = [
            layer.mlp
            for layer in self.model.layers
            if isinstance(layer.mlp, SparseMoE)
        ]
        if not self.kernel_policy.use_triton or not moe_layers:
            return 0
        first = moe_layers[0]
        geometry = (
            first.gate_up_weight.shape[0],
            first.gate_up_weight.shape[2],
            first.gate_up_weight.shape[1] // 2,
        )
        if any(
            (
                mlp.gate_up_weight.shape[0],
                mlp.gate_up_weight.shape[2],
                mlp.gate_up_weight.shape[1] // 2,
            )
            != geometry
            for mlp in moe_layers
        ):
            raise ModelImplementationError(
                "shared MoE workspace requires one expert geometry"
            )
        num_experts, hidden_size, intermediate_size = geometry
        workspace = GroupedMoEWorkspace.allocate(
            max_tokens=max_batched_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=first.gate_up_weight.dtype,
            device=first.gate_up_weight.device,
            max_top_k=max(mlp.top_k for mlp in moe_layers),
        )
        self.moe_workspace = workspace
        for mlp in moe_layers:
            mlp.workspace = workspace
        return workspace.nbytes

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        hidden_states = self.model(batch, cache)
        lm_head = (
            self.model.embed_tokens.weight
            if self.config.tie_word_embeddings
            else self.lm_head_weight
        )
        assert lm_head is not None
        # Pure decode has exactly one token and one sampled row per sequence.
        # Scheduler staging and CUDA Graph buckets both publish sample_indices
        # as arange(B), so gathering and masking those rows only adds kernels
        # to every decode replay.
        if batch.all_decode:
            return F.linear(hidden_states, lm_head)
        valid_sample_indices = batch.sample_indices.clamp_min(0).long()
        sampled_hidden = hidden_states.index_select(0, valid_sample_indices)
        logits = F.linear(sampled_hidden, lm_head)
        valid = batch.sample_indices >= 0
        return tensorplay.where(valid[:, None], logits, 0)

    @tensorplay.inference_mode()
    def warmup_prefill_kernels(
        self,
        *,
        max_batched_tokens: int,
    ) -> tuple[int, ...]:
        """Compile every shape-class selected by the top-1 prefill policy.

        The tensors never enter the scheduler or KV cache. One representative
        MoE layer per unique weight geometry is sufficient because Triton
        specializations depend on geometry and policy, not parameter addresses.
        """

        if max_batched_tokens <= 0:
            raise ModelImplementationError(
                "prefill warmup token budget must be positive"
            )
        if not self.kernel_policy.use_triton:
            return ()
        grouped_min = self.kernel_policy.grouped_moe_min_tokens
        if grouped_min > max_batched_tokens:
            return ()
        token_counts: list[int] = []
        small_count = max(grouped_min, min(64, max_batched_tokens))
        if small_count < 512:
            token_counts.append(small_count)
        large_count = max(512, grouped_min)
        if large_count <= max_batched_tokens:
            token_counts.append(large_count)

        warmed_geometries: set[tuple[int, int, int, int, tensorplay.dtype]] = set()
        for layer in self.model.layers:
            mlp = layer.mlp
            if not isinstance(mlp, SparseMoE):
                continue
            geometry = (
                mlp.gate_up_weight.shape[0],
                mlp.gate_up_weight.shape[2],
                mlp.gate_up_weight.shape[1] // 2,
                mlp.top_k,
                mlp.gate_up_weight.dtype,
            )
            if geometry in warmed_geometries:
                continue
            warmed_geometries.add(geometry)
            hidden_size = mlp.gate_up_weight.shape[2]
            for token_count in token_counts:
                hidden_states = tensorplay.zeros(
                    (token_count, hidden_size),
                    dtype=mlp.gate_up_weight.dtype,
                    device=mlp.gate_up_weight.device,
                )
                mlp(hidden_states, decode=False)
        return tuple(token_counts) if warmed_geometries else ()

    def load_weights(
        self,
        model_path: str | Path,
        *,
        source: SafetensorsSource | None = None,
        key_prefix: str = "",
        strict: bool = True,
    ) -> WeightLoadReport:
        """Load Qwen3 tensors, optionally from a prefixed sub-checkpoint.

        OXR stores its LLM under ``llm.``; passing ``key_prefix`` strips that
        namespace before the existing packed-weight routing runs.
        """

        parameters = dict(self.named_parameters())
        loaded_targets: set[str] = set()
        packed_parts: dict[str, set[str]] = {}
        expert_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
            r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight"
        )
        dense_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.mlp\."
            r"(?P<projection>gate_proj|up_proj)\.weight"
        )
        qkv_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.self_attn\."
            r"(?P<projection>q_proj|k_proj|v_proj)\.weight"
        )

        def copy_parameter(target_name: str, tensor: tensorplay.Tensor) -> None:
            try:
                target = parameters[target_name]
            except KeyError as exc:
                raise ModelImplementationError(
                    f"checkpoint tensor has no first-party target: {target_name}"
                ) from exc
            if tuple(target.shape) != tuple(tensor.shape):
                raise ModelImplementationError(
                    f"shape mismatch for {target_name}: "
                    f"expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
                )
            target.copy_(tensor.to(device=self.device, dtype=self.dtype))
            loaded_targets.add(target_name)

        @tensorplay.no_grad()
        def consume(source_name: str, tensor: tensorplay.Tensor) -> None:
            if key_prefix:
                if not source_name.startswith(key_prefix):
                    return
                source_name = source_name[len(key_prefix) :]
            if source_name == "lm_head.weight" and self.config.tie_word_embeddings:
                return
            qkv_match = qkv_pattern.fullmatch(source_name)
            if qkv_match is not None:
                layer_index = int(qkv_match.group("layer"))
                projection = qkv_match.group("projection")
                attention = self.model.layers[layer_index].self_attn
                target_name = (
                    f"model.layers.{layer_index}.self_attn.qkv_proj.weight"
                )
                offsets = {
                    "q_proj": (0, attention.query_width),
                    "k_proj": (
                        attention.query_width,
                        attention.query_width + attention.kv_width,
                    ),
                    "v_proj": (
                        attention.query_width + attention.kv_width,
                        attention.query_width + 2 * attention.kv_width,
                    ),
                }
                start, end = offsets[projection]
                target = attention.qkv_proj.weight[start:end]
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ModelImplementationError(
                        f"shape mismatch for packed {source_name}"
                    )
                target.copy_(tensor.to(device=self.device, dtype=self.dtype))
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                }:
                    loaded_targets.add(target_name)
                return

            expert_match = expert_pattern.fullmatch(source_name)
            if expert_match is not None:
                layer_index = int(expert_match.group("layer"))
                expert_index = int(expert_match.group("expert"))
                projection = expert_match.group("projection")
                mlp = self.model.layers[layer_index].mlp
                if not isinstance(mlp, SparseMoE):
                    raise ModelImplementationError(
                        f"expert weight supplied for dense layer {layer_index}"
                    )
                if projection == "down_proj":
                    target_name = (
                        f"model.layers.{layer_index}.mlp.down_weight"
                    )
                    target = mlp.down_weight[expert_index]
                else:
                    target_name = (
                        f"model.layers.{layer_index}.mlp.gate_up_weight"
                    )
                    offset = (
                        0
                        if projection == "gate_proj"
                        else self.config.moe_intermediate_size
                    )
                    target = mlp.gate_up_weight[
                        expert_index,
                        offset : offset + self.config.moe_intermediate_size,
                    ]
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ModelImplementationError(
                        f"shape mismatch for packed {source_name}"
                    )
                target.copy_(tensor.to(device=self.device, dtype=self.dtype))
                packed_parts.setdefault(target_name, set()).add(
                    f"{expert_index}:{projection}"
                )
                expected_parts = (
                    self.config.num_experts
                    if projection == "down_proj"
                    else self.config.num_experts * 2
                )
                relevant = {
                    part
                    for part in packed_parts[target_name]
                    if (
                        projection == "down_proj"
                        and part.endswith(":down_proj")
                    )
                    or (
                        projection != "down_proj"
                        and not part.endswith(":down_proj")
                    )
                }
                if len(relevant) == expected_parts:
                    loaded_targets.add(target_name)
                return

            dense_match = dense_pattern.fullmatch(source_name)
            if dense_match is not None:
                layer_index = int(dense_match.group("layer"))
                projection = dense_match.group("projection")
                mlp = self.model.layers[layer_index].mlp
                if not isinstance(mlp, DenseMLP):
                    raise ModelImplementationError(
                        f"dense MLP weight supplied for MoE layer {layer_index}"
                    )
                target_name = (
                    f"model.layers.{layer_index}.mlp.gate_up_weight"
                )
                offset = 0 if projection == "gate_proj" else self.config.intermediate_size
                target = mlp.gate_up_weight[
                    offset : offset + self.config.intermediate_size
                ]
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ModelImplementationError(
                        f"shape mismatch for packed {source_name}"
                    )
                target.copy_(tensor.to(device=self.device, dtype=self.dtype))
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {"gate_proj", "up_proj"}:
                    loaded_targets.add(target_name)
                return

            target_name = (
                "lm_head_weight"
                if source_name == "lm_head.weight"
                else source_name
            )
            copy_parameter(target_name, tensor)

        if source is None:
            source = SafetensorsSource(model_path)
        report = source.stream(consume)
        missing = sorted(set(parameters) - loaded_targets)
        if strict and missing:
            raise ModelImplementationError(
                f"checkpoint did not initialize parameters: {missing}"
            )
        return report
