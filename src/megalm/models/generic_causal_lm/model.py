from __future__ import annotations

import re
from pathlib import Path

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings, bind_kernel_plan
from megalm.kernels.ops import GroupedMoEWorkspace, KernelPolicy
from megalm.loading import SafetensorsSource, WeightLoadReport
from megalm.models.generic_causal_lm.config import GenericCausalLMConfig
from megalm.models.qwen3 import (
    DenseMLP,
    Qwen3Attention,
    RMSNorm,
    SparseMoE,
    TokenEmbedding,
)
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class GenericCausalLMError(RuntimeError):
    """Raised when a Llama-family checkpoint violates the native contract."""


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


class GenericRotaryCache(nn.Module):
    """Paged RoPE cache with optional partial, interleaved channel layout."""

    def __init__(
        self,
        config: GenericCausalLMConfig,
        *,
        device: tensorplay.device,
    ) -> None:
        super().__init__()
        head_dim = config.head_dim
        rotary_dim = int(head_dim * config.partial_rotary_factor)
        half_head = head_dim // 2
        half_rotary = rotary_dim // 2
        inverse_frequency = 1.0 / (
            config.rope_theta
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
        positions = tensorplay.arange(
            config.max_position_embeddings,
            dtype=tensorplay.float32,
            device=device,
        )
        frequencies = tensorplay.outer(positions, inverse_frequency)
        cos = tensorplay.cat(
            (
                frequencies.cos(),
                tensorplay.ones(
                    (config.max_position_embeddings, half_head - half_rotary),
                    dtype=tensorplay.float32,
                    device=device,
                ),
            ),
            dim=-1,
        )
        sin = tensorplay.cat(
            (
                frequencies.sin(),
                tensorplay.zeros(
                    (config.max_position_embeddings, half_head - half_rotary),
                    dtype=tensorplay.float32,
                    device=device,
                ),
            ),
            dim=-1,
        )
        self.register_buffer("cos", cos.contiguous(), persistent=False)
        self.register_buffer("sin", sin.contiguous(), persistent=False)
        permutation = _rope_permutation(
            head_dim,
            rotary_dim,
            config.rotary_style,
        )
        if permutation is None:
            self.register_buffer("permutation", None, persistent=False)
        else:
            self.register_buffer(
                "permutation",
                permutation.to(device=device),
                persistent=False,
            )


def _rope_permutation(
    head_dim: int,
    rotary_dim: int,
    style: str,
) -> tensorplay.Tensor | None:
    if rotary_dim == head_dim and style == "neox":
        return None
    tail = tensorplay.arange(
        rotary_dim,
        head_dim,
        dtype=tensorplay.long,
    )
    padding = head_dim // 2 - rotary_dim // 2
    if style == "interleaved":
        first = tensorplay.cat(
            (
                tensorplay.arange(0, rotary_dim, 2, dtype=tensorplay.long),
                tail[:padding],
            )
        )
        second = tensorplay.cat(
            (
                tensorplay.arange(1, rotary_dim, 2, dtype=tensorplay.long),
                tail[padding:],
            )
        )
    else:
        first = tensorplay.cat(
            (
                tensorplay.arange(
                    0,
                    rotary_dim // 2,
                    dtype=tensorplay.long,
                ),
                tail[:padding],
            )
        )
        second = tensorplay.cat(
            (
                tensorplay.arange(
                    rotary_dim // 2,
                    rotary_dim,
                    dtype=tensorplay.long,
                ),
                tail[padding:],
            )
        )
    return tensorplay.cat((first, second))


class BiasedLinear(nn.Module):
    def __init__(
        self,
        output_features: int,
        input_features: int,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.weight = _parameter(
            (output_features, input_features),
            device=device,
            dtype=dtype,
        )
        self.bias = _parameter((output_features,), device=device, dtype=dtype)

    def forward(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
        return F.linear(x, self.weight, self.bias)


class GenericSparseMoE(SparseMoE):
    """Top-k routing with the fused Triton router for every top-k."""

    def __init__(
        self,
        config: GenericCausalLMConfig,
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


class GenericAttention(Qwen3Attention):
    """Qwen3 attention geometry with optional QK-Norm and QKV bias."""

    def __init__(
        self,
        config: GenericCausalLMConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        rotary: GenericRotaryCache,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__(
            config,
            device=device,
            dtype=dtype,
            rotary=rotary,
            kernel_bindings=kernel_bindings,
        )
        self.rope_cache_write = kernel_bindings.core.rope_cache_write
        if config.qkv_bias:
            self.qkv_proj = BiasedLinear(
                self.query_width + 2 * self.kv_width,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
        if not config.qk_norm:
            self.q_norm = None
            self.k_norm = None

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
        if self.q_norm is None or self.k_norm is None:
            query, key, value = qkv.split(
                (self.query_width, self.kv_width, self.kv_width),
                dim=-1,
            )
            if self.rotary.permutation is not None:
                query = query[..., self.rotary.permutation]
                key = key[..., self.rotary.permutation]
            query = self.rope_cache_write(
                query.view(-1, self.num_query_heads, self.head_dim),
                key.view(-1, self.num_kv_heads, self.head_dim),
                value.view(-1, self.num_kv_heads, self.head_dim),
                batch.positions,
                batch.slot_mapping,
                self.rotary.cos,
                self.rotary.sin,
                key_cache,
                value_cache,
                batch.token_active,
            )
        else:
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
            attended = (
                self._mixed_attention_triton
                if self._paged_prefill is not None
                else self._mixed_attention_tensorplay
            )(query, key_cache, value_cache, batch)
        return self.o_proj(attended.reshape(token_count, -1))


class GenericDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GenericCausalLMConfig,
        layer_index: int,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        rotary: GenericRotaryCache,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.config = config
        self.self_attn = GenericAttention(
            config,
            device=device,
            dtype=dtype,
            rotary=rotary,
            kernel_bindings=kernel_bindings,
        )
        self.mlp: DenseMLP | GenericSparseMoE
        if config.is_moe_layer(layer_index):
            self.mlp = GenericSparseMoE(
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
        hidden_states = hidden_states * self.config.residual_multiplier
        hidden_states, residual = self.post_attention_layernorm.with_residual(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states, decode=batch.all_decode)
        return hidden_states * self.config.residual_multiplier, residual


class GenericCausalLMModel(nn.Module):
    def __init__(
        self,
        config: GenericCausalLMConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.embed_tokens = TokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.rotary = GenericRotaryCache(config, device=device)
        self.layers = nn.ModuleList(
            [
                GenericDecoderLayer(
                    config,
                    layer_index,
                    device=device,
                    dtype=dtype,
                    rotary=self.rotary,
                    kernel_bindings=kernel_bindings,
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
        hidden_states = self.embed_tokens(batch.input_ids)
        if self.config.embedding_multiplier != 1.0:
            hidden_states = hidden_states * self.config.embedding_multiplier
        residual: tensorplay.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                batch=batch,
                cache=cache,
            )
        assert residual is not None
        hidden_states, _ = self.norm.with_residual(hidden_states, residual)
        return hidden_states


class GenericCausalLM(nn.Module):
    def __init__(
        self,
        config: GenericCausalLMConfig,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
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
                model_type=config.model_type,
            )
        if kernel_bindings.policy != kernel_policy:
            raise GenericCausalLMError(
                "kernel bindings do not match the model kernel policy"
            )
        self.kernel_bindings = kernel_bindings
        self.model = GenericCausalLMModel(
            config,
            device=self.device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
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
    ) -> tuple[GenericCausalLM, WeightLoadReport]:
        config = GenericCausalLMConfig.from_path(model_path)
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
        scale = self.config.logit_scale
        if batch.all_decode:
            logits = F.linear(hidden_states, lm_head)
            return logits * scale if scale != 1.0 else logits
        indices = batch.sample_indices.clamp_min(0).long()
        logits = F.linear(hidden_states.index_select(0, indices), lm_head)
        if scale != 1.0:
            logits = logits * scale
        return tensorplay.where(
            (batch.sample_indices >= 0)[:, None],
            logits,
            0,
        )

    def configure_moe_workspace(self, max_batched_tokens: int) -> int:
        if max_batched_tokens <= 0:
            raise GenericCausalLMError(
                "MoE workspace token budget must be positive"
            )
        moe_layers = [
            layer.mlp
            for layer in self.model.layers
            if isinstance(layer.mlp, GenericSparseMoE)
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
            raise GenericCausalLMError(
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
        for mlp in moe_layers:
            mlp.workspace = workspace
        return workspace.nbytes

    @tensorplay.inference_mode()
    def warmup_prefill_kernels(
        self,
        *,
        max_batched_tokens: int,
    ) -> tuple[int, ...]:
        if max_batched_tokens <= 0:
            raise GenericCausalLMError(
                "prefill warmup token budget must be positive"
            )
        moe_layers = [
            layer.mlp
            for layer in self.model.layers
            if isinstance(layer.mlp, GenericSparseMoE)
        ]
        if not self.kernel_policy.use_triton or not moe_layers:
            return ()
        grouped_min = self.kernel_policy.grouped_moe_min_tokens
        if grouped_min > max_batched_tokens:
            return ()
        token_counts = [max(grouped_min, min(64, max_batched_tokens))]
        large_count = max(512, grouped_min)
        if large_count <= max_batched_tokens:
            token_counts.append(large_count)
        warmed: set[tuple[int, int, int, int, object]] = set()
        for mlp in moe_layers:
            geometry = (
                mlp.gate_up_weight.shape[0],
                mlp.gate_up_weight.shape[2],
                mlp.gate_up_weight.shape[1] // 2,
                mlp.top_k,
                mlp.gate_up_weight.dtype,
            )
            if geometry in warmed:
                continue
            warmed.add(geometry)
            for token_count in token_counts:
                hidden_states = tensorplay.zeros(
                    (
                        token_count,
                        mlp.gate_up_weight.shape[2],
                    ),
                    dtype=mlp.gate_up_weight.dtype,
                    device=mlp.gate_up_weight.device,
                )
                mlp(hidden_states, decode=False)
        return tuple(token_counts)

    def load_weights(self, model_path: str | Path) -> WeightLoadReport:
        config = self.config
        parameters = dict(self.named_parameters())
        loaded: set[str] = set()
        packed_parts: dict[str, set[str]] = {}
        qkv_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.self_attn\."
            r"(?P<projection>q_proj|k_proj|v_proj)\.(?P<field>weight|bias)"
        )
        packed_qkv_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.self_attn\.qkv_proj\.weight"
        )
        packed_gate_up_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.mlp\.gate_up_proj\.weight"
        )
        intern_wqkv_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.attention\.wqkv\.weight"
        )
        intern_wo_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.attention\.wo\.weight"
        )
        intern_norm_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\."
            r"(?P<norm>attention_norm|ffn_norm)\.weight"
        )
        intern_mlp_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.feed_forward\."
            r"(?P<projection>w1|w2|w3)\.weight"
        )
        dense_pattern = re.compile(
            r"model\.layers\.(?P<layer>\d+)\.mlp\."
            r"(?P<projection>gate_proj|up_proj|down_proj)\.weight"
        )
        if config.model_type == "mixtral":
            expert_pattern = re.compile(
                r"model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\."
                r"(?P<expert>\d+)\.(?P<projection>w1|w2|w3)\.weight"
            )
            router_pattern = re.compile(
                r"model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.gate\.weight"
            )
            projection_names = {
                "w1": "gate_proj",
                "w3": "up_proj",
                "w2": "down_proj",
            }
        else:
            expert_pattern = re.compile(
                r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
                r"(?P<expert>\d+)\."
                r"(?P<projection>gate_proj|up_proj|down_proj)\.weight"
            )
            router_pattern = re.compile(
                r"model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight"
            )
            projection_names = {}

        def copy(target_name: str, tensor: tensorplay.Tensor) -> None:
            try:
                target = parameters[target_name]
            except KeyError as exc:
                raise GenericCausalLMError(
                    f"checkpoint tensor has no native target: {target_name}"
                ) from exc
            if tuple(target.shape) != tuple(tensor.shape):
                raise GenericCausalLMError(
                    f"shape mismatch for {target_name}: expected "
                    f"{tuple(target.shape)}, got {tuple(tensor.shape)}"
                )
            target.copy_(tensor.to(device=self.device, dtype=self.dtype))
            loaded.add(target_name)

        def copy_slice(
            target: tensorplay.Tensor,
            tensor: tensorplay.Tensor,
            source_name: str,
        ) -> None:
            if tuple(target.shape) != tuple(tensor.shape):
                raise GenericCausalLMError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(tensor.to(device=self.device, dtype=self.dtype))

        @tensorplay.no_grad()
        def consume(source_name: str, tensor: tensorplay.Tensor) -> None:
            if source_name == "lm_head.weight" and config.tie_word_embeddings:
                return
            if source_name == "model.tok_embeddings.weight":
                copy("model.embed_tokens.weight", tensor)
                return
            if source_name == "output.weight":
                if config.tie_word_embeddings:
                    return
                copy("lm_head_weight", tensor)
                return

            match = intern_norm_pattern.fullmatch(source_name)
            if match is not None:
                target_name = (
                    "input_layernorm"
                    if match.group("norm") == "attention_norm"
                    else "post_attention_layernorm"
                )
                copy(
                    f"model.layers.{int(match.group('layer'))}.{target_name}.weight",
                    tensor,
                )
                return

            match = intern_wqkv_pattern.fullmatch(source_name)
            if match is not None:
                attention = self.model.layers[
                    int(match.group("layer"))
                ].self_attn
                groups = attention.query_width // attention.kv_width
                packed = tensor.to(
                    device=self.device,
                    dtype=self.dtype,
                ).view(
                    attention.num_kv_heads,
                    groups + 2,
                    attention.head_dim,
                    config.hidden_size,
                )
                query = packed[:, :groups].reshape(
                    attention.query_width,
                    config.hidden_size,
                )
                key = packed[:, groups].reshape(
                    attention.kv_width,
                    config.hidden_size,
                )
                value = packed[:, groups + 1].reshape(
                    attention.kv_width,
                    config.hidden_size,
                )
                target_name = (
                    f"model.layers.{int(match.group('layer'))}."
                    "self_attn.qkv_proj.weight"
                )
                copy_slice(
                    attention.qkv_proj.weight,
                    tensorplay.cat((query, key, value), dim=0),
                    source_name,
                )
                loaded.add(target_name)
                return

            match = intern_wo_pattern.fullmatch(source_name)
            if match is not None:
                copy(
                    f"model.layers.{int(match.group('layer'))}."
                    "self_attn.o_proj.weight",
                    tensor,
                )
                return

            match = intern_mlp_pattern.fullmatch(source_name)
            if match is not None:
                layer_index = int(match.group("layer"))
                projection = {
                    "w1": "gate_proj",
                    "w3": "up_proj",
                    "w2": "down_proj",
                }[match.group("projection")]
                mlp = self.model.layers[layer_index].mlp
                if not isinstance(mlp, DenseMLP):
                    raise GenericCausalLMError(
                        "InternLM2 MLP weight supplied for a MoE layer"
                    )
                if projection == "down_proj":
                    copy(
                        f"model.layers.{layer_index}.mlp.down_proj.weight",
                        tensor,
                    )
                    return
                target_name = f"model.layers.{layer_index}.mlp.gate_up_weight"
                start = 0 if projection == "gate_proj" else config.intermediate_size
                copy_slice(
                    mlp.gate_up_weight[
                        start : start + config.intermediate_size
                    ],
                    tensor,
                    source_name,
                )
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {"gate_proj", "up_proj"}:
                    loaded.add(target_name)
                return

            match = packed_qkv_pattern.fullmatch(source_name)
            if match is not None:
                copy(
                    f"model.layers.{int(match.group('layer'))}."
                    "self_attn.qkv_proj.weight",
                    tensor,
                )
                return

            match = packed_gate_up_pattern.fullmatch(source_name)
            if match is not None:
                copy(
                    f"model.layers.{int(match.group('layer'))}."
                    "mlp.gate_up_weight",
                    tensor,
                )
                return

            match = qkv_pattern.fullmatch(source_name)
            if match is not None:
                layer_index = int(match.group("layer"))
                projection = match.group("projection")
                field = match.group("field")
                attention = self.model.layers[layer_index].self_attn
                target_name = f"model.layers.{layer_index}.self_attn.qkv_proj.{field}"
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
                target = getattr(attention.qkv_proj, field)[start:end]
                copy_slice(target, tensor, source_name)
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                }:
                    loaded.add(target_name)
                return

            match = dense_pattern.fullmatch(source_name)
            if match is not None:
                layer_index = int(match.group("layer"))
                projection = match.group("projection")
                mlp = self.model.layers[layer_index].mlp
                if not isinstance(mlp, DenseMLP):
                    raise GenericCausalLMError(
                        f"dense MLP weight supplied for MoE layer {layer_index}"
                    )
                if projection == "down_proj":
                    copy(
                        f"model.layers.{layer_index}.mlp.down_proj.weight",
                        tensor,
                    )
                    return
                target_name = f"model.layers.{layer_index}.mlp.gate_up_weight"
                start = 0 if projection == "gate_proj" else config.intermediate_size
                copy_slice(
                    mlp.gate_up_weight[
                        start : start + config.intermediate_size
                    ],
                    tensor,
                    source_name,
                )
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {"gate_proj", "up_proj"}:
                    loaded.add(target_name)
                return

            match = expert_pattern.fullmatch(source_name)
            if match is not None:
                layer_index = int(match.group("layer"))
                expert_index = int(match.group("expert"))
                projection = projection_names.get(
                    match.group("projection"),
                    match.group("projection"),
                )
                mlp = self.model.layers[layer_index].mlp
                if not isinstance(mlp, GenericSparseMoE):
                    raise GenericCausalLMError(
                        f"expert weight supplied for dense layer {layer_index}"
                    )
                if projection == "down_proj":
                    target_name = f"model.layers.{layer_index}.mlp.down_weight"
                    target = mlp.down_weight[expert_index]
                    expected_parts = config.num_experts
                else:
                    target_name = f"model.layers.{layer_index}.mlp.gate_up_weight"
                    offset = (
                        0
                        if projection == "gate_proj"
                        else config.moe_intermediate_size
                    )
                    target = mlp.gate_up_weight[
                        expert_index,
                        offset : offset + config.moe_intermediate_size,
                    ]
                    expected_parts = config.num_experts * 2
                copy_slice(target, tensor, source_name)
                packed_parts.setdefault(target_name, set()).add(
                    f"{expert_index}:{projection}"
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
                    loaded.add(target_name)
                return

            match = router_pattern.fullmatch(source_name)
            if match is not None:
                copy(
                    f"model.layers.{int(match.group('layer'))}.mlp.gate.weight",
                    tensor,
                )
                return

            target_name = source_name
            if source_name == "lm_head.weight":
                target_name = "lm_head_weight"
            copy(target_name, tensor)

        source = SafetensorsSource(model_path)
        report = source.stream(consume)
        missing = sorted(set(parameters) - loaded)
        if missing:
            raise GenericCausalLMError(
                f"checkpoint did not initialize parameters: {missing}"
            )
        return report


__all__ = ["GenericAttention", "GenericCausalLM", "GenericCausalLMModel"]
