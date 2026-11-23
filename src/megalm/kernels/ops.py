from __future__ import annotations

from dataclasses import dataclass

import tensorplay
import tensorplay.nn.functional as F

from megalm.kernels import fallback
from megalm.kernels.greedy_argmax import (
    greedy_argmax as triton_greedy_argmax,
)
from megalm.kernels.moe_decode import (
    routed_moe_decode as triton_routed_moe_decode,
)
from megalm.kernels.moe_grouped import (
    routed_moe_grouped_tensorcore as triton_routed_moe_grouped,
)
from megalm.kernels.moe_top1 import (
    moe_top1 as triton_moe_top1,
    moe_top1_from_hidden as triton_moe_top1_from_hidden,
)
from megalm.kernels.moe_topk import (
    moe_topk_softmax as triton_moe_topk_softmax,
)
from megalm.kernels.paged_decode import (
    paged_decode_attention as triton_paged_decode,
)
from megalm.kernels.paged_prefill import (
    paged_prefill_attention as triton_paged_prefill,
)
from megalm.kernels.rms_norm import (
    fused_add_rms_norm as triton_fused_add_rms_norm,
    rms_norm as triton_rms_norm,
)
from megalm.kernels.rotary_cache import (
    fused_qkv_norm_rope_cache_write as triton_fused_qkv_rope_cache,
    rope_and_cache_write as triton_rope_cache,
)
from megalm.kernels.silu_mul import silu_and_mul as triton_silu_mul


@dataclass(frozen=True, slots=True)
class KernelPolicy:
    use_triton: bool = True
    moe_max_tokens: int = 64
    moe_tensorcore_min_tokens: int = 8
    grouped_moe_min_tokens: int = 8

    def __post_init__(self) -> None:
        if min(
            self.moe_max_tokens,
            self.moe_tensorcore_min_tokens,
            self.grouped_moe_min_tokens,
        ) <= 0:
            raise ValueError("MoE shape guards must be positive")
        if self.grouped_moe_min_tokens < self.moe_tensorcore_min_tokens:
            raise ValueError(
                "grouped MoE guard cannot precede the Tensor Core guard"
            )

    def triton_enabled_for(self, tensor: tensorplay.Tensor) -> bool:
        return self.use_triton and tensor.is_cuda


DEFAULT_POLICY = KernelPolicy()


@dataclass(slots=True)
class GroupedMoEWorkspace:
    """Stable scratch tensors shared by serial MoE layers.

    One Replica executes one model batch at a time on one ordered CUDA stream,
    so every MoE layer can safely reuse these addresses. Keeping the buffers
    outside ``forward`` also makes the grouped path CUDA-Graph safe. The arena
    is sized for ``max_top_k`` routes per token so top-k>1 layers can use the
    same grouped tensor-core kernels as top-1 layers.
    """

    max_tokens: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    max_top_k: int
    compact_min_tokens: int
    expert_capacity: int
    expert_counts: tensorplay.Tensor
    expert_cursors: tensorplay.Tensor
    topk_ids: tensorplay.Tensor
    topk_weights: tensorplay.Tensor
    sorted_hidden: tensorplay.Tensor
    sorted_indices: tensorplay.Tensor
    sorted_weights: tensorplay.Tensor
    intermediate: tensorplay.Tensor
    output: tensorplay.Tensor
    route_output: tensorplay.Tensor | None = None

    @staticmethod
    def _capacities(
        max_tokens: int,
        num_experts: int,
        max_top_k: int = 1,
    ) -> tuple[int, int, int]:
        if min(max_tokens, num_experts, max_top_k) <= 0:
            raise ValueError("MoE workspace dimensions must be positive")
        route_count = max_tokens * max_top_k
        compact_min_tokens = min(
            512,
            max(8, 2048 // num_experts + 1),
        )
        padded_capacity = num_experts * min(
            route_count,
            compact_min_tokens - 1,
        )
        compact_capacity = (
            route_count if route_count >= compact_min_tokens else 0
        )
        return (
            compact_min_tokens,
            padded_capacity,
            max(padded_capacity, compact_capacity),
        )

    @classmethod
    def estimate_nbytes(
        cls,
        *,
        max_tokens: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: tensorplay.dtype,
        max_top_k: int = 1,
    ) -> int:
        """Return the exact stable scratch size without allocating a tensor."""

        if min(
            max_tokens,
            num_experts,
            hidden_size,
            intermediate_size,
            max_top_k,
        ) <= 0:
            raise ValueError("MoE workspace dimensions must be positive")
        _, padded_capacity, expert_capacity = cls._capacities(
            max_tokens,
            num_experts,
            max_top_k,
        )
        element_size = tensorplay.empty((), dtype=dtype).element_size()
        route_bytes = (
            max_tokens * max_top_k * hidden_size * 4
            if max_top_k > 1
            else 0
        )
        return (
            num_experts * 4
            + num_experts * 4
            + max_tokens * 8
            + max_tokens * 4
            + padded_capacity * hidden_size * element_size
            + expert_capacity * 8
            + expert_capacity * 4
            + expert_capacity * intermediate_size * element_size
            + max_tokens * hidden_size * element_size
            + route_bytes
        )

    @classmethod
    def allocate(
        cls,
        *,
        max_tokens: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: tensorplay.dtype,
        device: tensorplay.device | str,
        max_top_k: int = 1,
    ) -> GroupedMoEWorkspace:
        if min(
            max_tokens,
            num_experts,
            hidden_size,
            intermediate_size,
            max_top_k,
        ) <= 0:
            raise ValueError("MoE workspace dimensions must be positive")

        # The padded scatter is faster for a few experts and small batches, but
        # its E*M storage would explode for many-expert models. Switch to the
        # compact route early enough to cap padded capacity at about 2K rows.
        (
            compact_min_tokens,
            padded_capacity,
            expert_capacity,
        ) = cls._capacities(
            max_tokens,
            num_experts,
            max_top_k,
        )
        target_device = tensorplay.device(device)

        return cls(
            max_tokens=max_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_top_k=max_top_k,
            compact_min_tokens=compact_min_tokens,
            expert_capacity=expert_capacity,
            expert_counts=tensorplay.empty(
                num_experts,
                dtype=tensorplay.int32,
                device=target_device,
            ),
            expert_cursors=tensorplay.empty(
                num_experts,
                dtype=tensorplay.int32,
                device=target_device,
            ),
            topk_ids=tensorplay.empty(
                (max_tokens, 1),
                dtype=tensorplay.int64,
                device=target_device,
            ),
            topk_weights=tensorplay.empty(
                (max_tokens, 1),
                dtype=tensorplay.float32,
                device=target_device,
            ),
            sorted_hidden=tensorplay.empty(
                (padded_capacity, hidden_size),
                dtype=dtype,
                device=target_device,
            ),
            sorted_indices=tensorplay.empty(
                expert_capacity,
                dtype=tensorplay.int64,
                device=target_device,
            ),
            sorted_weights=tensorplay.empty(
                expert_capacity,
                dtype=tensorplay.float32,
                device=target_device,
            ),
            intermediate=tensorplay.empty(
                (expert_capacity, intermediate_size),
                dtype=dtype,
                device=target_device,
            ),
            output=tensorplay.empty(
                (max_tokens, hidden_size),
                dtype=dtype,
                device=target_device,
            ),
            route_output=(
                tensorplay.empty(
                    (max_tokens * max_top_k, hidden_size),
                    dtype=tensorplay.float32,
                    device=target_device,
                )
                if max_top_k > 1
                else None
            ),
        )

    @property
    def nbytes(self) -> int:
        tensors = (
            self.expert_counts,
            self.expert_cursors,
            self.topk_ids,
            self.topk_weights,
            self.sorted_hidden,
            self.sorted_indices,
            self.sorted_weights,
            self.intermediate,
            self.output,
            self.route_output,
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None
        )


def rms_norm(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
    *,
    policy: KernelPolicy = DEFAULT_POLICY,
) -> tensorplay.Tensor:
    if policy.triton_enabled_for(x):
        return triton_rms_norm(x, weight, eps)
    return fallback.rms_norm(x, weight, eps)


def fused_add_rms_norm(
    x: tensorplay.Tensor,
    residual: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
    *,
    policy: KernelPolicy = DEFAULT_POLICY,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    if policy.triton_enabled_for(x):
        return triton_fused_add_rms_norm(
            x,
            residual,
            weight,
            eps,
        )
    return fallback.fused_add_rms_norm(x, residual, weight, eps)


def silu_and_mul(
    gate_up: tensorplay.Tensor,
    *,
    policy: KernelPolicy = DEFAULT_POLICY,
) -> tensorplay.Tensor:
    if policy.triton_enabled_for(gate_up):
        return triton_silu_mul(gate_up)
    return fallback.silu_and_mul(gate_up)


def rope_and_cache_write(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    slot_mapping: tensorplay.Tensor,
    cos_cache: tensorplay.Tensor,
    sin_cache: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    active: tensorplay.Tensor | None = None,
    *,
    policy: KernelPolicy = DEFAULT_POLICY,
) -> tensorplay.Tensor:
    if active is None:
        active = tensorplay.ones(
            query.shape[0],
            dtype=tensorplay.bool,
            device=query.device,
        )
    if policy.triton_enabled_for(query):
        return triton_rope_cache(
            query,
            key,
            value,
            positions,
            slot_mapping,
            cos_cache,
            sin_cache,
            key_cache,
            value_cache,
            active,
        )
    return fallback.rope_and_cache_write(
        query,
        key,
        value,
        positions,
        slot_mapping,
        cos_cache,
        sin_cache,
        key_cache,
        value_cache,
        active,
    )


def fused_qkv_norm_rope_cache_write(
    qkv: tensorplay.Tensor,
    q_norm_weight: tensorplay.Tensor,
    k_norm_weight: tensorplay.Tensor,
    eps: float,
    positions: tensorplay.Tensor,
    slot_mapping: tensorplay.Tensor,
    cos_cache: tensorplay.Tensor,
    sin_cache: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    active: tensorplay.Tensor,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    policy: KernelPolicy = DEFAULT_POLICY,
) -> tensorplay.Tensor:
    if policy.triton_enabled_for(qkv):
        return triton_fused_qkv_rope_cache(
            qkv,
            q_norm_weight,
            k_norm_weight,
            eps,
            positions,
            slot_mapping,
            cos_cache,
            sin_cache,
            key_cache,
            value_cache,
            active,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
        )
    head_dim = q_norm_weight.numel()
    query_width = num_query_heads * head_dim
    kv_width = num_kv_heads * head_dim
    query, key, value = qkv.split(
        (query_width, kv_width, kv_width),
        dim=-1,
    )
    query = fallback.rms_norm(
        query.view(-1, num_query_heads, head_dim),
        q_norm_weight,
        eps,
    )
    key = fallback.rms_norm(
        key.view(-1, num_kv_heads, head_dim),
        k_norm_weight,
        eps,
    )
    return fallback.rope_and_cache_write(
        query,
        key,
        value.view(-1, num_kv_heads, head_dim),
        positions,
        slot_mapping,
        cos_cache,
        sin_cache,
        key_cache,
        value_cache,
        active,
    )


def paged_decode_attention(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    sequence_lengths: tensorplay.Tensor,
    page_size: int,
    *,
    policy: KernelPolicy = DEFAULT_POLICY,
) -> tensorplay.Tensor:
    if policy.triton_enabled_for(query):
        return triton_paged_decode(
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            page_size,
        )
    return fallback.paged_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        page_size,
    )


def paged_prefill_attention(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    query_start_loc: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    *,
    query_lengths: tuple[int, ...],
    context_lengths: tuple[int, ...],
    page_size: int,
    policy: KernelPolicy = DEFAULT_POLICY,
) -> tensorplay.Tensor:
    if not policy.triton_enabled_for(query):
        raise ValueError(
            "the packed paged prefill path requires the planned Triton backend"
        )
    return triton_paged_prefill(
        query,
        key_cache,
        value_cache,
        block_tables,
        query_start_loc,
        positions,
        query_lengths=query_lengths,
        context_lengths=context_lengths,
        page_size=page_size,
    )


def routed_moe_decode(
    hidden_states: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor,
    topk_weights: tensorplay.Tensor,
    *,
    policy: KernelPolicy = DEFAULT_POLICY,
) -> tensorplay.Tensor:
    if policy.triton_enabled_for(hidden_states):
        return triton_routed_moe_decode(
            hidden_states,
            gate_up_weight,
            down_weight,
            topk_ids,
            topk_weights,
        )
    return fallback.routed_moe_decode(
        hidden_states,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_weights,
    )


def routed_moe_tensorcore(
    hidden_states: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor,
    topk_weights: tensorplay.Tensor,
) -> tensorplay.Tensor:
    """Graph-safe all-expert Tensor Core path for medium/large decode batches.

    Qwen3-MoE in the v1 target domain has few experts. Computing their
    projections as one dense GEMM plus one strided batched GEMM is faster than
    token-wise GEMV once the decode batch reaches the immutable shape guard.
    The final gather preserves sparse routing semantics.
    """

    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden]")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("routed expert metadata has an invalid shape")
    token_count, hidden_size = hidden_states.shape
    num_experts, packed_intermediate, weight_hidden = gate_up_weight.shape
    if weight_hidden != hidden_size or packed_intermediate % 2:
        raise ValueError("gate/up expert weights have an invalid layout")
    intermediate_size = packed_intermediate // 2
    if down_weight.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("down expert weights have an invalid layout")

    gate_up = F.linear(
        hidden_states,
        gate_up_weight.view(num_experts * packed_intermediate, hidden_size),
    ).view(token_count, num_experts, packed_intermediate)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    expert_outputs = tensorplay.bmm(
        activated.transpose(0, 1),
        down_weight.transpose(1, 2),
    ).transpose(0, 1)
    selected = expert_outputs.gather(
        1,
        topk_ids[..., None].expand(-1, -1, hidden_size),
    )
    return (selected * topk_weights[..., None].to(selected.dtype)).sum(dim=1)


def routed_moe_grouped_tensorcore(
    hidden_states: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor,
    topk_weights: tensorplay.Tensor,
    *,
    policy: KernelPolicy = DEFAULT_POLICY,
    workspace: GroupedMoEWorkspace | None = None,
) -> tensorplay.Tensor:
    if not policy.triton_enabled_for(hidden_states):
        return routed_moe_tensorcore(
            hidden_states,
            gate_up_weight,
            down_weight,
            topk_ids,
            topk_weights,
        )
    return triton_routed_moe_grouped(
        hidden_states,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_weights,
        workspace=workspace,
    )


def routed_moe_grouped_top1_from_hidden(
    hidden_states: tensorplay.Tensor,
    router_weight: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    *,
    normalize: bool,
    policy: KernelPolicy = DEFAULT_POLICY,
    workspace: GroupedMoEWorkspace | None = None,
) -> tensorplay.Tensor:
    if not policy.triton_enabled_for(hidden_states):
        topk_weights, topk_ids = moe_top1(
            F.linear(hidden_states, router_weight),
            normalize=normalize,
            policy=policy,
        )
        return routed_moe_grouped_tensorcore(
            hidden_states,
            gate_up_weight,
            down_weight,
            topk_ids,
            topk_weights,
            policy=policy,
            workspace=workspace,
        )
    return triton_routed_moe_grouped(
        hidden_states,
        gate_up_weight,
        down_weight,
        None,
        None,
        workspace=workspace,
        router_weight=router_weight,
        normalize_router=normalize,
    )


def moe_top1(
    router_logits: tensorplay.Tensor,
    *,
    normalize: bool,
    policy: KernelPolicy = DEFAULT_POLICY,
    output: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    if policy.triton_enabled_for(router_logits):
        return triton_moe_top1(
            router_logits,
            normalize=normalize,
            output=output,
        )
    if output is not None:
        raise ValueError("preallocated top-1 output requires the Triton backend")
    routing_weights = tensorplay.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = tensorplay.topk(routing_weights, 1, dim=-1)
    if normalize:
        topk_weights = tensorplay.ones_like(topk_weights)
    return topk_weights, topk_ids


def moe_topk_softmax(
    router_logits: tensorplay.Tensor,
    top_k: int,
    *,
    normalize: bool,
    gating: str = "softmax",
    policy: KernelPolicy = DEFAULT_POLICY,
    output: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    if policy.triton_enabled_for(router_logits):
        return triton_moe_topk_softmax(
            router_logits,
            top_k,
            normalize=normalize,
            gating=gating,
            output=output,
        )
    if output is not None:
        raise ValueError("preallocated top-k output requires the Triton backend")
    return fallback.moe_topk_softmax(
        router_logits,
        top_k,
        normalize=normalize,
        gating=gating,
    )


def moe_top1_from_hidden(
    hidden_states: tensorplay.Tensor,
    gate_weight: tensorplay.Tensor,
    *,
    normalize: bool,
    policy: KernelPolicy = DEFAULT_POLICY,
    output: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    if not policy.triton_enabled_for(hidden_states):
        return moe_top1(
            F.linear(hidden_states, gate_weight),
            normalize=normalize,
            policy=policy,
            output=output,
        )
    return triton_moe_top1_from_hidden(
        hidden_states,
        gate_weight,
        normalize=normalize,
        output=output,
    )


def greedy_argmax(
    logits: tensorplay.Tensor,
    *,
    policy: KernelPolicy = DEFAULT_POLICY,
) -> tensorplay.Tensor:
    if policy.triton_enabled_for(logits):
        return triton_greedy_argmax(logits)
    return fallback.greedy_argmax(logits)
