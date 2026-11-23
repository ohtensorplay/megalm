from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _moe_topk_softmax_kernel(
    router_logits_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    SIGMOID: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Fused router: one pass writes descending top-k ids and weights.

    Softmax top-k selection is order-preserving, so ``topk(softmax(x))``
    equals ``topk(x)``.  With ``RENORMALIZE`` the weights are the softmax of
    the selected logits; without it they are the full softmax probabilities
    gathered at the selected experts (Qwen3-30B style ``norm_topk_prob`` off).
    ``SIGMOID`` replaces the softmax score with ``sigmoid(logit)``
    (DeepSeek/Inkling-2 style), optionally normalised over the top-k.
    """
    token_index = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_experts
    logits = tl.load(
        router_logits_ptr + token_index * num_experts + expert_offsets,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)
    if SIGMOID:
        scores = tl.sigmoid(logits)
        scores = tl.where(expert_mask, scores, -float("inf"))
    else:
        scores = logits
    if not SIGMOID and not RENORMALIZE:
        maximum = tl.max(logits, axis=0)
        full_denominator = tl.sum(
            tl.where(expert_mask, tl.exp(logits - maximum), 0.0),
            axis=0,
        )
    remaining = scores
    for rank in range(top_k):
        top_index = tl.argmax(remaining, axis=0)
        top_value = tl.max(remaining, axis=0)
        tl.store(
            topk_ids_ptr + token_index * top_k + rank,
            top_index.to(tl.int64),
        )
        tl.store(
            topk_weights_ptr + token_index * top_k + rank,
            top_value,
        )
        remaining = tl.where(
            expert_offsets == top_index,
            -float("inf"),
            remaining,
        )
    if SIGMOID:
        if RENORMALIZE:
            weight_sum = tl.zeros((), tl.float32)
            for rank in range(top_k):
                weight_sum += tl.load(
                    topk_weights_ptr + token_index * top_k + rank
                )
            for rank in range(top_k):
                weight = tl.load(
                    topk_weights_ptr + token_index * top_k + rank
                )
                tl.store(
                    topk_weights_ptr + token_index * top_k + rank,
                    weight / weight_sum,
                )
    elif RENORMALIZE:
        maximum = tl.load(topk_weights_ptr + token_index * top_k)
        denominator = tl.zeros((), tl.float32)
        for rank in range(top_k):
            value = tl.load(
                topk_weights_ptr + token_index * top_k + rank
            )
            denominator += tl.exp(value - maximum)
        for rank in range(top_k):
            value = tl.load(
                topk_weights_ptr + token_index * top_k + rank
            )
            tl.store(
                topk_weights_ptr + token_index * top_k + rank,
                tl.exp(value - maximum) / denominator,
            )
    else:
        for rank in range(top_k):
            value = tl.load(
                topk_weights_ptr + token_index * top_k + rank
            )
            tl.store(
                topk_weights_ptr + token_index * top_k + rank,
                tl.exp(value - maximum) / full_denominator,
            )


def moe_topk_softmax(
    router_logits: tensorplay.Tensor,
    top_k: int,
    *,
    normalize: bool,
    gating: str = "softmax",
    output: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Top-k routing in one Triton launch.

    Drop-in replacement for the ``softmax(logits) -> topk -> renorm`` chain
    used by Qwen3-MoE / Qwen3.5-MoE / Mixtral decode and prefill.
    """
    if router_logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    if gating not in {"softmax", "sigmoid"}:
        raise ValueError("gating must be 'softmax' or 'sigmoid'")
    router_logits = router_logits.contiguous()
    token_count, num_experts = router_logits.shape
    if not 1 <= top_k <= min(num_experts, 32):
        raise ValueError("fused top-k router requires 1 <= top_k <= 32")
    if output is None:
        topk_weights = tensorplay.empty(
            (token_count, top_k),
            dtype=tensorplay.float32,
            device=router_logits.device,
        )
        topk_ids = tensorplay.empty(
            (token_count, top_k),
            dtype=tensorplay.int64,
            device=router_logits.device,
        )
    else:
        topk_weights, topk_ids = output
        if (
            topk_weights.shape != (token_count, top_k)
            or topk_weights.dtype != tensorplay.float32
            or topk_weights.device != router_logits.device
            or topk_ids.shape != (token_count, top_k)
            or topk_ids.dtype != tensorplay.int64
            or topk_ids.device != router_logits.device
        ):
            raise ValueError("preallocated top-k output has an invalid layout")
    _moe_topk_softmax_kernel[(token_count,)](
        router_logits,
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        RENORMALIZE=bool(normalize),
        SIGMOID=gating == "sigmoid",
        BLOCK_E=triton.next_power_of_2(num_experts),
        num_warps=1,
    )
    return topk_weights, topk_ids


__all__ = ["moe_topk_softmax"]
