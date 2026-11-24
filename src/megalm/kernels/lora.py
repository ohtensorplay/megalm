"""Batched multi-adapter LoRA delta (Punica/BGMV style) for LoRA-enabled models.

Each token carries an adapter slot index; ``-1`` means "no adapter" and leaves
the base output untouched.  A and B are fixed-shape banks of shape
``[slots, rank, in]`` and ``[slots, out, rank]``, so the same kernel handles a
batch where different requests use different adapters.  The Triton path is a
two-stage shrink/expand pair; the TensorPlay path is the same math expressed
with gathered batched products and is used on CPU and as a numerical reference.
"""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _lora_shrink_kernel(
    X,
    A,
    Slots,
    Tmp,
    IN: tl.constexpr,
    R: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    slot = tl.load(Slots + token)
    rows = tl.arange(0, BLOCK_R)
    if slot < 0:
        tl.store(
            Tmp + token * R + rows,
            tl.zeros((BLOCK_R,), dtype=tl.float32),
            mask=rows < R,
        )
        return
    accumulator = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for start in range(0, IN, BLOCK_IN):
        cols = start + tl.arange(0, BLOCK_IN)
        x = tl.load(X + token * IN + cols, mask=cols < IN, other=0).to(tl.float32)
        a = tl.load(
            A + slot * R * IN + rows[:, None] * IN + cols[None, :],
            mask=(rows[:, None] < R) & (cols[None, :] < IN),
            other=0,
        ).to(tl.float32)
        accumulator += tl.sum(a * x[None, :], axis=1)
    tl.store(Tmp + token * R + rows, accumulator, mask=rows < R)


@triton.jit
def _lora_expand_kernel(
    Y,
    B,
    Slots,
    Tmp,
    OUT: tl.constexpr,
    R: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SCALE: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    slot = tl.load(Slots + token)
    if slot < 0:
        return
    outputs = tl.program_id(1) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    rows = tl.arange(0, BLOCK_R)
    lora_hidden = tl.load(
        Tmp + token * R + rows,
        mask=rows < R,
        other=0,
    )
    weights = tl.load(
        B + slot * OUT * R + outputs[:, None] * R + rows[None, :],
        mask=(outputs[:, None] < OUT) & (rows[None, :] < R),
        other=0,
    ).to(tl.float32)
    delta = SCALE * tl.sum(weights * lora_hidden[None, :], axis=1)
    base = tl.load(
        Y + token * OUT + outputs,
        mask=outputs < OUT,
        other=0,
    ).to(tl.float32)
    tl.store(
        Y + token * OUT + outputs,
        base + delta,
        mask=outputs < OUT,
    )


def lora_add(
    output: tensorplay.Tensor,
    hidden: tensorplay.Tensor,
    lora_a: tensorplay.Tensor,
    lora_b: tensorplay.Tensor,
    slots: tensorplay.Tensor | None,
    *,
    scale: float = 1.0,
    use_triton: bool = True,
) -> None:
    """Add ``scale * (hidden @ A[slot].T) @ B[slot].T`` into ``output``.

    ``output`` is modified in place.  The Triton path requires contiguous
    arguments; strided tensors transparently use the TensorPlay path.  A
    ``None`` ``slots`` tensor is a cheap no-op for the LoRA-disabled path.
    """

    if slots is None or not slots.numel():
        return
    if hidden.ndim != 2 or output.ndim != 2:
        raise ValueError("LoRA hidden and output must be two-dimensional")
    tokens, in_features = tuple(hidden.shape)
    out_features = tuple(output.shape)[1]
    rank = tuple(lora_a.shape)[1]
    if output.shape[0] != tokens or lora_b.shape[1] != out_features:
        raise ValueError("LoRA batch/output geometry does not match")
    if lora_a.shape[0] != lora_b.shape[0] or lora_a.shape[1] != lora_b.shape[2]:
        raise ValueError("LoRA A/B slot banks have different rank capacity")

    # The Triton kernels address the banks with flat pointer arithmetic and
    # therefore require contiguous storage.  Fall back to the TensorPlay path
    # instead of silently producing wrong deltas on a strided view.
    contiguous = all(
        tensor.is_contiguous()
        for tensor in (hidden, output, lora_a, lora_b, slots)
    )
    if use_triton and contiguous and hidden.device.type == "cuda":
        temporary = tensorplay.empty(
            (tokens, rank),
            dtype=tensorplay.float32,
            device=hidden.device,
        )
        block_in = 64
        block_r = triton.next_power_of_2(rank)
        _lora_shrink_kernel[(tokens,)](
            hidden,
            lora_a,
            slots,
            temporary,
            in_features,
            rank,
            block_in,
            block_r,
            num_warps=4,
        )
        block_out = 64
        _lora_expand_kernel[(tokens, triton.cdiv(out_features, block_out))](
            output,
            lora_b,
            slots,
            temporary,
            out_features,
            rank,
            block_out,
            block_r,
            float(scale),
            num_warps=4,
        )
        return

    # Reference / CPU path: gather one A and B per token, then two batched
    # products.  Keeping it in TensorPlay makes the fallback differentiable
    # in the same way as the model's other reference kernels.
    safe_slots = slots.clamp_min(0)
    a = lora_a.index_select(0, safe_slots)
    b = lora_b.index_select(0, safe_slots)
    shrink = tensorplay.einsum("bi,bri->br", hidden, a)
    delta = tensorplay.einsum("br,bor->bo", shrink, b)
    active = (slots >= 0).unsqueeze(-1)
    output.add_(tensorplay.where(active, delta * scale, 0))


__all__ = ["lora_add"]
