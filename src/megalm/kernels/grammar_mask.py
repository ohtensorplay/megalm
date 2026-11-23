"""Apply xgrammar token bitmasks to logits in place.

The bitmask layout is xgrammar's: ``int32[batch, ceil(vocab_size / 32)]``, bit
``token_id % 32`` of word ``token_id // 32`` being 1 when the token is allowed.
The kernel writes ``-inf`` to every disallowed logit, so both the greedy argmax
and the stochastic sampler paths respect the grammar without changing the
sampler itself.

``logits_vocab`` (the row stride of the logits tensor) and ``grammar_vocab``
(the vocabulary the grammar was compiled for) are kept separate. Some model
configurations pad the lm_head output beyond the tokenizer vocabulary; those
padding logits are always suppressed because no grammar ever allows them.

A pure-tensorplay fallback handles CPU / non-CUDA devices, where Triton is not
applicable. Both paths leave allowed logits bit-for-bit unchanged.
"""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _apply_bitmask_kernel(
    logits_ptr,
    bitmask_ptr,
    rows_ptr,
    logits_vocab: tl.constexpr,
    grammar_vocab: tl.constexpr,
    words: tl.constexpr,
    block: tl.constexpr,
) -> None:
    mask_row = tl.program_id(0)
    block_index = tl.program_id(1)
    logits_row = tl.load(rows_ptr + mask_row)
    offsets = block_index * block + tl.arange(0, block)
    valid = offsets < logits_vocab
    in_grammar = offsets < grammar_vocab
    word_index = offsets // 32
    bit_index = offsets % 32
    word = tl.load(
        bitmask_ptr + mask_row * words + word_index,
        mask=valid & in_grammar,
        other=0,
    )
    allowed = (((word >> bit_index) & 1) != 0) & in_grammar
    values = tl.load(
        logits_ptr + logits_row * logits_vocab + offsets,
        mask=valid,
        other=-float("inf"),
    )
    tl.store(
        logits_ptr + logits_row * logits_vocab + offsets,
        tl.where(allowed & valid, values, -float("inf")),
        mask=valid,
    )


def apply_grammar_bitmask(
    logits: tensorplay.Tensor,
    bitmask: tensorplay.Tensor,
    rows: tensorplay.Tensor,
    *,
    vocab_size: int | None = None,
    grammar_vocab: int | None = None,
) -> None:
    """Mask ``logits[rows[i]]`` with ``bitmask[i]``, in place.

    ``bitmask`` must be an ``int32`` tensor of shape
    ``(rows.shape[0], ceil(grammar_vocab / 32))`` on the same device as
    ``logits``. ``rows`` is an ``int32``/``int64`` device tensor of logit row
    indices. ``vocab_size`` is the logits row stride; when omitted it is taken
    from ``logits.shape[1]``. ``grammar_vocab`` is the compiled grammar's
    vocabulary; when omitted it equals ``vocab_size``.
    """

    if logits.ndim != 2 or bitmask.ndim != 2 or rows.ndim != 1:
        raise ValueError("grammar bitmask inputs have invalid ranks")
    count = rows.shape[0]
    if bitmask.shape[0] != count:
        raise ValueError("grammar bitmask batch does not match row indices")
    size = vocab_size if vocab_size is not None else logits.shape[1]
    valid_vocab = grammar_vocab if grammar_vocab is not None else size
    if valid_vocab > size:
        raise ValueError("grammar vocabulary cannot exceed the logits vocabulary")
    words = (valid_vocab + 31) // 32
    if bitmask.shape[1] != words:
        raise ValueError("grammar bitmask width does not match grammar vocabulary")
    if count == 0:
        return
    if rows.device.type == "cuda" and logits.device.type == "cuda":
        block = 256
        _apply_bitmask_kernel[(count, triton.cdiv(size, block))](
            logits,
            bitmask,
            rows,
            size,
            valid_vocab,
            words,
            block,
        )
        return

    # CPU fallback: expand the packed mask into a byte mask and use the native
    # masked_fill_ so semantics stay identical to the Triton path.
    import numpy as np

    packed = np.asarray(bitmask.cpu())
    row_ids = [int(value) for value in np.asarray(rows.cpu())]
    positions = np.arange(size, dtype=np.int64)
    effective = np.minimum(positions, valid_vocab - 1)
    # ``True`` marks a logit that must be suppressed. Rows not listed in
    # ``rows`` keep an all-false mask and are left untouched. Padding tokens at
    # or beyond ``valid_vocab`` are always suppressed.
    suppressed = np.zeros((logits.shape[0], size), dtype=bool)
    for slot, target in enumerate(row_ids):
        allowed = (
            (packed[slot, effective // 32] >> (effective % 32)) & 1
        ).astype(bool)
        suppressed[target] = ~allowed
        suppressed[target, valid_vocab:] = True
    mask = tensorplay.tensor(
        suppressed,
        dtype=tensorplay.bool,
        device=logits.device,
    )
    logits.masked_fill_(mask, -float("inf"))


__all__ = ["apply_grammar_bitmask"]
