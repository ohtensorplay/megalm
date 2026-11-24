"""DFlash speculative mode: block drafts + greedy chain verification.

DFlash is a prompt-free speculative mode: rather than a separate draft model,
the target model produces a *block* of lookahead ("bonus") tokens in one pass,
which are then verified with greedy chain acceptance. The GPU kernels already
exist in ``kernels/dflash_query.py`` and ``kernels/speculative_verify.py``; this
module is the CPU-testable, reference-level wrapper that exposes their contract
and the draft source plugged into the scheduler's unified DraftSource ABI.

Wire format and semantics mirror the existing kernel exactly:

* draft / target rows: ``[num_speculative_tokens]`` token ids;
* output row: the accepted prefix, with the first divergence replaced by the
  target token and all later slots set to ``-1``;
* accepted count: number of consecutive matching draft tokens.

The default ``off`` / ``ngram`` / ``suffix`` / ``mtp`` paths never import or call
this module; enabling ``--speculative-mode dflash`` is the only way to activate
it, so there is no hot-path cost when disabled.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from megalm.runtime.speculative import DraftSource, SharedSuffixDraftSource


@dataclass(frozen=True, slots=True)
class ChainVerification:
    output_token_ids: tuple[int, ...]
    accepted_count: int

    @property
    def fully_accepted(self) -> bool:
        return self.accepted_count == len(self.output_token_ids)


def verify_chain(
    draft_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
) -> ChainVerification:
    """Reference greedy chain verification for one lookahead block.

    This is the pure-Python twin of ``kernels/speculative_verify.py``'s
    ``greedy_chain_verify_into``: accept draft tokens while they match the
    target, emit the target token at the first mismatch, and mark the rest as
    ``-1``. When every draft matches, the final target token is emitted so the
    step still advances by one extra token.
    """

    if len(draft_token_ids) != len(target_token_ids):
        raise ValueError("draft and target chains must have equal length")
    width = len(target_token_ids)
    if width == 0:
        return ChainVerification((), 0)
    output = [-1] * width
    accepted = 0
    for index in range(width):
        draft = int(draft_token_ids[index])
        target = int(target_token_ids[index])
        if draft != target:
            output[index] = target
            break
        output[index] = draft
        accepted += 1
    else:
        output[-1] = int(target_token_ids[-1])
    return ChainVerification(tuple(output), accepted)


def verify_chain_batch(
    draft_rows: Sequence[Sequence[int]],
    target_rows: Sequence[Sequence[int]],
) -> tuple[ChainVerification, ...]:
    """Verify a batch of lookahead blocks (one row per sequence)."""

    if len(draft_rows) != len(target_rows):
        raise ValueError("draft and target batches must have equal row counts")
    return tuple(verify_chain(d, t) for d, t in zip(draft_rows, target_rows))


class DFlashDraftSource:
    """Block-granularity draft source for the ``dflash`` mode.

    DFlash dwells on cross-request consensus: a lookahead block is proposed
    only when at least ``min_candidate_agreement`` of observed continuations
    agree on its first two tokens, then emitted at full width. It reuses the
    bounded shared-suffix index (no new cache or checkpoint) and implements the
    same DraftSource ABI as ``ngram`` / ``suffix``, so the scheduler's existing
    prepare / verify / rollback path is unchanged.
    """

    name = "dflash"

    def __init__(
        self,
        *,
        max_ngram_size: int = 8,
        min_candidate_agreement: float = 0.5,
        max_requests: int = 512,
        max_ngrams: int = 8192,
    ) -> None:
        if not 0.0 <= min_candidate_agreement <= 1.0:
            raise ValueError("DFlash agreement must be in [0, 1]")
        self._min_agreement = min_candidate_agreement
        self._inner = SharedSuffixDraftSource(
            max_ngram_size=max_ngram_size,
            max_requests=max_requests,
            max_ngrams=max_ngrams,
        )

    def propose(
        self,
        request_id: str,
        token_ids: Sequence[int],
        *,
        max_draft_tokens: int,
        min_ngram_size: int,
        max_ngram_size: int,
        excluded_drafts: Collection[tuple[int, ...]] = (),
        min_candidate_support: int = 1,
        min_candidate_agreement: float = 0.0,
    ) -> tuple[int, ...]:
        # DFlash imposes its own consensus threshold on top of the caller's;
        # the effective threshold is the stricter of the two.
        effective = max(self._min_agreement, min_candidate_agreement)
        return self._inner.propose(
            request_id,
            token_ids,
            max_draft_tokens=max_draft_tokens,
            min_ngram_size=min_ngram_size,
            max_ngram_size=max_ngram_size,
            excluded_drafts=excluded_drafts,
            min_candidate_support=min_candidate_support,
            min_candidate_agreement=effective,
        )

    def observe(self, request_id: str, token_ids: Sequence[int]) -> None:
        self._inner.observe(request_id, token_ids)

    def forget(self, request_id: str) -> None:
        self._inner.forget(request_id)


__all__ = [
    "ChainVerification",
    "DFlashDraftSource",
    "verify_chain",
    "verify_chain_batch",
]
