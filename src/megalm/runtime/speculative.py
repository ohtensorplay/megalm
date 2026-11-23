from __future__ import annotations

from collections import deque
from collections.abc import Collection, Sequence


class PromptLookupIndex:
    """Incremental bounded n-gram index for one append-only token stream."""

    def __init__(
        self,
        *,
        max_ngram_size: int,
        max_positions_per_ngram: int = 16,
    ) -> None:
        if min(max_ngram_size, max_positions_per_ngram) <= 0:
            raise ValueError("prompt lookup index limits must be positive")
        self.max_ngram_size = max_ngram_size
        self.max_positions_per_ngram = max_positions_per_ngram
        self.indexed_tokens = 0
        self.positions: dict[tuple[int, ...], deque[int]] = {}

    def update(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        tokens = tuple(int(token_id) for token_id in token_ids)
        if self.indexed_tokens > len(tokens):
            raise ValueError("prompt lookup token stream is not append-only")
        for end in range(self.indexed_tokens, len(tokens)):
            for ngram_size in range(
                1,
                min(self.max_ngram_size, end + 1) + 1,
            ):
                start = end - ngram_size + 1
                ngram = tokens[start : end + 1]
                starts = self.positions.get(ngram)
                if starts is None:
                    starts = deque(maxlen=self.max_positions_per_ngram)
                    self.positions[ngram] = starts
                starts.append(start)
        self.indexed_tokens = len(tokens)
        return tokens

    def propose(
        self,
        token_ids: Sequence[int],
        *,
        max_draft_tokens: int,
        min_ngram_size: int,
        max_ngram_size: int,
        excluded_drafts: Collection[tuple[int, ...]] = (),
        min_candidate_support: int = 1,
        min_candidate_agreement: float = 0.0,
    ) -> tuple[int, ...]:
        if min(
            max_draft_tokens,
            min_ngram_size,
            max_ngram_size,
            min_candidate_support,
        ) <= 0:
            raise ValueError("prompt lookup limits must be positive")
        if (
            min_ngram_size > max_ngram_size
            or max_ngram_size > self.max_ngram_size
            or not 0.0 <= min_candidate_agreement <= 1.0
        ):
            raise ValueError("prompt lookup n-gram range is invalid")
        tokens = self.update(token_ids)
        for ngram_size in range(
            min(max_ngram_size, len(tokens) - 1),
            min_ngram_size - 1,
            -1,
        ):
            suffix = tokens[-ngram_size:]
            starts = self.positions.get(suffix, ())
            continuations: list[tuple[int, ...]] = []
            self_start = len(tokens) - ngram_size
            for start in reversed(starts):
                if start >= self_start:
                    continue
                continuation_start = start + ngram_size
                continuation = tokens[
                    continuation_start : continuation_start + max_draft_tokens
                ]
                if any(
                    continuation[: len(excluded)] == excluded
                    for excluded in excluded_drafts
                    if excluded
                ):
                    continue
                if len(continuation) >= min(2, max_draft_tokens):
                    continuations.append(continuation)
            supported: list[tuple[int, int, tuple[int, ...]]] = []
            for continuation in continuations:
                agreement_width = (
                    len(continuation)
                    if min_candidate_agreement > 0.0
                    else min(2, len(continuation))
                )
                support = sum(
                    other[:agreement_width]
                    == continuation[:agreement_width]
                    for other in continuations
                )
                agreement = support / len(continuations)
                if (
                    support >= min_candidate_support
                    and agreement >= min_candidate_agreement
                ):
                    supported.append(
                        (support, len(continuation), continuation)
                    )
            if supported:
                return max(supported, key=lambda item: item[:2])[2]
        return ()


def prompt_lookup_draft(
    token_ids: Sequence[int],
    *,
    max_draft_tokens: int,
    min_ngram_size: int,
    max_ngram_size: int,
    excluded_drafts: Collection[tuple[int, ...]] = (),
    min_candidate_support: int = 1,
    min_candidate_agreement: float = 0.0,
) -> tuple[int, ...]:
    """Propose the continuation of the longest repeated suffix n-gram."""

    index = PromptLookupIndex(max_ngram_size=max_ngram_size)
    return index.propose(
        token_ids,
        max_draft_tokens=max_draft_tokens,
        min_ngram_size=min_ngram_size,
        max_ngram_size=max_ngram_size,
        excluded_drafts=excluded_drafts,
        min_candidate_support=min_candidate_support,
        min_candidate_agreement=min_candidate_agreement,
    )


__all__ = ["PromptLookupIndex", "prompt_lookup_draft"]
