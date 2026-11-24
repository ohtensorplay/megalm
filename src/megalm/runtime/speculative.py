from __future__ import annotations

from collections import deque
from collections.abc import Collection, Sequence
from typing import Protocol


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


class DraftSource(Protocol):
    """Unified draft generator plugged into the scheduler's prepare/verify path.

    One instance is owned by the Engine for the whole server lifetime. The
    scheduler calls ``propose`` per sequence on every speculative step and
    ``observe`` whenever tokens are committed, so a source can keep per-request
    or cross-request state without the Engine knowing its internals.
    """

    name: str

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
    ) -> tuple[int, ...]: ...

    def observe(self, request_id: str, token_ids: Sequence[int]) -> None: ...

    def forget(self, request_id: str) -> None: ...


class PromptLookupDraftSource:
    """ABI adapter around the existing per-request ``PromptLookupIndex``.

    This is the ``ngram`` mode. Its ``propose`` delegates directly to the
    original index, so enabling the plugin ABI does not change ngram behavior.
    """

    name = "ngram"

    def __init__(
        self,
        *,
        max_ngram_size: int,
        max_positions_per_ngram: int = 16,
    ) -> None:
        self._max_ngram_size = max_ngram_size
        self._max_positions_per_ngram = max_positions_per_ngram
        self._indexes: dict[str, PromptLookupIndex] = {}

    def _index(self, request_id: str) -> PromptLookupIndex:
        index = self._indexes.get(request_id)
        if index is None:
            index = PromptLookupIndex(
                max_ngram_size=self._max_ngram_size,
                max_positions_per_ngram=self._max_positions_per_ngram,
            )
            self._indexes[request_id] = index
        return index

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
        return self._index(request_id).propose(
            token_ids,
            max_draft_tokens=max_draft_tokens,
            min_ngram_size=min_ngram_size,
            max_ngram_size=max_ngram_size,
            excluded_drafts=excluded_drafts,
            min_candidate_support=min_candidate_support,
            min_candidate_agreement=min_candidate_agreement,
        )

    def observe(self, request_id: str, token_ids: Sequence[int]) -> None:
        self._index(request_id).update(token_ids)

    def forget(self, request_id: str) -> None:
        self._indexes.pop(request_id, None)


class SharedSuffixDraftSource:
    """Cross-request suffix index (``suffix`` mode).

    Unlike ``ngram``, which only matches a pattern against one request's own
    history, this source indexes n-grams from every request in a bounded LRU.
    A pattern seen in an earlier (or concurrent) request can therefore seed the
    draft, which is the defining behavior of suffix decoding. The index is
    append-only per request; a changed prefix triggers a bounded rebuild.
    """

    name = "suffix"

    def __init__(
        self,
        *,
        max_ngram_size: int = 8,
        max_positions_per_ngram: int = 8,
        max_requests: int = 512,
        max_ngrams: int = 8192,
    ) -> None:
        if min(max_ngram_size, max_positions_per_ngram, max_requests, max_ngrams) <= 0:
            raise ValueError("suffix index limits must be positive")
        self._max_ngram_size = max_ngram_size
        self._max_positions = max_positions_per_ngram
        self._max_requests = max_requests
        self._max_ngrams = max_ngrams
        self._streams: dict[str, tuple[int, ...]] = {}
        self._stream_order: deque[str] = deque()
        self._positions: dict[tuple[int, ...], deque[tuple[str, int]]] = {}
        self._ngram_order: deque[tuple[int, ...]] = deque()

    def _reset_index(self) -> None:
        self._streams.clear()
        self._stream_order.clear()
        self._positions.clear()
        self._ngram_order.clear()

    def _remember(self, gram: tuple[int, ...]) -> deque[tuple[str, int]]:
        positions = self._positions.get(gram)
        if positions is None:
            positions = deque(maxlen=self._max_positions)
            self._positions[gram] = positions
            self._ngram_order.append(gram)
            while len(self._ngram_order) > self._max_ngrams:
                self._positions.pop(self._ngram_order.popleft(), None)
        return positions

    def observe(self, request_id: str, token_ids: Sequence[int]) -> None:
        tokens = tuple(int(token_id) for token_id in token_ids)
        prior = self._streams.get(request_id)
        if prior is not None and tokens[: len(prior)] != prior:
            # A rewritten prefix invalidates every recorded position; rebuild.
            self._reset_index()
            prior = None
        start = len(prior) if prior is not None else 0
        if prior is None and request_id not in self._streams:
            self._streams[request_id] = ()
            self._stream_order.append(request_id)
            while len(self._stream_order) > self._max_requests:
                self._streams.pop(self._stream_order.popleft(), None)
        self._streams[request_id] = tokens
        for end in range(start, len(tokens)):
            for size in range(1, min(self._max_ngram_size, end + 1) + 1):
                gram = tokens[end - size + 1 : end + 1]
                self._remember(gram).append((request_id, end - size + 1))

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
        self.observe(request_id, token_ids)
        tokens = self._streams.get(request_id, ())
        limit = min(max_ngram_size, self._max_ngram_size, len(tokens) - 1)
        for size in range(limit, min_ngram_size - 1, -1):
            owners = self._positions.get(tokens[-size:], ())
            continuations: list[tuple[int, ...]] = []
            for owner, start in owners:
                candidate = self._streams.get(owner, ())[
                    start + size : start + size + max_draft_tokens
                ]
                if len(candidate) < 2:
                    continue
                if any(
                    candidate[: len(excluded)] == excluded
                    for excluded in excluded_drafts
                ):
                    continue
                continuations.append(candidate)
            if not continuations:
                continue
            width = (
                len(continuations[0])
                if min_candidate_agreement > 0
                else min(2, len(continuations[0]))
            )
            supported: list[tuple[int, int, tuple[int, ...]]] = []
            for candidate in continuations:
                support = sum(
                    1
                    for other in continuations
                    if other[:width] == candidate[:width]
                )
                agreement = support / len(continuations)
                if support >= min_candidate_support and agreement >= min_candidate_agreement:
                    supported.append((support, len(candidate), candidate))
            if supported:
                return max(supported, key=lambda item: item[:2])[2]
        return ()

    def forget(self, request_id: str) -> None:
        self._streams.pop(request_id, None)
        if request_id in self._stream_order:
            self._stream_order.remove(request_id)
        for gram, owners in tuple(self._positions.items()):
            remaining = [owner for owner in owners if owner[0] != request_id]
            if not remaining:
                self._positions.pop(gram, None)
                if gram in self._ngram_order:
                    self._ngram_order.remove(gram)
            else:
                owners.clear()
                owners.extend(remaining)


def build_draft_source(mode: str, *, max_ngram_size: int) -> DraftSource | None:
    """Return the draft source for a prompt-lookup based speculative mode."""

    if mode == "ngram":
        return PromptLookupDraftSource(max_ngram_size=max_ngram_size)
    if mode == "suffix":
        return SharedSuffixDraftSource(max_ngram_size=max_ngram_size)
    if mode == "dflash":
        # Imported lazily so the default off/ngram/suffix paths never load the
        # DFlash module (zero hot-path cost when the mode is not selected).
        from megalm.runtime.dflash import DFlashDraftSource

        return DFlashDraftSource(max_ngram_size=max_ngram_size)
    return None


__all__ = [
    "DraftSource",
    "PromptLookupDraftSource",
    "PromptLookupIndex",
    "SharedSuffixDraftSource",
    "build_draft_source",
    "prompt_lookup_draft",
]
