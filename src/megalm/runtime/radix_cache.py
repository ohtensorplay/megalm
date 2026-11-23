from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence, TypeAlias

from megalm.megalm_rs import PyRadixPrefixCache


class RadixCacheError(RuntimeError):
    """Raised when the compressed prefix tree violates its ownership contract."""


PageKey = tuple[int, ...]
RetainPages = Callable[[tuple[int, ...]], None]
ReleasePages = Callable[[tuple[int, ...]], None]


@dataclass(frozen=True, slots=True)
class RadixKey:
    """Exact, collision-free token pages prepared outside the cache hot path."""

    page_size: int
    token_count: int
    pages: tuple[PageKey, ...]
    namespace: str = ""

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise RadixCacheError("radix key page size must be positive")
        if self.token_count < 0:
            raise RadixCacheError("radix key token count must be non-negative")
        if not isinstance(self.namespace, str) or "\0" in self.namespace:
            raise RadixCacheError("radix key namespace is invalid")
        if len(self.pages) != self.token_count // self.page_size:
            raise RadixCacheError(
                "radix key must contain every complete token page"
            )
        if any(len(page) != self.page_size for page in self.pages):
            raise RadixCacheError("radix key contains a partial token page")

    @classmethod
    def from_tokens(
        cls,
        token_ids: Sequence[int],
        *,
        page_size: int,
        namespace: str = "",
    ) -> RadixKey:
        if page_size <= 0:
            raise RadixCacheError("radix key page size must be positive")
        token_count = len(token_ids)
        page_count = token_count // page_size
        if isinstance(token_ids, tuple):
            pages = tuple(
                token_ids[offset : offset + page_size]
                for offset in range(0, page_count * page_size, page_size)
            )
        else:
            pages = tuple(
                tuple(
                    int(token)
                    for token in token_ids[offset : offset + page_size]
                )
                for offset in range(0, page_count * page_size, page_size)
            )
        return cls(
            page_size=page_size,
            token_count=token_count,
            pages=pages,
            namespace=namespace,
        )

    def prefix(self, token_count: int) -> RadixKey:
        if not 0 <= token_count <= self.token_count:
            raise RadixCacheError("radix key prefix is out of bounds")
        if token_count == self.token_count:
            return self
        return RadixKey(
            page_size=self.page_size,
            token_count=token_count,
            pages=self.pages[: token_count // self.page_size],
            namespace=self.namespace,
        )


class RadixKeyBuilder:
    """Incrementally materialize exact pages for one append-only sequence."""

    __slots__ = ("page_size", "namespace", "_observed_tokens", "_pages")

    def __init__(
        self,
        *,
        page_size: int,
        token_ids: Sequence[int] = (),
        namespace: str = "",
    ) -> None:
        if page_size <= 0:
            raise RadixCacheError("radix key page size must be positive")
        if not isinstance(namespace, str) or "\0" in namespace:
            raise RadixCacheError("radix key namespace is invalid")
        self.page_size = page_size
        self.namespace = namespace
        self._observed_tokens = 0
        self._pages: list[PageKey] = []
        self.sync(token_ids)

    @property
    def token_count(self) -> int:
        return self._observed_tokens

    @property
    def complete_pages(self) -> int:
        return len(self._pages)

    def sync(self, token_ids: Sequence[int]) -> None:
        """Append only newly completed pages from an append-only token list."""

        token_count = len(token_ids)
        if token_count < self._observed_tokens:
            raise RadixCacheError("an append-only radix key cannot shrink")
        target_pages = token_count // self.page_size
        for page_index in range(len(self._pages), target_pages):
            offset = page_index * self.page_size
            self._pages.append(
                tuple(
                    int(token)
                    for token in token_ids[offset : offset + self.page_size]
                )
            )
        self._observed_tokens = token_count

    def snapshot(self, *, token_count: int | None = None) -> RadixKey:
        if token_count is None:
            token_count = self._observed_tokens
        if not 0 <= token_count <= self._observed_tokens:
            raise RadixCacheError(
                "radix key snapshot exceeds the observed token sequence"
            )
        page_count = token_count // self.page_size
        pages = (
            tuple(self._pages)
            if page_count == len(self._pages)
            else tuple(self._pages[:page_count])
        )
        return RadixKey(
            page_size=self.page_size,
            token_count=token_count,
            pages=pages,
            namespace=self.namespace,
        )


RadixKeyInput: TypeAlias = RadixKey | Sequence[int]


@dataclass(frozen=True, slots=True)
class RadixCacheHandle:
    """Pinned logical position in the radix tree."""

    node_id: int
    cached_tokens: int
    page_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RadixMatch:
    handle: RadixCacheHandle | None
    eligible_tokens: int

    @property
    def cached_tokens(self) -> int:
        return 0 if self.handle is None else self.handle.cached_tokens

    @property
    def page_ids(self) -> tuple[int, ...]:
        return () if self.handle is None else self.handle.page_ids


@dataclass(frozen=True, slots=True)
class RadixInsertResult:
    handle: RadixCacheHandle | None
    already_cached_pages: int
    inserted_pages: int


@dataclass(frozen=True, slots=True)
class RadixCacheStats:
    nodes: int
    cached_pages: int
    flattened_page_refs: int
    pinned_pages: int
    evictable_pages: int
    lookups: int
    hits: int
    misses: int
    lookup_pages: int
    hit_pages: int
    insertions: int
    inserted_pages: int
    evictions: int
    evicted_pages: int

    @property
    def request_hit_rate(self) -> float:
        return 0.0 if self.lookups == 0 else self.hits / self.lookups

    @property
    def token_hit_rate(self) -> float:
        return (
            0.0
            if self.lookup_pages == 0
            else self.hit_pages / self.lookup_pages
        )

    def snapshot(self) -> dict[str, int | float]:
        """Return counters together with their computed hit rates."""

        return {
            **asdict(self),
            "request_hit_rate": self.request_hit_rate,
            "token_hit_rate": self.token_hit_rate,
        }


def _with_tuple_pages(handle: tuple[int, int, Sequence[int]]) -> tuple:
    """Adapt the Rust handle payload to the Python page-id tuple contract."""

    node_id, cached_tokens, page_ids = handle
    return node_id, cached_tokens, tuple(page_ids)


class RadixPrefixCache:
    """Leader-owned, page-aligned compressed radix cache.

    The tree lives in the Rust accelerator layer (`megalm_rs`); this facade
    only adapts the Python call contract and error type. Page-ownership
    retain/release callbacks are invoked exactly like the pure-Python
    original, so cache-manager ownership semantics are unchanged.
    """

    def __init__(
        self,
        *,
        page_size: int,
        retain_pages: RetainPages,
        release_pages: ReleasePages,
        trust_page_ownership: bool = False,
    ) -> None:
        if page_size <= 0:
            raise RadixCacheError("radix page size must be positive")
        self.page_size = page_size
        self._retain_pages = retain_pages
        self._release_pages = release_pages
        self._trust_page_ownership = bool(trust_page_ownership)
        self._cache = PyRadixPrefixCache(page_size, bool(trust_page_ownership))

    @staticmethod
    def _translate(exc: ValueError) -> RadixCacheError:
        return RadixCacheError(str(exc))

    @property
    def _roots(self) -> dict[str, int]:
        """Namespace → root-node view kept for introspection parity."""

        return dict(self._cache.root_namespaces())

    def _coerce_key(self, key_or_tokens: RadixKeyInput) -> RadixKey:
        if isinstance(key_or_tokens, RadixKey):
            if key_or_tokens.page_size != self.page_size:
                raise RadixCacheError(
                    "radix key page size does not match the cache"
                )
            return key_or_tokens
        return RadixKey.from_tokens(
            key_or_tokens,
            page_size=self.page_size,
        )

    @property
    def cached_pages(self) -> int:
        return self._cache.stats()[1]

    @property
    def pinned_pages(self) -> int:
        return self._cache.stats()[3]

    @property
    def evictable_pages(self) -> int:
        return self._cache.stats()[4]

    def match_prefix(
        self,
        key_or_tokens: RadixKeyInput,
        *,
        reserve_tokens: int = 1,
        pin: bool = True,
    ) -> RadixMatch:
        """Return the longest full-page prefix while reserving trailing tokens.

        A request normally reserves its final input token because KV alone
        cannot reproduce the logits that follow a completely cached prompt.
        """

        if reserve_tokens < 0:
            raise RadixCacheError("reserve_tokens must be non-negative")
        key = self._coerce_key(key_or_tokens)
        try:
            eligible_tokens, handle = self._cache.match_prefix(
                key.token_count,
                [list(page) for page in key.pages],
                key.namespace,
                reserve_tokens,
                pin,
            )
        except ValueError as exc:
            raise self._translate(exc) from exc
        return RadixMatch(
            handle=None
            if handle is None
            else RadixCacheHandle(*_with_tuple_pages(handle)),
            eligible_tokens=eligible_tokens,
        )

    def insert_prefix(
        self,
        key_or_tokens: RadixKeyInput,
        page_ids: Sequence[int],
        *,
        pin: bool = False,
        replace_handle: RadixCacheHandle | None = None,
    ) -> RadixInsertResult:
        """Insert all complete token pages, retaining only the missing suffix."""

        if replace_handle is not None and not pin:
            raise RadixCacheError("replacing a radix pin requires pin=True")
        key = self._coerce_key(key_or_tokens)
        try:
            handle, already, inserted = self._cache.insert_prefix(
                key.token_count,
                [list(page) for page in key.pages],
                key.namespace,
                list(page_ids),
                pin,
                None
                if replace_handle is None
                else (replace_handle.node_id, replace_handle.cached_tokens),
                retain=self._retain_pages,
                release=self._release_pages,
            )
        except ValueError as exc:
            raise self._translate(exc) from exc
        return RadixInsertResult(
            handle=None if handle is None else RadixCacheHandle(*_with_tuple_pages(handle)),
            already_cached_pages=already,
            inserted_pages=inserted,
        )

    def pin(self, handle: RadixCacheHandle) -> None:
        try:
            self._cache.pin(handle.node_id)
        except ValueError as exc:
            raise self._translate(exc) from exc

    def unpin(self, handle: RadixCacheHandle) -> None:
        try:
            self._cache.unpin(handle.node_id)
        except ValueError as exc:
            raise self._translate(exc) from exc

    def replace_pin(
        self,
        old_handle: RadixCacheHandle | None,
        new_handle: RadixCacheHandle | None,
    ) -> None:
        if old_handle == new_handle:
            return
        if new_handle is not None:
            self.pin(new_handle)
        if old_handle is not None:
            self.unpin(old_handle)

    def evict_pages(self, target_pages: int) -> tuple[int, ...]:
        """Evict cold, unpinned leaves until at least target_pages are released."""

        if target_pages < 0:
            raise RadixCacheError("eviction target must be non-negative")
        try:
            return tuple(self._cache.evict_pages(target_pages, self._release_pages))
        except ValueError as exc:
            raise self._translate(exc) from exc

    def stats(self) -> RadixCacheStats:
        return RadixCacheStats(*self._cache.stats())

    def cached_page_ids(self) -> tuple[int, ...]:
        return tuple(self._cache.cached_page_ids())

    def reset(self) -> None:
        try:
            self._cache.reset(self._release_pages)
        except ValueError as exc:
            raise self._translate(exc) from exc

    def assert_integrity(self) -> None:
        try:
            self._cache.assert_integrity()
        except ValueError as exc:
            raise self._translate(exc) from exc


__all__ = [
    "RadixCacheError",
    "RadixCacheHandle",
    "RadixCacheStats",
    "RadixInsertResult",
    "RadixKey",
    "RadixKeyBuilder",
    "RadixMatch",
    "RadixPrefixCache",
]
