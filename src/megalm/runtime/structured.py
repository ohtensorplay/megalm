"""Engine-level constrained decoding backed by xgrammar.

One :class:`GrammarManager` is owned by the Engine. It compiles and caches
compiled grammars (keyed by the canonical structured-output request), and
creates one :class:`RequestGrammar` matcher per request stream.

Design points that matter in production:

* **Optional dependency.** xgrammar is imported lazily. When it is absent,
  ``available`` is false and ``begin()`` returns ``None``; the gateway keeps
  its prompt-injection plus post-validation best-effort path. A missing
  grammar library never fails a user request.
* **Canonical caching.** ``StructuredOutput`` canonicalizes its schema text,
  so equivalent requests reuse one compiled grammar. The cache is bounded and
  LRU, because compiled grammars hold tokenizer-sized tables.
* **Fail-open compilation.** A schema xgrammar cannot compile is logged and
  degrades to best-effort instead of aborting generation.
* **CPU-side matcher.** ``GrammarMatcher`` is advanced on the host between
  engine steps; only the packed bitmask crosses to the device.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from typing import Any

from megalm.contracts.request import SamplingParams, StructuredOutput
from megalm.observability import log_event

_LOGGER = logging.getLogger("megalm.structured")

# Bound the set of distinct reasons we ever log, so a hostile stream of bad
# schemas cannot grow it without limit.
_MAX_WARN_REASONS = 256

try:  # Optional extra: pip install megalm-runtime[structured]
    import xgrammar as _xgrammar
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _xgrammar = None


class StructuredOutputError(RuntimeError):
    """Raised for malformed internal constrained-decoding state."""


@dataclass(slots=True)
class RequestGrammar:
    """Per-request matcher state for one generated stream."""

    matcher: Any
    kind: str
    # Set when xgrammar itself raises. The request degrades to best-effort and
    # is never called into again, so one bad matcher cannot fail the replica.
    failed: bool = False
    error_counted: bool = False

    def fill_bitmask(self, bitmask: Any, index: int = 0) -> bool:
        """Fill the next-token bitmask; return whether it must be applied."""

        if self.failed:
            return False
        try:
            return bool(self.matcher.fill_next_token_bitmask(bitmask, index))
        except Exception as exc:
            self.failed = True
            self._log_error("fill_bitmask", exc)
            return False

    def accept_token(self, token_id: int) -> bool:
        """Advance the grammar with a sampled token.

        Returns ``False`` when the token was rejected by the grammar. A
        rejected token must never reach the bitmask-generated stream; the
        engine logs it and resets the matcher state for that request.
        """

        if self.failed:
            return False
        try:
            return bool(self.matcher.accept_token(int(token_id)))
        except Exception as exc:
            self.failed = True
            self._log_error("accept_token", exc)
            return False

    def is_terminated(self) -> bool:
        if self.failed:
            return False
        try:
            return bool(self.matcher.is_terminated())
        except Exception as exc:  # pragma: no cover - defensive
            self.failed = True
            self._log_error("is_terminated", exc)
            return False

    def reset(self) -> None:
        if self.failed:
            return
        try:
            self.matcher.reset()
        except Exception as exc:  # pragma: no cover - defensive
            self.failed = True
            self._log_error("reset", exc)

    @staticmethod
    def _log_error(stage: str, exc: Exception) -> None:
        log_event(
            _LOGGER,
            logging.WARNING,
            "constrained_grammar_error",
            stage=stage,
            error=f"{type(exc).__name__}: {exc}",
        )


class GrammarManager:
    """Compiles schemas and owns per-request grammar matchers."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        vocab_size: int,
        max_cache_entries: int = 32,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._vocab_size = int(vocab_size)
        self._tokenizer_info: Any = None
        self._compiler: Any = None
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_limit = max(1, int(max_cache_entries))
        self.compile_failures = 0
        self.matcher_errors = 0
        self.degraded_by_reason: Counter[str] = Counter()
        self.last_degrade_reason: str = ""
        self._lock = threading.RLock()
        self._warned_reasons: set[str] = set()
        self._matchers: dict[str, RequestGrammar] = {}
        # Reusable CPU bitmask buffers keyed by row count. xgrammar fills them
        # synchronously on the host, so a buffer can be recycled every step
        # instead of allocating a fresh tokenizer-sized tensor each time.
        self._bitmask_buffers: dict[int, Any] = {}
        if not self._enabled or _xgrammar is None:
            return
        try:
            self._tokenizer_info = self._build_tokenizer_info(tokenizer)
            self._compiler = _xgrammar.GrammarCompiler(
                self._tokenizer_info,
                max_threads=4,
                cache_enabled=True,
            )
        except Exception as exc:  # fail open: grammar is an enhancement
            self._warn(f"cannot initialize xgrammar: {exc}")

    @property
    def available(self) -> bool:
        return self._compiler is not None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _degrade(self, reason: str) -> None:
        """Record why a request fell back to best-effort shaping."""

        self.last_degrade_reason = reason
        self.degraded_by_reason[reason] += 1

    def _warn(self, message: str) -> None:
        # Deduplicate by the exact reason, not globally: a second, different
        # schema failure must still be observable.
        with self._lock:
            if message in self._warned_reasons:
                return
            if len(self._warned_reasons) >= _MAX_WARN_REASONS:
                self._warned_reasons.clear()
            self._warned_reasons.add(message)
        log_event(
            _LOGGER,
            logging.WARNING,
            "constrained_decoding_unavailable",
            reason=message,
        )

    def _build_tokenizer_info(self, tokenizer: Any) -> Any:
        backend = tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else tokenizer
        tokenizer_vocab = backend.get_vocab_size()
        encoded_vocab = [
            backend.id_to_token(index) or "" for index in range(tokenizer_vocab)
        ]
        vocab_dict = {token: index for index, token in enumerate(encoded_vocab)}
        vocab_type = _xgrammar.TokenizerInfo._detect_vocab_type_from_vocab(vocab_dict)
        eos = getattr(tokenizer, "eos_token_id", None)
        return _xgrammar.TokenizerInfo(
            encoded_vocab,
            vocab_type or _xgrammar.VocabType.RAW,
            vocab_size=self._vocab_size,
            stop_token_ids=[] if eos is None else [int(eos)],
        )

    def _compiled(self, request: StructuredOutput) -> Any | None:
        key = f"{request.kind}|{request.schema_json or ''}"
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            try:
                if request.kind == "json_object":
                    compiled = self._compiler.compile_builtin_json_grammar()
                else:
                    compiled = self._compiler.compile_json_schema(request.schema_json)
            except Exception as exc:
                self.compile_failures += 1
                self._degrade("unsupported_schema")
                self._warn(f"cannot compile structured output schema: {exc}")
                return None
            self._cache[key] = compiled
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
            return compiled

    def begin(
        self, request_id: str, request: StructuredOutput | None
    ) -> RequestGrammar | None:
        """Start one request's grammar state, or return ``None`` to degrade."""

        if request is None or not self.available:
            if request is not None:
                self._degrade("unavailable")
            return None
        compiled = self._compiled(request)
        if compiled is None:
            return None
        try:
            matcher = _xgrammar.GrammarMatcher(compiled)
        except Exception as exc:
            self.matcher_errors += 1
            self._degrade("matcher_error")
            self._warn(f"cannot create grammar matcher: {exc}")
            return None
        grammar = RequestGrammar(matcher=matcher, kind=request.kind)
        with self._lock:
            if request_id in self._matchers:
                # Never clobber a live matcher: the existing request must keep
                # its grammar. Fail open for the duplicate submission and make
                # the collision observable.
                log_event(
                    _LOGGER,
                    logging.WARNING,
                    "constrained_request_id_reused",
                    request_id=request_id,
                )
                self._degrade("duplicate_id")
                return None
            self._matchers[request_id] = grammar
        return grammar

    def get(self, request_id: str) -> RequestGrammar | None:
        with self._lock:
            return self._matchers.get(request_id)

    def release(self, request_id: str) -> None:
        with self._lock:
            self._matchers.pop(request_id, None)

    def stats(self) -> dict[str, int]:
        """Snapshot for observability; safe to call from any thread."""

        with self._lock:
            return {
                "compile_failures": self.compile_failures,
                "matcher_errors": self.matcher_errors,
                "degraded_by_reason": dict(sorted(self.degraded_by_reason.items())),
                "active_matchers": len(self._matchers),
                "bitmask_buffers": len(self._bitmask_buffers),
            }

    def allocate_bitmask(self, batch_size: int) -> Any:
        """Allocate a CPU bitmask batch sized for the model vocabulary."""

        if _xgrammar is None:
            raise StructuredOutputError("xgrammar is not available")
        return _xgrammar.allocate_token_bitmask(batch_size, self._vocab_size)

    def bitmask_buffer(self, batch_size: int) -> Any:
        """Return a reusable host bitmask for ``batch_size`` rows.

        ``fill_next_token_bitmask`` overwrites every word of the addressed row,
        so recycling a buffer cannot leak stale bits between steps. Bounding
        the cache to common batch sizes avoids unbounded growth from odd
        ``active`` counts while keeping the decode hot path allocation-free.
        """

        if batch_size <= 0:
            return self.allocate_bitmask(1)
        with self._lock:
            mask = self._bitmask_buffers.get(batch_size)
            if mask is None:
                mask = self.allocate_bitmask(batch_size)
                self._bitmask_buffers[batch_size] = mask
            return mask


def resolve_structured_output(
    sampling: SamplingParams,
    *,
    manager: GrammarManager,
    request_id: str,
    speculative_mode: str,
    logger: Any = _LOGGER,
) -> SamplingParams:
    """Gate a request's constrained output and start its matcher.

    Returns the sampling params to actually use. When constraints cannot be
    honoured (speculative replica, xgrammar missing, unsupported schema), the
    structured field is cleared and the request continues on the gateway's
    best-effort prompt + post-validation path.
    """

    if sampling.structured_output is None:
        return sampling
    if speculative_mode != "off":
        manager._degrade("speculative_mode")
        log_event(
            logger,
            logging.WARNING,
            "constrained_decoding_skipped",
            request_id=request_id,
            reason="speculative_mode",
        )
        return replace(sampling, structured_output=None)
    if manager.begin(request_id, sampling.structured_output) is None:
        return replace(sampling, structured_output=None)
    return sampling


__all__ = [
    "GrammarManager",
    "RequestGrammar",
    "StructuredOutputError",
    "resolve_structured_output",
]
