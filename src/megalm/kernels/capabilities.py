"""Per-backend capability and shape guards.

``devdoc/GAP-PLAN.md`` §7 rejects a global backend priority table: the exact
backend id and its shape guard are fixed during DISCOVER/DRAFT and sealed in
``ExecutionPlan.kernels``. This module is the *cold-path* registry that defines
each backend's guard set and declared capabilities. The hot path never queries
it — layers call bound callables chosen at startup.

Capabilities are explicit. A backend that does not declare ``mla``, ``sparse``,
``sink``, ``non_causal`` or ``dcp`` cannot be selected for a request needing
them, so an unsupported shape is rejected before serving instead of failing in
a kernel at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from megalm.contracts.plan import ShapeGuard

CAUSAL = "causal"
NON_CAUSAL = "non_causal"
GQA = "gqa"
MLA = "mla"
SPARSE = "sparse"
SINK = "sink"
DCP = "dcp"

_ALL_CAPABILITIES = frozenset(
    {CAUSAL, NON_CAUSAL, GQA, MLA, SPARSE, SINK, DCP}
)


class CapabilityError(ValueError):
    """Raised when no backend satisfies a requested capability/guard set."""


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """One backend's declared guard set for an operator/phase pair."""

    operator: str
    phase: str
    backend: str
    guard: ShapeGuard
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        if not self.operator or not self.backend or not self.phase:
            raise CapabilityError("backend capability identity is incomplete")
        unknown = self.capabilities - _ALL_CAPABILITIES
        if unknown:
            raise CapabilityError(
                f"unknown backend capabilities: {sorted(unknown)}"
            )

    def supports(self, *, required: Iterable[str] = ()) -> bool:
        return set(required) <= self.capabilities

    def accepts(self, batch_size: int, tokens: int) -> bool:
        return (
            self.guard.min_batch_size <= batch_size <= self.guard.max_batch_size
            and self.guard.min_tokens <= tokens <= self.guard.max_tokens
        )


class BackendCapabilityMatrix:
    """Immutable registry of backend guard sets (not a priority table)."""

    def __init__(self, entries: Iterable[BackendCapability]) -> None:
        self._entries = tuple(
            sorted(
                entries,
                key=lambda item: (item.operator, item.phase, item.backend),
            )
        )
        seen: set[tuple[str, str, str]] = set()
        for entry in self._entries:
            key = (entry.operator, entry.phase, entry.backend)
            if key in seen:
                raise CapabilityError(f"duplicate backend capability {key}")
            seen.add(key)

    @property
    def entries(self) -> tuple[BackendCapability, ...]:
        return self._entries

    def entry(self, operator: str, phase: str, backend: str) -> BackendCapability:
        for item in self._entries:
            if (
                item.operator == operator
                and item.phase == phase
                and item.backend == backend
            ):
                return item
        raise CapabilityError(
            f"no capability declaration for {operator}/{phase}/{backend}"
        )

    def capabilities_for(
        self, operator: str, phase: str, backend: str
    ) -> tuple[str, ...]:
        return tuple(sorted(self.entry(operator, phase, backend).capabilities))

    def resolve(
        self,
        operator: str,
        phase: str,
        *,
        backend: str | None = None,
        batch_size: int,
        tokens: int,
        required: Iterable[str] = (),
    ) -> BackendCapability:
        """Validate the DISCOVER-picked backend against its guard set.

        The backend id is fixed cold by DISCOVER; this method never ranks
        alternatives. When ``backend`` is omitted the caller must leave exactly
        one guard set matching, otherwise the registry rejects the ambiguity.
        """

        matches = [
            item
            for item in self._entries
            if item.operator == operator
            and item.phase == phase
            and (backend is None or item.backend == backend)
            and item.accepts(batch_size, tokens)
            and item.supports(required=required)
        ]
        if not matches:
            raise CapabilityError(
                f"no backend for {operator}/{phase} with {tuple(required)}"
            )
        if len(matches) > 1:
            raise CapabilityError(
                f"ambiguous backend selection for {operator}/{phase}"
            )
        return matches[0]

    def validate(self, operator: str, phase: str, backend: str) -> None:
        self.entry(operator, phase, backend)


def _guard(min_batch: int, max_batch: int, min_tokens: int, max_tokens: int):
    return ShapeGuard(
        min_batch_size=min_batch,
        max_batch_size=max_batch,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )


# The currently shipped backends. MLA / sparse / sink / non-causal / DCP are
# declared only where an implementation exists; requesting them elsewhere
# fails cold rather than inside a kernel. Additions here are additive and do
# not change any existing plan without an explicit guard/capability change.
DEFAULT_CAPABILITIES: tuple[BackendCapability, ...] = (
    BackendCapability(
        operator="linear_and_embedding",
        phase="prefill",
        backend="tensorplay.cublas.linear",
        guard=_guard(1, 1 << 20, 1, 1 << 22),
        capabilities=frozenset({CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="linear_and_embedding",
        phase="decode",
        backend="tensorplay.cublas.linear",
        guard=_guard(1, 1 << 12, 1, 1 << 12),
        capabilities=frozenset({CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="linear_and_embedding",
        phase="prefill",
        backend="megalm.triton.batch_invariant_linear",
        guard=_guard(1, 1 << 20, 1, 1 << 22),
        capabilities=frozenset({CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="linear_and_embedding",
        phase="decode",
        backend="megalm.triton.batch_invariant_linear",
        guard=_guard(1, 1 << 12, 1, 1 << 12),
        capabilities=frozenset({CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="prefill_attention",
        phase="prefill",
        backend="megalm.triton.varlen_paged_flash_prefill",
        guard=_guard(1, 1 << 16, 1, 1 << 20),
        capabilities=frozenset({CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="prefill_attention",
        phase="prefill",
        backend="tensorplay.sdpa.paged_gather",
        guard=_guard(1, 1 << 16, 1, 1 << 20),
        capabilities=frozenset({CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="paged_decode_attention",
        phase="decode",
        backend="megalm.triton.single_pass_paged_online_softmax",
        guard=_guard(1, 1 << 12, 1, 1 << 12),
        capabilities=frozenset({CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="paged_decode_attention",
        phase="decode",
        backend="megalm.triton.split_kv_online_softmax",
        guard=_guard(1, 1 << 12, 1, 1 << 12),
        capabilities=frozenset({CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="paged_decode_attention",
        phase="decode",
        backend="megalm.tensorplay.paged_decode",
        guard=_guard(1, 1 << 12, 1, 1 << 12),
        capabilities=frozenset({CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="prefill_attention",
        phase="prefill",
        backend="megalm.attention.non_causal_sdpa",
        guard=_guard(1, 1 << 16, 1, 1 << 20),
        capabilities=frozenset({NON_CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="paged_decode_attention",
        phase="decode",
        backend="megalm.attention.non_causal_sdpa",
        guard=_guard(1, 1 << 12, 1, 1 << 12),
        capabilities=frozenset({NON_CAUSAL, GQA}),
    ),
    BackendCapability(
        operator="prefill_attention",
        phase="prefill",
        backend="megalm.attention.sink_sdpa",
        guard=_guard(1, 1 << 16, 1, 1 << 20),
        capabilities=frozenset({CAUSAL, GQA, SINK}),
    ),
    BackendCapability(
        operator="paged_decode_attention",
        phase="decode",
        backend="megalm.attention.sink_sdpa",
        guard=_guard(1, 1 << 12, 1, 1 << 12),
        capabilities=frozenset({CAUSAL, GQA, SINK}),
    ),
    BackendCapability(
        operator="prefill_attention",
        phase="prefill",
        backend="megalm.attention.sparse_index_sdpa",
        guard=_guard(1, 1 << 16, 1, 1 << 20),
        capabilities=frozenset({CAUSAL, GQA, SPARSE}),
    ),
    BackendCapability(
        operator="paged_decode_attention",
        phase="decode",
        backend="megalm.attention.sparse_index_sdpa",
        guard=_guard(1, 1 << 12, 1, 1 << 12),
        capabilities=frozenset({CAUSAL, GQA, SPARSE}),
    ),
    BackendCapability(
        operator="prefill_attention",
        phase="prefill",
        backend="megalm.attention.mla_latent_prefill",
        guard=_guard(1, 1 << 16, 1, 1 << 20),
        capabilities=frozenset({CAUSAL, GQA, MLA}),
    ),
    BackendCapability(
        operator="paged_decode_attention",
        phase="decode",
        backend="megalm.attention.mla_latent_decode",
        guard=_guard(1, 1 << 12, 1, 1 << 12),
        capabilities=frozenset({CAUSAL, GQA, MLA}),
    ),
)

DEFAULT_MATRIX = BackendCapabilityMatrix(DEFAULT_CAPABILITIES)
