import pytest

from megalm.contracts.plan import ShapeGuard
from megalm.kernels.capabilities import (
    CAUSAL,
    GQA,
    MLA,
    BackendCapability,
    BackendCapabilityMatrix,
    CapabilityError,
    DEFAULT_MATRIX,
)


def _entry(backend, capabilities, *, min_tokens=1, max_tokens=1024):
    return BackendCapability(
        operator="prefill_attention",
        phase="prefill",
        backend=backend,
        guard=ShapeGuard(
            min_batch_size=1,
            max_batch_size=16,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        ),
        capabilities=frozenset(capabilities),
    )


def test_capability_identity_validation():
    with pytest.raises(CapabilityError, match="identity"):
        BackendCapability(
            operator="",
            phase="prefill",
            backend="x",
            guard=ShapeGuard(1, 16, 1, 1024),
            capabilities=frozenset(),
        )
    with pytest.raises(CapabilityError, match="unknown"):
        _entry("x", {"telepathy"})


def test_guard_sets_select_exactly_one_backend():
    matrix = BackendCapabilityMatrix(
        (_entry("small", {CAUSAL, GQA}, max_tokens=128),)
    )
    selected = matrix.resolve(
        "prefill_attention",
        "prefill",
        batch_size=1,
        tokens=64,
        required=(CAUSAL, GQA),
    )
    assert selected.backend == "small"
    with pytest.raises(CapabilityError, match="no backend"):
        matrix.resolve(
            "prefill_attention", "prefill", batch_size=1, tokens=256
        )
    with pytest.raises(CapabilityError, match="no backend"):
        matrix.resolve(
            "prefill_attention",
            "prefill",
            batch_size=1,
            tokens=8,
            required=(MLA,),
        )


def test_ambiguous_guard_sets_are_rejected():
    matrix = BackendCapabilityMatrix((_entry("a", {CAUSAL}), _entry("b", {CAUSAL})))
    with pytest.raises(CapabilityError, match="ambiguous"):
        matrix.resolve(
            "prefill_attention", "prefill", batch_size=1, tokens=8
        )


@pytest.mark.parametrize("capability", DEFAULT_MATRIX.entries, ids=lambda c: c.backend)
def test_every_backend_guard_accepts_its_boundaries(capability):
    # Each guard combination is exercised at both boundaries with the backend
    # id fixed by DISCOVER; no runtime ranking is involved.
    guard = capability.guard
    for tokens in (guard.min_tokens, guard.max_tokens):
        selected = DEFAULT_MATRIX.resolve(
            capability.operator,
            capability.phase,
            backend=capability.backend,
            batch_size=guard.min_batch_size,
            tokens=tokens,
            required=tuple(sorted(capability.capabilities)),
        )
        assert selected.backend == capability.backend


def test_sealed_kernels_carry_capabilities_from_the_registry():
    from tests.test_model_and_plan import _tp_plan_kwargs

    from megalm.planning import build_execution_plan

    plan = build_execution_plan(**_tp_plan_kwargs())
    attention = [
        kernel
        for kernel in plan.kernels
        if kernel.operator in {"linear_and_embedding", "prefill_attention"}
    ]
    assert attention
    assert all(kernel.capabilities for kernel in attention)
    assert all("causal" in kernel.capabilities for kernel in attention)
    # Capabilities are part of the sealed digest.
    assert plan.verify_digest() is None


def test_default_matrix_covers_shipped_backends():
    entries = DEFAULT_MATRIX.entries
    assert entries == tuple(sorted(entries, key=lambda e: (e.operator, e.phase, e.backend)))
    linear = DEFAULT_MATRIX.resolve(
        "linear_and_embedding",
        "prefill",
        backend="tensorplay.cublas.linear",
        batch_size=4,
        tokens=128,
    )
    assert {CAUSAL, GQA} <= linear.capabilities
    assert MLA not in linear.capabilities
    assert DEFAULT_MATRIX.capabilities_for(
        "prefill_attention", "prefill", "tensorplay.sdpa.paged_gather"
    ) == (CAUSAL, GQA)
