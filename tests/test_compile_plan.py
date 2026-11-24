from dataclasses import replace

import pytest
import tensorplay

from megalm.contracts.plan import CompiledPlan, PlanContractError
from megalm.kernels.bindings import discover_kernel_plan
from megalm.kernels.ops import KernelPolicy
from megalm.planning import build_execution_plan
from megalm.runtime.compile_backend import compiled_identity_digest
from megalm.startup import build_plan_draft
from tests.test_model_and_plan import (
    _MIB,
    _TP_HARDWARE,
    _TP_WORKLOAD,
    _dense_qwen3_spec,
    _tp_plan_kwargs,
)


_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64


def _candidate() -> CompiledPlan:
    return CompiledPlan(
        backend="stax",
        regions=("dense_mlp",),
        buckets=(1024,),
        digest=_DIGEST,
        artifact_bytes=0,
        max_artifact_bytes=16 * _MIB,
    )


def _sealed(artifact_bytes: int = 4 * _MIB, *, digest: str = _DIGEST) -> CompiledPlan:
    return CompiledPlan(
        backend="stax",
        regions=("dense_mlp",),
        buckets=(1024,),
        digest=digest,
        artifact_bytes=artifact_bytes,
        max_artifact_bytes=16 * _MIB,
    )


def _draft():
    spec = _dense_qwen3_spec()
    binding_spec = discover_kernel_plan(
        KernelPolicy(use_triton=False),
        device=tensorplay.device("cpu"),
        dtype=tensorplay.bfloat16,
        model_type=spec.model_type,
    )
    return binding_spec, build_plan_draft(
        model_spec=spec,
        hardware=_TP_HARDWARE,
        source_dtype="bfloat16",
        dtype="bfloat16",
        workload=_TP_WORKLOAD,
        estimated_model_bytes=100 * _MIB + 1,
        page_size=16,
        max_batched_tokens=256,
        cuda_graph_buckets=(),
        binding_spec=binding_spec,
        enable_prefix_cache=True,
        compiled=_candidate(),
    )


def _seal(binding_spec, compiled: CompiledPlan, *, workspace_bytes: int = 0):
    # Dense spec: provider workspace is zero, so the sealed workspace is just
    # the admitted artifact bytes added by the engine.
    return build_execution_plan(
        **_tp_plan_kwargs(
            workspace_bytes=workspace_bytes + compiled.artifact_bytes,
            linear_backend=binding_spec.linear_backend,
            prefill_backend=binding_spec.prefill_backend,
            kernel_artifacts=binding_spec.artifacts,
            kernel_binding_digest=binding_spec.digest,
            compiled=compiled,
        )
    )


def test_compiled_identity_digest_is_deterministic_and_policy_bound():
    kwargs = dict(
        backend="stax",
        regions=("dense_mlp",),
        buckets=(1024,),
        dtype="bfloat16",
        device_name="dev",
        model_digest="c" * 64,
        tolerance=None,
        min_speedup=1.05,
    )
    assert compiled_identity_digest(**kwargs) == compiled_identity_digest(**kwargs)
    assert compiled_identity_digest(**kwargs) != compiled_identity_digest(
        **dict(kwargs, min_speedup=1.1)
    )
    assert len(compiled_identity_digest(**kwargs)) == 64


def test_compiled_plan_validates_identity_and_budget():
    assert CompiledPlan.disabled().minimal is True
    with pytest.raises(PlanContractError, match="digest"):
        CompiledPlan(backend="stax", regions=("dense_mlp",), digest="bad")
    with pytest.raises(PlanContractError, match="exceed"):
        CompiledPlan(
            backend="stax",
            regions=("dense_mlp",),
            digest=_DIGEST,
            artifact_bytes=8,
            max_artifact_bytes=4,
        )


def test_plan_digest_and_binding_cover_admitted_artifacts():
    baseline = _tp_plan_kwargs(model_spec=_dense_qwen3_spec())
    plain = build_execution_plan(**baseline)
    with_artifact = build_execution_plan(
        **dict(baseline, compiled=_sealed(), workspace_bytes=16 * _MIB)
    )
    assert plain.plan_digest != with_artifact.plan_digest
    assert plain.kernel_binding_digest != with_artifact.kernel_binding_digest
    with_artifact.verify_digest()


def test_verify_sealed_accepts_admitted_artifact_within_budget():
    binding_spec, draft = _draft()
    assert draft.compiled == _candidate()
    sealed = _seal(binding_spec, _sealed())
    draft.verify_sealed(sealed)


def test_verify_sealed_rejects_artifact_bytes_over_drafted_budget():
    binding_spec, draft = _draft()
    drifted = _seal(
        binding_spec,
        CompiledPlan(
            backend="stax",
            regions=("dense_mlp",),
            buckets=(1024,),
            digest=_DIGEST,
            artifact_bytes=20 * _MIB,
            max_artifact_bytes=32 * _MIB,
        ),
    )
    with pytest.raises(PlanContractError, match="budget"):
        draft.verify_sealed(drifted)


def test_verify_sealed_rejects_artifact_identity_drift():
    binding_spec, draft = _draft()
    drifted = _seal(binding_spec, _sealed(digest=_OTHER_DIGEST))
    with pytest.raises(PlanContractError, match="artifact"):
        draft.verify_sealed(drifted)


def test_verify_sealed_rejects_regions_outside_the_draft():
    binding_spec, draft = _draft()
    drifted = _seal(
        binding_spec,
        replace(_sealed(), regions=("dense_mlp", "attn_proj")),
    )
    with pytest.raises(PlanContractError, match="regions"):
        draft.verify_sealed(drifted)
