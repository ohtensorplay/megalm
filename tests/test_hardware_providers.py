"""G5 gold tests: hardware provider probing, declarations and evidence lock."""

from __future__ import annotations

import subprocess
import sys

import pytest

from megalm.contracts.plan import HardwareSpec
from megalm.providers.registry import (
    CPUProvider,
    CUDAProvider,
    ProviderError,
    ProviderEvidence,
    ProviderRegistry,
    probe_hardware,
)


def test_import_and_cpu_probe_do_not_import_tensorplay() -> None:
    """An unselected provider must not be imported or allocate device memory."""

    code = (
        "import sys;"
        "from megalm.providers import CPUProvider, probe_hardware;"
        "assert 'tensorplay' not in sys.modules;"
        "e = probe_hardware(device='cpu');"
        "assert e.provider == 'cpu';"
        "assert 'tensorplay' not in sys.modules;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_cpu_provider_declares_only_portable_backends() -> None:
    cpu = CPUProvider()
    assert cpu.name == "cpu"
    assert cpu.supports(
        dtype="float16", backends=["tensorplay.cublas.linear"]
    )
    assert not cpu.supports(
        dtype="float16",
        backends=["megalm.triton.batch_invariant_linear"],
    )
    with pytest.raises(ProviderError, match="does not declare backends"):
        cpu.validate(
            dtype="float16",
            backends=["megalm.triton.batch_invariant_linear"],
        )
    with pytest.raises(ProviderError, match="does not support dtype"):
        cpu.validate(dtype="int8", backends=["tensorplay.cublas.linear"])


def test_cuda_provider_declares_triton_and_portable_backends() -> None:
    cuda = CUDAProvider()
    assert cuda.accelerator == "cuda"
    assert cuda.supports(
        dtype="bfloat16",
        backends=[
            "tensorplay.cublas.linear",
            "megalm.triton.varlen_paged_flash_prefill",
        ],
    )
    assert not cuda.supports(dtype="float32", backends=[])


def test_evidence_digest_is_stable_and_detects_drift() -> None:
    cpu = CPUProvider()
    first = cpu.probe(total_memory_bytes=8 << 30)
    second = cpu.probe(total_memory_bytes=8 << 30)
    assert first.digest == second.digest
    first.verify(second)

    drifted = cpu.probe(total_memory_bytes=16 << 30)
    assert drifted.digest != first.digest
    with pytest.raises(ProviderError, match="evidence drift"):
        drifted.verify(first)
    assert len(first.digest) == 64


def test_evidence_describe_exposes_small_scalars_only() -> None:
    evidence = CPUProvider().probe()
    description = evidence.describe()
    assert description["provider"] == "cpu"
    assert description["compute_capability"] == (0, 0)
    assert isinstance(description["backends"], tuple)
    assert isinstance(description["digest"], str)


def test_registry_disabled_provider_is_never_constructed() -> None:
    created: list[str] = []

    def make_cpu() -> CPUProvider:
        created.append("cpu")
        return CPUProvider()

    registry = ProviderRegistry({"cpu": make_cpu, "cuda": CUDAProvider}, disabled=("cpu",))
    assert registry.available == ("cuda",)
    with pytest.raises(ProviderError, match="disabled"):
        registry.select(device="cpu")
    assert created == []
    with pytest.raises(ProviderError, match="disabled"):
        registry.select(provider="cpu")
    assert created == []


def test_registry_rejects_unknown_provider_and_accelerator() -> None:
    registry = ProviderRegistry({"cpu": CPUProvider})
    with pytest.raises(ProviderError, match="unknown hardware provider"):
        registry.select(device="cuda")
    with pytest.raises(ProviderError, match="no provider for accelerator"):
        registry.for_accelerator("tpu")


def test_probe_hardware_selects_by_device_without_cuda() -> None:
    evidence = probe_hardware(device="cpu")
    assert evidence.provider == "cpu"
    assert evidence.accelerator == "cpu"


def test_planning_rejects_unsupported_provider_backend() -> None:
    from megalm.contracts.plan import WorkloadProfile
    from megalm.kernels.ops import KernelPolicy
    from megalm.adapters.inkling2 import inkling2_model_spec_from_config
    from megalm.planning import build_execution_plan
    from synthetic_fixtures import INKLING2_CONFIG

    spec = inkling2_model_spec_from_config(INKLING2_CONFIG)
    hardware = HardwareSpec(
        accelerator="cpu",
        device_name="cpu-reference",
        # Keep the compute-capability guard out of the way so this asserts the
        # provider declaration check specifically.
        compute_capability=(8, 0),
        total_memory_bytes=8 << 30,
    )
    with pytest.raises(Exception, match="does not declare backends"):
        build_execution_plan(
            model_spec=spec,
            hardware=hardware,
            resident_model_bytes=1 << 20,
            source_dtype="float16",
            dtype="float16",
            workload=WorkloadProfile(
                max_batch_size=4,
                max_sequence_length=128,
                typical_prefill_tokens=16,
                typical_decode_tokens=1,
            ),
            cache_bytes=1 << 20,
            graph_bytes=0,
            workspace_bytes=1 << 20,
            page_size=4,
            max_batched_tokens=8,
            cuda_graph_buckets=(),
            kernel_policy=KernelPolicy(use_triton=True),
        )


def test_planning_accepts_declared_cuda_backend() -> None:
    from megalm.contracts.plan import WorkloadProfile
    from megalm.kernels.ops import KernelPolicy
    from megalm.adapters.inkling2 import inkling2_model_spec_from_config
    from megalm.planning import build_execution_plan
    from synthetic_fixtures import INKLING2_CONFIG

    spec = inkling2_model_spec_from_config(INKLING2_CONFIG)
    hardware = HardwareSpec(
        accelerator="cuda",
        device_name="test-ampere",
        compute_capability=(8, 6),
        total_memory_bytes=8 << 30,
    )
    plan = build_execution_plan(
        model_spec=spec,
        hardware=hardware,
        resident_model_bytes=1 << 20,
        source_dtype="bfloat16",
        dtype="bfloat16",
        workload=WorkloadProfile(
            max_batch_size=4,
            max_sequence_length=128,
            typical_prefill_tokens=16,
            typical_decode_tokens=1,
        ),
        cache_bytes=1 << 20,
        graph_bytes=0,
        workspace_bytes=1 << 20,
        page_size=4,
        max_batched_tokens=8,
        cuda_graph_buckets=(),
        kernel_policy=KernelPolicy(use_triton=True),
    )
    assert plan.hardware.accelerator == "cuda"
