import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import tensorplay

from synthetic_fixtures import (
    INKLING2_CONFIG,
    write_json,
    write_tiny_shard,
)

from megalm.adapters.qwen3 import qwen3_model_spec_from_config
from megalm.adapters.registry import load_model_spec
from megalm.contracts.model import (
    FeedForwardKind,
    ModelContractError,
    ModelSpec,
    StateKind,
)
from megalm.contracts.plan import (
    CollectiveKind,
    CollectiveSite,
    HardwareSpec,
    PlanContractError,
    TopologySpec,
    WorkloadProfile,
)
from megalm.loading import SafetensorsSource
from megalm.kernels.bindings import (
    bind_kernel_plan,
    discover_kernel_plan,
)
from megalm.kernels.binding_types import (
    TENSORPLAY_CUBLAS_LINEAR_BACKEND,
    TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND,
)
from megalm.kernels.ops import GroupedMoEWorkspace, KernelPolicy
from megalm.planning import build_execution_plan
from megalm.startup import build_plan_draft

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MIB = 1024**2


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A tiny Inkling-2 model: real contract config, header-only weights."""

    write_json(tmp_path, "config.json", INKLING2_CONFIG)
    write_tiny_shard(tmp_path)
    return tmp_path


def test_bundled_inkling2_uses_the_native_loop_model_spec(model_dir: Path) -> None:
    spec = load_model_spec(model_dir)

    assert spec.model_type == "inkling2_loop"
    assert spec.num_layers == 24
    assert spec.hidden_size == 2304
    assert spec.loop is not None
    assert spec.loop.core_start == 6
    assert spec.loop.core_end == 18
    assert spec.loop.num_passes == 3
    assert all(
        layer.feed_forward is FeedForwardKind.MOE for layer in spec.layers
    )
    assert spec.parallel.supported_tp_degrees == (1, 2, 4, 8)
    assert spec.state_schema[0].kind is StateKind.PAGED_KV
    assert len(spec.digest) == 64
    assert spec.map_weight_name("model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.weight",
        spec.weight_mapping[0].transform,
    )


def test_qwen3_moe_sparse_step_keeps_dense_layers_explicit() -> None:
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "moe_intermediate_size": 96,
        "max_position_embeddings": 2048,
        "vocab_size": 256,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 2,
        "mlp_only_layers": [3],
    }
    spec = qwen3_model_spec_from_config(config)
    assert tuple(layer.feed_forward for layer in spec.layers) == (
        FeedForwardKind.DENSE,
        FeedForwardKind.MOE,
        FeedForwardKind.DENSE,
        FeedForwardKind.DENSE,
    )


def test_model_spec_rejects_an_unowned_model_family() -> None:
    with pytest.raises(ModelContractError, match="does not support"):
        qwen3_model_spec_from_config({"model_type": "llama"})


def test_execution_plan_is_stable_immutable_and_digest_verified(model_dir: Path) -> None:
    spec = load_model_spec(model_dir)
    source = SafetensorsSource(model_dir)
    hardware = HardwareSpec(
        accelerator="cuda",
        device_name="test-ampere",
        compute_capability=(8, 6),
        total_memory_bytes=24 * 1024**3,
    )
    workload = WorkloadProfile(
        max_batch_size=16,
        max_sequence_length=32768,
        typical_prefill_tokens=1024,
        typical_decode_tokens=256,
    )
    kwargs = {
        "model_spec": spec,
        "hardware": hardware,
        "resident_model_bytes": source.total_bytes,
        "source_dtype": "bfloat16",
        "dtype": "bfloat16",
        "workload": workload,
        "cache_bytes": 64 * 1024**2,
        "graph_bytes": 8 * 1024**2,
        "workspace_bytes": 16 * 1024**2,
        "page_size": 16,
        "max_batched_tokens": 1024,
        "cuda_graph_buckets": (1, 2, 4, 8, 16),
        "kernel_policy": KernelPolicy(),
    }
    first = build_execution_plan(**kwargs)
    second = build_execution_plan(**kwargs)

    assert first == second
    assert first.plan_digest == second.plan_digest
    assert len(first.plan_digest) == 64
    assert first.cache.capacity_bytes == first.memory.elastic_cache_bytes
    assert first.cache.prefix_backend == "megalm.radix.page_aligned.v1"
    assert first.cache.copy_on_write
    assert first.memory.graph_bytes == 8 * 1024**2
    assert first.memory.reserved_bytes > 0
    assert first.cuda_graph_buckets == (1, 2, 4, 8, 16)
    assert all("transformers" not in kernel.backend for kernel in first.kernels)
    assert any(
        kernel.backend == "megalm.triton.split_kv_online_softmax"
        for kernel in first.kernels
    )
    assert any(
        kernel.backend == "megalm.triton.radix_rejection_sampling"
        for kernel in first.kernels
    )
    assert any(
        kernel.operator == "moe_router_topk"
        or kernel.operator == "routed_expert_gemv_small"
        for kernel in first.kernels
    )
    first.verify_digest()
    with pytest.raises(FrozenInstanceError):
        first.dtype = "bfloat16"  # type: ignore[misc]
    with pytest.raises(PlanContractError, match="digest mismatch"):
        first.verify_digest("0" * 64)


def test_execution_plan_rejects_pre_ampere_hardware(model_dir: Path) -> None:
    spec = load_model_spec(model_dir)
    with pytest.raises(PlanContractError, match="below"):
        build_execution_plan(
            model_spec=spec,
            hardware=HardwareSpec(
                accelerator="cuda",
                device_name="old-gpu",
                compute_capability=(7, 5),
                total_memory_bytes=16 * 1024**3,
            ),
            resident_model_bytes=400 * 1024**2,
            source_dtype="bfloat16",
            dtype="bfloat16",
            workload=WorkloadProfile(
                max_batch_size=1,
                max_sequence_length=2048,
                typical_prefill_tokens=128,
                typical_decode_tokens=32,
            ),
            cache_bytes=64 * 1024**2,
            graph_bytes=0,
            workspace_bytes=0,
            page_size=16,
            max_batched_tokens=256,
            cuda_graph_buckets=(),
            kernel_policy=KernelPolicy(),
        )


def test_plan_draft_binds_before_load_and_seals_without_kernel_drift(
    model_dir: Path,
) -> None:
    spec = load_model_spec(model_dir)
    source = SafetensorsSource(model_dir)
    hardware = HardwareSpec(
        accelerator="cuda",
        device_name="portable-test",
        compute_capability=(8, 0),
        total_memory_bytes=24 * 1024**3,
    )
    workload = WorkloadProfile(
        max_batch_size=4,
        max_sequence_length=2048,
        typical_prefill_tokens=256,
        typical_decode_tokens=128,
    )
    policy = KernelPolicy(use_triton=False)
    binding_spec = discover_kernel_plan(
        policy,
        device="cpu",
        dtype=tensorplay.bfloat16,
    )
    bindings = bind_kernel_plan(
        policy,
        device="cpu",
        dtype=tensorplay.bfloat16,
        expected=binding_spec,
    )
    draft = build_plan_draft(
        model_spec=spec,
        hardware=hardware,
        source_dtype="bfloat16",
        dtype="bfloat16",
        workload=workload,
        estimated_model_bytes=source.total_bytes,
        page_size=16,
        max_batched_tokens=256,
        cuda_graph_buckets=(),
        binding_spec=binding_spec,
        enable_prefix_cache=True,
    )
    sealed = build_execution_plan(
        model_spec=spec,
        hardware=hardware,
        resident_model_bytes=source.total_bytes,
        source_dtype="bfloat16",
        dtype="bfloat16",
        workload=workload,
        cache_bytes=64 * 1024**2,
        graph_bytes=0,
        workspace_bytes=draft.estimated_workspace_bytes,
        page_size=16,
        max_batched_tokens=256,
        cuda_graph_buckets=(),
        kernel_policy=policy,
        linear_backend=bindings.linear_backend,
        prefill_backend=bindings.prefill_backend,
        kernel_artifacts=bindings.artifacts,
        kernel_binding_digest=bindings.digest,
    )

    draft.verify_sealed(sealed)
    assert sealed.schema_version == 3
    assert sealed.kernel_binding_digest == bindings.digest
    assert all(len(kernel.artifact_digest) == 64 for kernel in sealed.kernels)
    assert all(kernel.abi_version == 1 for kernel in sealed.kernels)
    assert all(
        kernel.weight_layout == "qwen3.native-packed.v1"
        for kernel in sealed.kernels
    )


def test_qwen25_omni_tensorplay_plan_keeps_model_sdpa_prefill() -> None:
    policy = KernelPolicy(use_triton=False)
    binding_spec = discover_kernel_plan(
        policy,
        device="cpu",
        dtype=tensorplay.float16,
        model_type="qwen2_5_omni",
    )
    bindings = bind_kernel_plan(
        policy,
        device="cpu",
        dtype=tensorplay.float16,
        expected=binding_spec,
        model_type="qwen2_5_omni",
    )

    assert binding_spec.prefill_backend == TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND
    assert bindings.prefill_backend == TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND
    assert binding_spec.linear_backend == TENSORPLAY_CUBLAS_LINEAR_BACKEND
    assert bindings.linear_backend == TENSORPLAY_CUBLAS_LINEAR_BACKEND
    assert bindings.core.paged_prefill_attention is None
    assert bindings.artifact_for_backend(
        TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND
    ).variant == "eager-cublas-sdpa"


def test_grouped_moe_workspace_estimate_matches_real_allocation() -> None:
    kwargs = {
        "max_tokens": 1024,
        "num_experts": 8,
        "hidden_size": 768,
        "intermediate_size": 256,
        "dtype": tensorplay.float16,
    }
    expected = GroupedMoEWorkspace.estimate_nbytes(**kwargs)
    workspace = GroupedMoEWorkspace.allocate(**kwargs, device="cpu")

    assert workspace.nbytes == expected


def test_grouped_moe_workspace_sizes_topk_routes() -> None:
    kwargs = {
        "max_tokens": 64,
        "num_experts": 128,
        "hidden_size": 512,
        "intermediate_size": 128,
        "dtype": tensorplay.float16,
        "max_top_k": 4,
    }
    workspace = GroupedMoEWorkspace.allocate(**kwargs, device="cpu")

    assert workspace.max_top_k == 4
    assert workspace.route_output is not None
    assert tuple(workspace.route_output.shape) == (256, 512)
    assert workspace.nbytes == GroupedMoEWorkspace.estimate_nbytes(**kwargs)


def test_topology_allows_node_local_tp_with_cross_node_dp_only() -> None:
    topology = TopologySpec(
        tensor_parallel=8,
        data_parallel=2,
        node_count=2,
        gpus_per_node=8,
    )
    assert topology.world_size == 16

    with pytest.raises(PlanContractError, match="cross-node tensor"):
        TopologySpec(
            tensor_parallel=8,
            data_parallel=1,
            node_count=2,
            gpus_per_node=4,
        )


def _dense_qwen3_spec() -> ModelSpec:
    return qwen3_model_spec_from_config(
        {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 2048,
            "vocab_size": 256,
        }
    )


_TP_HARDWARE = HardwareSpec(
    accelerator="cuda",
    device_name="tp-test",
    compute_capability=(8, 6),
    total_memory_bytes=24 * 1024**3,
)

_TP_WORKLOAD = WorkloadProfile(
    max_batch_size=4,
    max_sequence_length=2048,
    typical_prefill_tokens=256,
    typical_decode_tokens=64,
)


def _tp_plan_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model_spec": _dense_qwen3_spec(),
        "hardware": _TP_HARDWARE,
        "resident_model_bytes": 100 * _MIB + 1,
        "source_dtype": "bfloat16",
        "dtype": "bfloat16",
        "workload": _TP_WORKLOAD,
        "cache_bytes": 64 * _MIB + 1,
        "graph_bytes": 0,
        "workspace_bytes": 16 * _MIB,
        "page_size": 16,
        "max_batched_tokens": 256,
        "cuda_graph_buckets": (),
        "kernel_policy": KernelPolicy(use_triton=False),
    }
    kwargs.update(overrides)
    return kwargs


def test_execution_plan_shards_memory_and_records_collectives_for_tp2() -> None:
    single = build_execution_plan(**_tp_plan_kwargs())
    sharded = build_execution_plan(
        **_tp_plan_kwargs(topology=TopologySpec(tensor_parallel=2, gpus_per_node=2))
    )

    assert single.topology.tensor_parallel == 1
    assert single.collectives == ()
    assert single.weights.tensor_parallel == 1
    assert single.weights.estimated_bytes == 100 * _MIB + 1
    assert "tensor_parallel_sharded.v1" not in single.weights.packing_rules

    assert sharded.topology.tensor_parallel == 2
    assert sharded.weights.tensor_parallel == 2
    assert sharded.weights.estimated_bytes == (100 * _MIB + 1 + 1) // 2
    assert sharded.cache.capacity_bytes == (64 * _MIB + 1) // 2
    assert sharded.memory.weights_bytes == sharded.weights.estimated_bytes
    assert sharded.memory.elastic_cache_bytes == sharded.cache.capacity_bytes
    # hidden (4 heads * 16 head_dim) * max_batched_tokens * 2 bytes, staged twice.
    collective_bytes = 2 * 64 * 256 * 2
    assert sharded.memory.workspace_bytes == 16 * _MIB + collective_bytes

    assert len(sharded.collectives) == 1
    site = sharded.collectives[0]
    assert site.kind is CollectiveKind.ALL_REDUCE
    assert site.after_operator == "linear_and_embedding"
    assert site.ranks == (0, 1)
    assert "tensor_parallel_sharded.v1" in sharded.weights.packing_rules
    assert sharded.plan_digest != single.plan_digest
    sharded.verify_digest()


def test_tensor_parallel_rejects_moe_and_undivisible_shards(model_dir: Path) -> None:
    bundled = load_model_spec(model_dir)
    with pytest.raises(PlanContractError, match="MoE layers"):
        build_execution_plan(
            **_tp_plan_kwargs(
                model_spec=bundled,
                topology=TopologySpec(tensor_parallel=2, gpus_per_node=2),
            )
        )

    # 4 attention heads do not divide by TP=3.
    with pytest.raises(PlanContractError, match="attention heads"):
        build_execution_plan(
            **_tp_plan_kwargs(topology=TopologySpec(tensor_parallel=3, gpus_per_node=4))
        )

    # 2 KV heads do not divide by TP=4.
    with pytest.raises(PlanContractError, match="KV heads"):
        build_execution_plan(
            **_tp_plan_kwargs(topology=TopologySpec(tensor_parallel=4, gpus_per_node=4))
        )


def test_execution_plan_rejects_collectives_outside_the_world() -> None:
    plan = build_execution_plan(**_tp_plan_kwargs())
    outside = CollectiveSite(
        after_operator="linear_and_embedding",
        kind=CollectiveKind.ALL_REDUCE,
        ranks=(0, 1),
    )
    with pytest.raises(PlanContractError, match="collective ranks"):
        replace(plan, collectives=(outside,))


def test_plan_draft_carries_topology_and_seals_consistently_for_tp2() -> None:
    spec = _dense_qwen3_spec()
    topology = TopologySpec(tensor_parallel=2, gpus_per_node=2)
    binding_spec = discover_kernel_plan(
        KernelPolicy(use_triton=False),
        device=tensorplay.device("cpu"),
        dtype=tensorplay.bfloat16,
        model_type=spec.model_type,
    )
    bindings = bind_kernel_plan(
        KernelPolicy(use_triton=False),
        device=tensorplay.device("cpu"),
        dtype=tensorplay.bfloat16,
        expected=binding_spec,
        model_type=spec.model_type,
    )
    draft = build_plan_draft(
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
        topology=topology,
    )
    assert draft.topology == topology
    # Collective staging is part of the drafted workspace, so the seal check
    # does not double-count it for TP>1.
    # This is a dense spec (no MoE arena), so only collective staging remains.
    assert draft.estimated_workspace_bytes == 2 * 64 * 256 * 2

    seal_kwargs = _tp_plan_kwargs(
        workspace_bytes=0,
        topology=topology,
        linear_backend=bindings.linear_backend,
        prefill_backend=bindings.prefill_backend,
        kernel_artifacts=bindings.artifacts,
        kernel_binding_digest=bindings.digest,
    )
    sealed = build_execution_plan(**seal_kwargs)
    draft.verify_sealed(sealed)
    assert sealed.topology == topology
    assert len(sealed.collectives) == 1

    other = build_execution_plan(
        **dict(seal_kwargs, topology=TopologySpec())
    )
    with pytest.raises(PlanContractError, match="topology changed"):
        draft.verify_sealed(other)
