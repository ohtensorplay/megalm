from __future__ import annotations

import warnings

import pytest
import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.runtime.compile_backend import (
    CompileBackendError,
    CompilePolicy,
    compile_model,
)
from megalm.runtime.engine import EngineConfig, EngineError


class _CudaShaped:
    """Device stand-in whose ``type`` admits the provider's CUDA gate."""

    type = "cuda"


class _Down(nn.Module):
    def __init__(self, weight: tp.Tensor) -> None:
        super().__init__()
        self.weight = weight
        self.bias = None
        self.use_decode_gemv = False


class _MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_up_weight = tp.randn((2 * intermediate, hidden))
        self.down_proj = _Down(tp.randn((hidden, intermediate)))
        self.lora = None
        self.use_decode_gemv = False

    @staticmethod
    def _silu_and_mul(gate_up: tp.Tensor) -> tp.Tensor:
        width = gate_up.shape[-1] // 2
        gate = gate_up[:, :width]
        up = gate_up[:, width:]
        return (gate * gate.sigmoid()) * up

    def forward(
        self,
        hidden_states: tp.Tensor,
        *,
        decode: bool = False,
        lora_slots: tp.Tensor | None = None,
    ) -> tp.Tensor:
        gate_up = F.linear(hidden_states, self.gate_up_weight)
        activated = self._silu_and_mul(gate_up)
        return F.linear(activated, self.down_proj.weight)


class _Model(nn.Module):
    def __init__(self, mlp: _MLP) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(
        self,
        hidden_states: tp.Tensor,
        *,
        decode: bool = False,
        lora_slots: tp.Tensor | None = None,
    ) -> tp.Tensor:
        return self.mlp(
            hidden_states,
            decode=decode,
            lora_slots=lora_slots,
        )


def _close(actual: tp.Tensor, expected: tp.Tensor) -> bool:
    return float((actual - expected).abs().max()) < 1e-6


def _identity_compile(fn):
    return fn


def test_policy_rejects_unknown_backend_and_region() -> None:
    with pytest.raises(CompileBackendError):
        CompilePolicy(backend="torch")
    with pytest.raises(CompileBackendError):
        CompilePolicy(backend="stax", regions=("attention",))
    with pytest.raises(CompileBackendError):
        CompilePolicy(backend="stax", candidate_tokens=(0,))


def test_engine_config_validates_compile_backend() -> None:
    with pytest.raises(EngineError):
        EngineConfig(model_path="model", compile_backend="torch")
    with pytest.raises(EngineError):
        EngineConfig(
            model_path="model",
            compile_backend="stax",
            compile_startup_budget_seconds=-1.0,
        )
    with pytest.raises(EngineError, match="prefill-only"):
        EngineConfig(
            model_path="model",
            compile_backend="stax",
            compile_buckets=(8,),
            cuda_graph_batch_sizes=(1, 16),
        )
    with pytest.raises(EngineError, match="max_num_batched_tokens"):
        EngineConfig(
            model_path="model",
            compile_backend="stax",
            compile_buckets=(64,),
            max_num_batched_tokens=32,
            cuda_graph_batch_sizes=(1,),
        )
    with pytest.raises(EngineError):
        EngineConfig(
            model_path="model",
            compile_backend="stax",
            compile_max_artifact_bytes=-1,
        )


def test_engine_buckets_default_to_the_prefill_chunk() -> None:
    default = EngineConfig(
        model_path="model",
        compile_backend="stax",
        prefill_chunk_size=777,
    )
    assert default.compile_buckets == (777,)
    explicit = EngineConfig(
        model_path="model",
        compile_backend="stax",
        compile_buckets=(2048, 1024, 1024),
        max_num_batched_tokens=4096,
    )
    assert explicit.compile_buckets == (1024, 2048)


def test_accepted_region_replaces_forward_and_keeps_parity() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    x = tp.randn((8, 32))
    expected = model(x)

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )

    assert report.attempted == 1
    assert report.accepted == 1
    assert report.regions[0].accepted
    assert model.mlp._stax_compiled is True
    assert _close(model(x), expected)


def test_parity_mismatch_rejects_region() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    x = tp.randn((8, 32))
    expected = model(x)

    def shifted(fn):
        def wrapped(value):
            return fn(value) * 2.0

        return wrapped

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=shifted,
    )

    assert report.accepted == 0
    assert "parity" in report.regions[0].reason
    assert not hasattr(model.mlp, "_stax_compiled")
    assert _close(model(x), expected)


def test_lowering_fallback_warning_rejects_region() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))

    def fallback(fn):
        warnings.warn(
            "Stax lowering failed; falling back to the graph executor: TypeError",
            stacklevel=2,
        )
        return fn

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=fallback,
    )

    assert report.accepted == 0
    assert "native lowering missing" in report.regions[0].reason


def test_compile_exception_rejects_region() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))

    def boom(fn):
        raise RuntimeError("no artifact")

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=boom,
    )

    assert report.accepted == 0
    assert "no artifact" in report.regions[0].reason


def test_min_speedup_gate_rejects_slower_region() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))

    def slower(fn):
        def wrapped(value):
            for _ in range(4):
                fn(value)
            return fn(value)

        return wrapped

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=1e9,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=slower,
    )

    assert report.accepted == 0
    assert "speedup" in report.regions[0].reason


def test_lora_and_moe_shaped_modules_are_not_candidates() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    model.mlp.lora = object()
    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert report.attempted == 0

    model.mlp.lora = None
    model.mlp.experts = object()
    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert report.attempted == 0


def test_dispatch_keeps_decode_and_small_batches_eager() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    x = tp.randn((8, 32))
    large = tp.randn((16, 32))
    calls = {"count": 0}

    def counting(fn):
        def wrapped(value, *params):
            calls["count"] += 1
            return fn(value, *params)

        return wrapped

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(16,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=counting,
    )
    assert report.accepted == 1
    calls["count"] = 0

    model(x, decode=True)
    assert calls["count"] == 0
    model(x, decode=False)
    assert calls["count"] == 0
    model(large, decode=False)
    assert calls["count"] == 1


class _BaseSignatureMLP(_MLP):
    def forward(
        self,
        hidden_states: tp.Tensor,
        *,
        decode: bool = False,
    ) -> tp.Tensor:
        return super().forward(hidden_states, decode=decode)


def test_base_forward_without_lora_slots_falls_back_cleanly() -> None:
    mlp = _BaseSignatureMLP(hidden=32, intermediate=64)
    model = _Model(mlp)
    x = tp.randn((8, 32))
    expected = mlp(x, decode=True)

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert report.accepted == 1
    assert _close(mlp(x, decode=True), expected)


class _Proj(nn.Module):
    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.weight = tp.randn((out_features, in_features))
        self.bias = None
        self.use_decode_gemv = False

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        return F.linear(x, self.weight, self.bias)


class _Attn(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.qkv_proj = _Proj(hidden, hidden)
        self.o_proj = _Proj(hidden, hidden)

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        return self.o_proj(self.qkv_proj(x))


class _TensorDownMLP(nn.Module):
    """Qwen3.5-style layout: tensors instead of a down_proj module."""

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_up_weight = tp.randn((2 * intermediate, hidden))
        self.down_proj_weight = tp.randn((hidden, intermediate))
        self._silu_and_mul = _MLP._silu_and_mul

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        gate_up = F.linear(x, self.gate_up_weight)
        return F.linear(self._silu_and_mul(gate_up), self.down_proj_weight)


class _BiasedPackedMLP(nn.Module):
    """PackedSwiGLU-style layout: down_weight plus biases."""

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_up_weight = tp.randn((2 * intermediate, hidden))
        self.gate_up_bias = tp.randn((2 * intermediate,))
        self.down_weight = tp.randn((hidden, intermediate))
        self.down_bias = tp.randn((hidden,))
        self._silu_and_mul = _MLP._silu_and_mul

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        gate_up = F.linear(x, self.gate_up_weight, self.gate_up_bias)
        return F.linear(
            self._silu_and_mul(gate_up),
            self.down_weight,
            self.down_bias,
        )


def test_tensor_down_layout_is_covered() -> None:
    mlp = _TensorDownMLP(hidden=32, intermediate=64)
    model = _Model(mlp)
    x = tp.randn((8, 32))
    expected = mlp(x)
    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            regions=("dense_mlp",),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert report.accepted == 1
    assert _close(mlp(x), expected)


def test_bias_variants_do_not_share_an_artifact() -> None:
    plain = _MLP(hidden=32, intermediate=64)
    biased = _BiasedPackedMLP(hidden=32, intermediate=64)
    model = _Model(plain)
    model.biased = biased
    probe = tp.randn((8, 32))
    expected = biased(probe)
    calls = {"count": 0}

    def counting(fn):
        calls["count"] += 1
        return fn

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=counting,
    )

    assert report.accepted == 2
    assert calls["count"] == 2
    assert _close(plain(probe), plain._eager_forward(probe, decode=False))
    assert _close(biased(probe), expected)


def test_biased_packed_layout_and_attention_projections() -> None:
    mlp = _BiasedPackedMLP(hidden=32, intermediate=64)
    model = _Model(mlp)
    model.attn = _Attn(hidden=32)
    x = tp.randn((8, 32))
    mlp_expected = mlp(x)
    qkv_expected = model.attn.qkv_proj(x)
    o_weight = model.attn.o_proj.weight
    qkv_out = model.attn.qkv_proj(x)
    o_expected = F.linear(qkv_out, o_weight)

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )

    assert report.accepted == 3
    assert _close(mlp(x), mlp_expected)
    assert _close(model.attn.qkv_proj(x), qkv_expected)
    assert _close(model.attn.o_proj(qkv_out), o_expected)


def test_decode_gemv_flag_does_not_block_prefill_coverage() -> None:
    mlp = _MLP(hidden=32, intermediate=64)
    mlp.use_decode_gemv = True
    model = _Model(mlp)
    model.attn = _Attn(hidden=32)
    model.attn.qkv_proj.use_decode_gemv = True
    x = tp.randn((8, 32))
    expected = mlp(x)

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )

    assert report.accepted == 3
    assert mlp._stax_compiled is True
    assert model.attn.qkv_proj._stax_compiled is True
    assert _close(mlp(x), expected)


def test_quantized_and_custom_linear_layouts_are_skipped() -> None:
    quantized = _MLP(hidden=32, intermediate=64)
    quantized.gate_up_int4 = object()
    model = _Model(quantized)
    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            regions=("dense_mlp",),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert report.attempted == 0

    custom = _MLP(hidden=32, intermediate=64)
    custom._linear = object()
    model = _Model(custom)
    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            regions=("dense_mlp",),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert report.attempted == 0


def test_region_selection_can_limit_attention_only() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    model.attn = _Attn(hidden=32)
    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            regions=("attn_proj",),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert report.attempted == 2
    assert report.accepted == 2
    assert not hasattr(model.mlp, "_stax_compiled")
    assert model.attn.qkv_proj._stax_compiled is True


def test_vision_region_is_opt_in_and_does_not_double_compile() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    model.tower = nn.Sequential(_BiasedPackedMLP(hidden=32, intermediate=64))
    tower_input = tp.randn((8, 32))
    tower_expected = model.tower(tower_input)

    default_report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            regions=("dense_mlp",),
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert default_report.attempted == 1
    assert not hasattr(model.tower[0], "_stax_compiled")

    vision_report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            regions=("vision_mlp",),
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert vision_report.attempted == 1
    assert vision_report.accepted == 1
    assert model.tower[0]._stax_compiled is True
    assert _close(model.tower(tower_input), tower_expected)

    combined_report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            regions=("dense_mlp", "vision_mlp"),
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert combined_report.attempted == 2
    assert {item.region for item in combined_report.regions} == {"mlp", "tower.0"}


def test_compiled_region_reads_live_weights_instead_of_baked_constants() -> None:
    mlp = _MLP(hidden=32, intermediate=64)
    model = _Model(mlp)
    captured: list[tuple[tp.Tensor, ...]] = []

    def capturing(fn):
        def wrapped(value, *params):
            captured.append(params)
            return fn(value, *params)

        return wrapped

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=capturing,
    )
    assert report.accepted == 1
    weight, bias, down_weight, down_bias = captured[-1]
    assert bias is None and down_bias is None
    assert weight.data_ptr() == mlp.gate_up_weight.data_ptr()
    assert down_weight.data_ptr() == mlp.down_proj.weight.data_ptr()
    assert weight.requires_grad and down_weight.requires_grad

    x = tp.randn((8, 32))
    expected = mlp._eager_forward(x, decode=False)
    with tp.no_grad():
        mlp.gate_up_weight.add_(0.25)
        mlp.down_proj.weight.add_(0.25)
    mutated = mlp._eager_forward(x, decode=False)
    assert not _close(mutated, expected)
    assert _close(mlp(x), mutated)


def test_runtime_failure_falls_back_to_eager_and_reports_once() -> None:
    mlp = _MLP(hidden=32, intermediate=64)
    model = _Model(mlp)

    state = {"fail": False}

    def failing_compile(fn):
        def wrapped(value, *params):
            if state["fail"]:
                raise RuntimeError("kernel died")
            return fn(value, *params)

        return wrapped

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=failing_compile,
    )
    assert report.accepted == 1
    x = tp.randn((8, 32))
    expected = mlp._eager_forward(x, decode=False)
    state["fail"] = True

    assert _close(mlp(x), expected)
    assert report.regions[0].runtime_fallbacks == 1
    assert mlp._stax_degraded is True
    assert _close(mlp(x), expected)
    assert report.regions[0].runtime_fallbacks == 1
    assert report.regions[0].runtime_calls == 0
    assert mlp._stax_buckets == ()
    assert report.regions[0].buckets == ()
    assert report.regions[0].dispatch_tokens is None


def test_startup_budget_stops_compilation() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    model.tower = nn.Sequential(_BiasedPackedMLP(hidden=32, intermediate=64))
    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            regions=("dense_mlp", "vision_mlp"),
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            startup_budget_seconds=1e-12,
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert report.attempted == 2
    assert report.accepted == 0
    assert all("budget" in item.reason for item in report.regions)


def test_equal_shapes_share_one_artifact_and_one_admission() -> None:
    mlp = _MLP(hidden=32, intermediate=64)
    twin = _MLP(hidden=32, intermediate=64)
    model = _Model(mlp)
    model.twin = twin
    x = tp.randn((8, 32))
    calls = {"count": 0}

    def counting(fn):
        calls["count"] += 1
        return fn

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=counting,
    )

    assert report.attempted == 2
    assert report.accepted == 2
    assert calls["count"] == 1
    assert mlp._stax_buckets == twin._stax_buckets == (8,)
    assert report.regions[1].artifact_bytes == 0
    assert not _close(mlp(x), twin(x))


def test_buckets_below_the_floor_are_not_compiled() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    calls = {"count": 0}

    def counting(fn):
        calls["count"] += 1
        return fn

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=32,
            candidate_tokens=(8, 16),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=counting,
    )

    assert calls["count"] == 0
    assert report.attempted == 1
    assert report.accepted == 0
    assert "compile floor" in report.regions[0].reason
    assert not hasattr(model.mlp, "_stax_compiled")


def test_report_dict_exposes_memory_and_admitted_buckets() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    data = report.as_dict()
    assert data["backend"] == "stax"
    assert data["memory_bytes"] >= 0
    assert data["artifact_bytes"] >= 0
    assert data["regions"][0]["buckets"] == [8]
    assert data["regions"][0]["runtime_fallbacks"] == 0


def test_multiple_buckets_dispatch_only_matching_shapes() -> None:
    mlp = _MLP(hidden=32, intermediate=64)
    model = _Model(mlp)
    calls = {"count": 0}

    def counting(fn):
        calls["count"] += 1
        return fn

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8, 16),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=counting,
    )

    assert report.accepted == 1
    assert report.regions[0].buckets == (8, 16)
    assert calls["count"] == 2

    for tokens in (8, 16):
        x = tp.randn((tokens, 32))
        assert tuple(mlp(x).shape) == (tokens, 32)
        assert _close(mlp(x), mlp._eager_forward(x, decode=False))

    unlisted = tp.randn((12, 32))
    assert _close(mlp(unlisted), mlp._eager_forward(unlisted, decode=False))
    assert report.regions[0].runtime_calls == 4


def test_three_dimensional_input_and_late_adapter_stay_eager() -> None:
    mlp = _MLP(hidden=32, intermediate=64)
    model = _Model(mlp)
    compiled_calls = {"count": 0}

    def counting_compile(fn):
        def wrapped(value, *params):
            compiled_calls["count"] += 1
            return fn(value, *params)

        return wrapped

    report = compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            verify_capture=False,
            min_tokens=8,
            candidate_tokens=(8,),
            min_speedup=0.0,
        ),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=counting_compile,
    )
    assert report.accepted == 1

    x = tp.randn((8, 32))
    assert _close(mlp(x), mlp._eager_forward(x, decode=False))
    assert report.regions[0].runtime_calls == 1
    assert compiled_calls["count"] > 0
    admitted_calls = compiled_calls["count"]

    batched = tp.randn((8, 2, 32))
    with pytest.raises(RuntimeError):
        mlp(batched)
    assert report.regions[0].runtime_calls == 1
    assert compiled_calls["count"] == admitted_calls

    mlp.lora = object()
    assert _close(mlp(x), mlp._eager_forward(x, decode=False))
    assert report.regions[0].runtime_calls == 1


def test_second_pass_reports_already_compiled_without_reinstalling() -> None:
    mlp = _MLP(hidden=32, intermediate=64)
    model = _Model(mlp)
    policy = CompilePolicy(
        backend="stax",
        verify_capture=False,
        min_tokens=8,
        candidate_tokens=(8,),
        min_speedup=0.0,
    )
    first = compile_model(
        model,
        policy=policy,
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert first.accepted == 1
    eager = mlp._eager_forward

    second = compile_model(
        model,
        policy=policy,
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )

    assert second.attempted == 1
    assert second.accepted == 0
    assert "already compiled" in second.regions[0].reason
    assert mlp._eager_forward is eager
    x = tp.randn((8, 32))
    assert _close(mlp(x), eager(x, decode=False))


def test_disabled_policy_is_a_noop() -> None:
    model = _Model(_MLP(hidden=32, intermediate=64))
    report = compile_model(
        model,
        policy=CompilePolicy(backend="none"),
        dtype="float32",
        device=_CudaShaped(),
        compile_fn=_identity_compile,
    )
    assert report.as_dict()["backend"] == "none"
    assert report.attempted == 0
