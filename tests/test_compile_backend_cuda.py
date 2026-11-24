from __future__ import annotations

import tensorplay

from megalm.kernels.bindings import bind_kernel_plan
from megalm.kernels.ops import KernelPolicy
from megalm.models.qwen3 import DenseMLP, Qwen3Config
from megalm.runtime.compile_backend import CompilePolicy, compile_model

pytestmark = __import__("pytest").mark.skipif(
    not tensorplay.cuda.is_available(),
    reason="the Stax provider needs a CUDA device",
)


class _Layer(tensorplay.nn.Module):
    def __init__(self, mlp: DenseMLP) -> None:
        super().__init__()
        self.mlp = mlp


def _model(tokens: int):
    config = Qwen3Config(
        model_type="qwen3",
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        hidden_act="silu",
        attention_bias=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        decoder_sparse_step=1,
        mlp_only_layers=(),
    )
    bindings = bind_kernel_plan(
        KernelPolicy(use_triton=True),
        device="cuda",
        dtype=tensorplay.bfloat16,
        model_type="qwen3",
    )
    mlp = DenseMLP(
        config,
        device="cuda",
        dtype=tensorplay.bfloat16,
        kernel_bindings=bindings,
        use_decode_gemv=False,
    )
    return _Layer(mlp), mlp, tokens


def _compile(model, mlp, tokens, min_speedup: float):
    return compile_model(
        model,
        policy=CompilePolicy(
            backend="stax",
            candidate_tokens=(tokens,),
            min_tokens=tokens,
            min_speedup=min_speedup,
        ),
        dtype="bfloat16",
        device=mlp.gate_up_weight.device,
    )


def test_admitted_region_matches_eager_and_skips_decode() -> None:
    model, mlp, tokens = _model(128)
    report = _compile(model, mlp, tokens, 0.0)
    assert report.accepted == 1
    assert report.regions[0].buckets == (tokens,)
    assert report.regions[0].reason == ""

    x = tensorplay.randn((tokens, 256), device="cuda", dtype=tensorplay.bfloat16)
    with tensorplay.inference_mode():
        expected = mlp._eager_forward(x, decode=False)
        actual = mlp(x, decode=False)
    tensorplay.cuda.synchronize()
    assert tuple(actual.shape) == tuple(expected.shape)
    error = float(
        (actual.float() - expected.float()).abs().max()
        / max(float(expected.float().abs().max()), 1.0)
    )
    assert error <= 2e-2
    assert report.regions[0].runtime_calls == 1
    assert report.regions[0].runtime_fallbacks == 0

    with tensorplay.inference_mode():
        decoded = mlp(x, decode=True)
        decoded_eager = mlp._eager_forward(x, decode=True)
    assert tuple(decoded.shape) == tuple(decoded_eager.shape)
    assert report.regions[0].runtime_calls == 1
    assert report.regions[0].runtime_fallbacks == 0


def test_default_gate_keeps_the_handwritten_kernel_when_it_wins() -> None:
    model, mlp, tokens = _model(128)
    report = _compile(model, mlp, tokens, 1.05)
    if report.accepted:
        # On a faster GPU Stax may genuinely win; then the region must be
        # installed and still match eager.
        assert mlp._stax_compiled is True
        x = tensorplay.randn((tokens, 256), device="cuda", dtype=tensorplay.bfloat16)
        with tensorplay.inference_mode():
            actual = mlp(x, decode=False)
            expected = mlp._eager_forward(x, decode=False)
        error = float(
            (actual.float() - expected.float()).abs().max()
            / max(float(expected.float().abs().max()), 1.0)
        )
        assert error <= 2e-2
    else:
        assert not hasattr(mlp, "_stax_compiled")
        assert "speedup" in report.regions[0].reason
