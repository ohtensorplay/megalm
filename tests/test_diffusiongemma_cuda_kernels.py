from __future__ import annotations

import pytest
import tensorplay

from megalm.kernels.diffusion_attention import (
    diffusion_attention,
    prepare_diffusion_qkv,
)
from megalm.kernels.diffusion_sample import sample_diffusion_logits
from megalm.kernels.marlin_int4 import _nvcc, pack_marlin_batch
from megalm.kernels.marlin_moe import grouped_marlin_moe, marlin_project, pack_experts
from megalm.models.diffusiongemma.model import _apply_rope, _rms_norm

pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(), reason="requires NVIDIA CUDA"
)


def test_grouped_marlin_repack_and_experts() -> None:
    props = tensorplay.cuda.get_device_properties(0)
    if props.major < 8:
        pytest.skip("grouped Marlin requires Ampere or newer")
    try:
        _nvcc()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    torch = pytest.importorskip("torch")
    tensorplay.manual_seed(31)
    experts, hidden, width, top_k = 8, 256, 704, 4

    def make_projection(n: int, k: int):
        codes = tensorplay.randint(
            0, 256, (experts, n, k // 2), device="cuda", dtype=tensorplay.uint8
        )
        scales = (
            tensorplay.rand((experts, n, k // 32), device="cuda") * 0.01
        ).bfloat16()
        transposed = scales.transpose(1, 2)
        scale_zero = tensorplay.stack(
            (transposed, tensorplay.full_like(transposed, 8)), dim=-1
        )
        q, s = pack_experts(codes, scale_zero)
        expected_q, expected_s = pack_marlin_batch(codes, scales, 32)
        assert tensorplay.equal(q, expected_q)
        assert tensorplay.equal(s, expected_s)
        packed = torch.from_numpy(codes.cpu().numpy())
        weight = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(experts, n, k)
        scale = torch.from_numpy(scales.float().cpu().numpy())
        weight = ((weight.float() - 8) * scale.repeat_interleave(32, dim=-1)).bfloat16()
        return q, s, weight

    q13, s13, w13 = make_projection(2 * width, hidden)
    q2, s2, w2 = make_projection(hidden, width)
    for tokens in (1, 23, 64):
        x = tensorplay.randn((tokens, hidden), device="cuda", dtype=tensorplay.bfloat16)
        _, ids = tensorplay.rand((tokens, experts), device="cuda").topk(top_k)
        weights = tensorplay.rand((tokens, top_k), device="cuda").softmax(dim=-1)
        actual = grouped_marlin_moe(x, ids, weights, q13, s13, q2, s2)

        reference_x = torch.from_numpy(x.float().cpu().numpy()).float()
        actual_projection = marlin_project(x, q13[0], s13[0])
        expected_projection = (reference_x @ w13[0].float().T).bfloat16().float()
        actual_projection = torch.from_numpy(actual_projection.float().cpu().numpy())
        assert (
            actual_projection - expected_projection
        ).norm() / expected_projection.norm() < 0.005
        reference_ids = torch.from_numpy(ids.cpu().numpy()).flatten()
        reference_w = torch.from_numpy(weights.cpu().numpy())
        routes = torch.empty((tokens * top_k, hidden))
        for expert in range(experts):
            indices = (reference_ids == expert).nonzero().flatten()
            gate_up = (reference_x[indices // top_k] @ w13[expert].float().T).bfloat16()
            gate, up = gate_up.chunk(2, dim=-1)
            activated = torch.nn.functional.gelu(gate, approximate="tanh") * up
            routes[indices] = (
                (activated.float() @ w2[expert].float().T).bfloat16().float()
            )
        expected = (routes.reshape(tokens, top_k, hidden) * reference_w[..., None]).sum(
            1
        )
        expected = expected.bfloat16().float()
        actual_cpu = torch.from_numpy(actual.float().cpu().numpy())
        assert torch.isfinite(actual_cpu).all()
        assert (actual_cpu - expected).norm() / expected.norm() < 0.006


@pytest.mark.parametrize("width", [17, 4129, 262144])
def test_diffusion_sampling_matches_full_vocab_reference(width: int) -> None:
    torch = pytest.importorskip("torch")
    tensorplay.manual_seed(33)
    logits = tensorplay.randn((2, 3, width), device="cuda", dtype=tensorplay.bfloat16)
    logits[0, 0] = 0
    uniforms = tensorplay.tensor(
        [[0.987, 0.29, 0.72], [0.001, 0.51, 0.96]],
        device="cuda",
        dtype=tensorplay.float32,
    )
    temperature, softcap = 0.57, 30.0
    sample, argmax, entropy, conditioning = sample_diffusion_logits(
        logits, uniforms, temperature, tensorplay.bfloat16, softcap=softcap
    )
    raw = torch.from_numpy(logits.float().cpu().numpy()).double()
    scaled = softcap * (raw / softcap).tanh() / temperature
    p = scaled.softmax(dim=-1)
    draw = torch.from_numpy(uniforms.cpu().numpy()).unsqueeze(-1)
    expected_sample = (p.cumsum(dim=-1) <= draw).sum(dim=-1)
    expected_entropy = -(p * scaled.log_softmax(dim=-1)).sum(dim=-1)
    expected_cond = scaled.bfloat16().double().softmax(dim=-1).bfloat16().float()
    assert torch.equal(torch.from_numpy(sample.cpu().numpy()), expected_sample)
    assert torch.equal(torch.from_numpy(argmax.cpu().numpy()), scaled.argmax(dim=-1))
    torch.testing.assert_close(
        torch.from_numpy(entropy.cpu().numpy()).double(),
        expected_entropy,
        atol=5e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        torch.from_numpy(conditioning.float().cpu().numpy()),
        expected_cond,
        atol=8e-6,
        rtol=0.008,
    )


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("causal", [True, False])
def test_diffusion_attention_causal_and_bidirectional(
    head_dim: int, causal: bool
) -> None:
    torch = pytest.importorskip("torch")
    tensorplay.manual_seed(7)
    q_len, kv_len = 37, 99
    q = tensorplay.randn((q_len, 4, head_dim), device="cuda", dtype=tensorplay.bfloat16)
    k = tensorplay.randn(
        (kv_len, 2, head_dim), device="cuda", dtype=tensorplay.bfloat16
    )
    v = tensorplay.randn(
        (kv_len, 2, head_dim), device="cuda", dtype=tensorplay.bfloat16
    )
    actual = diffusion_attention(q, k, v, causal=causal, sliding_window=47)

    qa, ka, va = (torch.from_numpy(t.float().cpu().numpy()) for t in (q, k, v))
    ka = ka.repeat_interleave(2, dim=1)
    va = va.repeat_interleave(2, dim=1)
    scores = (qa.transpose(0, 1) @ ka.permute(1, 2, 0)).bfloat16().float()
    qpos = torch.arange(kv_len - q_len, kv_len)[:, None]
    kpos = torch.arange(kv_len)[None, :]
    visible = kpos > qpos - 47
    if causal:
        visible &= kpos <= qpos
    probs = (
        scores.masked_fill(~visible, -float("inf")).softmax(dim=-1).bfloat16().float()
    )
    expected = (probs @ va.transpose(0, 1)).transpose(0, 1).reshape(q_len, -1)
    expected = expected.bfloat16().float()
    actual_cpu = torch.from_numpy(actual.float().cpu().numpy())
    torch.testing.assert_close(actual_cpu, expected, atol=0.025, rtol=0.01)


@pytest.mark.parametrize("shared_key_value", [True, False])
def test_diffusion_prepares_strided_qkv(shared_key_value: bool) -> None:
    tensorplay.manual_seed(59)
    tokens, head_dim = 13, 256
    packed = tensorplay.randn(
        (tokens, 8, head_dim), device="cuda", dtype=tensorplay.bfloat16
    )
    query, key = packed[:, :4], packed[:, 4:6]
    value = key if shared_key_value else packed[:, 6:]
    q_norm = tensorplay.rand((head_dim,), device="cuda").bfloat16()
    k_norm = tensorplay.rand((head_dim,), device="cuda").bfloat16()
    angles = tensorplay.rand((tokens, head_dim // 2), device="cuda")
    cos, sin = angles.cos(), angles.sin()
    eps = 1e-6
    actual_q, actual_k, actual_v = prepare_diffusion_qkv(
        query, key, value, q_norm, k_norm, cos, sin, eps
    )
    expected_q = _apply_rope(_rms_norm(query, q_norm, eps), cos, sin)
    expected_k = _apply_rope(_rms_norm(key, k_norm, eps), cos, sin)
    expected_v = _rms_norm(value, None, eps)
    for actual, expected in zip(
        (actual_q, actual_k, actual_v), (expected_q, expected_k, expected_v)
    ):
        assert tensorplay.allclose(
            actual.float(), expected.float(), atol=0.008, rtol=0.008
        )
