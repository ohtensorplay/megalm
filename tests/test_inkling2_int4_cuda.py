import pytest
import tensorplay

from megalm.kernels.moe_grouped_int4 import (
    Int4MoEWorkspace,
    pack_int4_expert_rowmajor,
    routed_moe_grouped_int4,
)
from megalm.kernels.silu_mul import capped_swiglu_cuda
from megalm.models.inkling2.model import capped_swiglu

pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(),
    reason="grouped int4 MoE tests require CUDA",
)

_HIDDEN = 128
_INTERMEDIATE = 64
_GROUP = 32


def _dequant(packed, scale_zero, group_size):
    n, half = packed.shape
    codes = tensorplay.stack(
        (packed & 0xF, (packed >> 4) & 0xF), dim=-1
    ).reshape(n, half * 2)
    scale = (
        scale_zero[:, :, 0]
        .permute(1, 0)
        .repeat_interleave(group_size, dim=1)
    )
    zero = (
        scale_zero[:, :, 1]
        .permute(1, 0)
        .repeat_interleave(group_size, dim=1)
    )
    return (codes.float() - zero.float()) * scale.float()


def _reference(x, packed13, scale13, packed2, scale2, ids, weights, group_size):
    output = tensorplay.zeros(
        (x.shape[0], x.shape[1]),
        dtype=tensorplay.float32,
        device=x.device,
    )
    intermediate_size = packed2.shape[-1] * 2
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate_up = x[token : token + 1].float() @ _dequant(
                packed13[expert], scale13[expert], group_size
            ).T
            activated = capped_swiglu(
                gate_up[:, :intermediate_size],
                gate_up[:, intermediate_size:],
                1.0,
                1.0,
            ).to(x.dtype)
            selected = activated.float() @ _dequant(
                packed2[expert], scale2[expert], group_size
            ).T
            output[token] += selected[0] * weights[token, slot]
    return output.to(x.dtype)


def _fixture(num_experts, top, tokens, intermediate_size, group_size):
    tensorplay.manual_seed(23)
    device = "cuda"
    dtype = tensorplay.bfloat16
    x = tensorplay.randn(tokens, _HIDDEN, device=device, dtype=dtype) * 0.1
    packed13, scale13, packed2, scale2 = [], [], [], []
    for _ in range(num_experts):
        w13 = (
            tensorplay.randn(
                2 * intermediate_size, _HIDDEN, device=device, dtype=dtype
            )
            * 0.05
        )
        w2 = (
            tensorplay.randn(
                _HIDDEN, intermediate_size, device=device, dtype=dtype
            )
            * 0.05
        )
        p13, s13 = pack_int4_expert_rowmajor(w13, group_size)
        p2, s2 = pack_int4_expert_rowmajor(w2, group_size)
        packed13.append(p13)
        scale13.append(s13)
        packed2.append(p2)
        scale2.append(s2)
    packed13 = tensorplay.stack(packed13)
    scale13 = tensorplay.stack(scale13)
    packed2 = tensorplay.stack(packed2)
    scale2 = tensorplay.stack(scale2)
    ids = tensorplay.randint(
        0,
        num_experts,
        (tokens, top),
        device=device,
    )
    weights = tensorplay.softmax(
        tensorplay.randn(tokens, top, device=device), dim=-1
    )
    return x, packed13, scale13, packed2, scale2, ids, weights


@pytest.mark.parametrize(
    "num_experts,top,tokens,intermediate_size,group_size",
    [
        (8, 2, 4, _INTERMEDIATE, _GROUP),
        (16, 4, 1, _INTERMEDIATE, _GROUP),
        (16, 8, 16, _INTERMEDIATE, _GROUP),
        (8, 4, 1, 1152, 128),
        (8, 4, 3, 1152, 128),
        (8, 4, 5, 1152, 128),
    ],
)
def test_grouped_int4_moe_matches_reference_with_and_without_workspace(
    num_experts, top, tokens, intermediate_size, group_size
):
    (
        x,
        packed13,
        scale13,
        packed2,
        scale2,
        ids,
        weights,
    ) = _fixture(num_experts, top, tokens, intermediate_size, group_size)
    expected = _reference(x, packed13, scale13, packed2, scale2, ids, weights, group_size)
    kwargs = {
        "group_size": group_size,
        "num_experts": num_experts,
        "gate_softcap": 1.0,
        "up_softcap": 1.0,
        "interleaved": False,
        "cap_fn": capped_swiglu,
    }
    actual = routed_moe_grouped_int4(
        x, packed13, scale13, packed2, scale2, ids, weights, **kwargs
    )
    tensorplay.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    workspace = Int4MoEWorkspace.allocate(
        max_tokens=tokens,
        num_experts=num_experts,
        hidden_size=_HIDDEN,
        out_width13=2 * intermediate_size,
        top_k=top,
        dtype=x.dtype,
        device=x.device,
    )
    pooled = routed_moe_grouped_int4(
        x,
        packed13,
        scale13,
        packed2,
        scale2,
        ids,
        weights,
        workspace=workspace,
        **kwargs,
    )
    assert pooled.data_ptr() == workspace.output.data_ptr()
    assert workspace.nbytes == Int4MoEWorkspace.estimate_nbytes(
        max_tokens=tokens,
        num_experts=num_experts,
        hidden_size=_HIDDEN,
        out_width13=2 * intermediate_size,
        top_k=top,
        dtype=x.dtype,
    )
    tensorplay.testing.assert_close(pooled, expected, atol=3e-2, rtol=3e-2)
    # Determinism check: reusing the same arena with the same inputs must
    # reproduce the same result (the reduction is in slot order, not atomic).
    pooled = pooled.clone()
    replay = routed_moe_grouped_int4(
        x,
        packed13,
        scale13,
        packed2,
        scale2,
        ids,
        weights,
        workspace=workspace,
        **kwargs,
    )
    assert tensorplay.equal(replay, pooled)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("dtype", [tensorplay.bfloat16, tensorplay.float16])
def test_fused_capped_swiglu_rounds_like_reference(interleaved, dtype):
    tensorplay.manual_seed(11)
    values = tensorplay.randn(8, 2304, device="cuda", dtype=dtype) * 5
    if interleaved:
        gate, up = values[:, 0::2], values[:, 1::2]
    else:
        gate, up = values[:, :1152], values[:, 1152:]
    for gate_cap, up_cap in ((4.0, 25.0), (0.0, 0.0)):
        expected = capped_swiglu(gate, up, gate_cap, up_cap).to(dtype)
        assert tensorplay.equal(capped_swiglu_cuda(gate, up, gate_cap, up_cap), expected)
