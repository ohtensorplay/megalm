"""Tests for static FP8 KV cache quantization."""

from pathlib import Path

import pytest
import tensorplay

from megalm.kernels import fallback, ops
from megalm.kernels.paged_prefill import paged_prefill_attention
from megalm.runtime.cache import PagedKVCache
from megalm.runtime.engine import EngineConfig
from megalm.runtime.kv_quant import (
    KVQuantConfig,
    KVQuantError,
    KVQuantMode,
    attach_kv_quant,
    attach_kv_scale,
    dequantize_tensor,
    kv_quant_of,
    kv_scale_of,
    quantize_tensor,
)

E5 = KVQuantConfig(tensorplay.float8_e5m2, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Configuration / geometry (device independent)
# ---------------------------------------------------------------------------


def test_kv_quant_config_validates_dtype_and_scales() -> None:
    with pytest.raises(KVQuantError):
        KVQuantConfig(tensorplay.float16)  # type: ignore[arg-type]
    with pytest.raises(KVQuantError):
        KVQuantConfig(tensorplay.float8_e5m2, key_scale=0.0)
    with pytest.raises(KVQuantError):
        KVQuantConfig(tensorplay.float8_e5m2, value_scale=-1.0)
    assert E5.triton_dtype == "e5m2"
    assert KVQuantConfig(
        tensorplay.float8_e4m3fn
    ).triton_dtype == "e4m3"
    assert E5.element_size == 1


def test_paged_cache_uses_quant_dtype_and_keeps_metadata() -> None:
    cache = PagedKVCache(
        num_layers=2,
        num_pages=4,
        page_size=8,
        num_kv_heads=2,
        head_dim=16,
        dtype=tensorplay.float16,
        device="cpu",
        quant=E5,
        compute_dtype=tensorplay.float16,
    )
    assert cache.key.dtype == tensorplay.float8_e5m2
    assert cache.compute_dtype == tensorplay.float16
    assert cache.element_size == 1
    assert cache.nbytes == 2 * 4 * 8 * 2 * 16 * 2
    key, value = cache.layer(1)
    assert kv_quant_of(key) is E5
    assert kv_quant_of(value) is E5
    view = attach_kv_quant(key.view(-1, 2, 16), kv_quant_of(key))
    assert kv_quant_of(view) is E5

    cache.register_conv_ring(1, channels=8, kernel_size=3)
    assert cache.conv_state(1).dtype == tensorplay.float16


def test_engine_config_kv_dtype_validation(tmp_path: Path) -> None:
    base = {"model_path": tmp_path / "model", "enable_cuda_graph": False}
    with pytest.raises(Exception, match="kv_cache_dtype"):
        EngineConfig(  # type: ignore[arg-type]
            **base, kv_cache_dtype="fp8_e4m3"
        )
    with pytest.raises(Exception, match="require a quantized"):
        EngineConfig(  # type: ignore[arg-type]
            **base, kv_cache_key_scale=0.5
        )
    with pytest.raises(Exception, match="positive and finite"):
        EngineConfig(  # type: ignore[arg-type]
            **base, kv_cache_dtype="fp8_e5m2", kv_cache_key_scale=0.0
        )
    config = EngineConfig(  # type: ignore[arg-type]
        **base,
        kv_cache_dtype="fp8_e5m2",
        kv_cache_key_scale=0.5,
        kv_cache_value_scale=2.0,
    )
    assert config.kv_cache_dtype == "fp8_e5m2"


# ---------------------------------------------------------------------------
# Native TP quant math + fallback kernels
# ---------------------------------------------------------------------------


def test_native_quantize_dequantize_roundtrip() -> None:
    values = tensorplay.tensor(
        [0.25, -1.5, 3.0, 0.001], dtype=tensorplay.float16
    )
    quant = KVQuantConfig(tensorplay.float8_e4m3fn, 1.0, 1.0)
    q = quantize_tensor(values, quant, "key")
    assert q.dtype == tensorplay.float8_e4m3fn
    d = dequantize_tensor(q, quant, "key", tensorplay.float16)
    tensorplay.testing.assert_close(d, values, atol=0.03, rtol=0.05)


def test_fallback_cache_write_and_decode_with_fp8() -> None:
    tensorplay.manual_seed(5)
    pages, page_size, heads, kv_heads, dim = 8, 16, 8, 4, 32
    query = tensorplay.randn(2, heads, dim, dtype=tensorplay.float16)
    key = tensorplay.randn(pages, page_size, kv_heads, dim, dtype=tensorplay.float16)
    value = tensorplay.randn_like(key)
    tables = tensorplay.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=tensorplay.int32)
    lengths = tensorplay.tensor([37, 63], dtype=tensorplay.int32)

    ref_cache = PagedKVCache(
        num_layers=1, num_pages=pages, page_size=page_size,
        num_kv_heads=kv_heads, head_dim=dim,
        dtype=tensorplay.float16, device="cpu",
    )
    quant_cache = PagedKVCache(
        num_layers=1, num_pages=pages, page_size=page_size,
        num_kv_heads=kv_heads, head_dim=dim,
        dtype=tensorplay.float16, device="cpu", quant=E5,
    )
    rk, rv = ref_cache.layer(0)
    qk, qv = quant_cache.layer(0)
    rk.copy_(key)
    rv.copy_(value)
    qk.copy_(quantize_tensor(key, E5, "key"))
    qv.copy_(quantize_tensor(value, E5, "value"))

    expected = fallback.paged_decode_attention(
        query, rk, rv, tables, lengths, page_size
    )
    actual = fallback.paged_decode_attention(
        query, qk, qv, tables, lengths, page_size
    )
    tensorplay.testing.assert_close(actual, expected, atol=0.2, rtol=0.2)


# ---------------------------------------------------------------------------
# Triton kernels (CUDA)
# ---------------------------------------------------------------------------

pytestmark_cuda = pytest.mark.skipif(
    not tensorplay.cuda.is_available(),
    reason="fp8 kernel tests require CUDA",
)


@pytestmark_cuda
def test_triton_decode_dequantizes_fp8_cache() -> None:
    tensorplay.manual_seed(7)
    dev = "cuda"
    pages, page_size, heads, kv_heads, dim = 8, 16, 8, 4, 64
    query = tensorplay.randn(2, heads, dim, device=dev, dtype=tensorplay.float16)
    key = tensorplay.randn(pages, page_size, kv_heads, dim, device=dev, dtype=tensorplay.float16)
    value = tensorplay.randn_like(key)
    tables = tensorplay.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=dev, dtype=tensorplay.int32)
    lengths = tensorplay.tensor([37, 63], device=dev, dtype=tensorplay.int32)

    ref = PagedKVCache(num_layers=1, num_pages=pages, page_size=page_size,
                       num_kv_heads=kv_heads, head_dim=dim,
                       dtype=tensorplay.float16, device=dev)
    quant = PagedKVCache(num_layers=1, num_pages=pages, page_size=page_size,
                         num_kv_heads=kv_heads, head_dim=dim,
                         dtype=tensorplay.float16, device=dev, quant=E5)
    rk, rv = ref.layer(0)
    qk, qv = quant.layer(0)
    rk.copy_(key)
    rv.copy_(value)
    qk.copy_(quantize_tensor(key, E5, "key"))
    qv.copy_(quantize_tensor(value, E5, "value"))

    expected = fallback.paged_decode_attention(query, rk, rv, tables, lengths, page_size)
    actual = ops.paged_decode_attention(query, qk, qv, tables, lengths, page_size)
    tensorplay.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)


@pytestmark_cuda
def test_triton_rope_write_quantizes_kv() -> None:
    tensorplay.manual_seed(11)
    dev = "cuda"
    tokens, qh, kh, dim = 3, 8, 4, 64
    query = tensorplay.randn(tokens, qh, dim, device=dev, dtype=tensorplay.float16)
    key = tensorplay.randn(tokens, kh, dim, device=dev, dtype=tensorplay.float16)
    value = tensorplay.randn_like(key)
    positions = tensorplay.tensor([0, 7, 11], device=dev)
    slots = tensorplay.tensor([0, 5, 13], device=dev)
    active = tensorplay.tensor([True, False, True], device=dev)
    inv = 1.0 / (
        1_000_000
        ** (tensorplay.arange(0, dim, 2, device=dev, dtype=tensorplay.float32) / dim)
    )
    frequencies = tensorplay.outer(
        tensorplay.arange(dim // 2, device=dev, dtype=tensorplay.float32), inv
    )
    cos, sin = frequencies.cos(), frequencies.sin()

    quant_cache = PagedKVCache(num_layers=1, num_pages=16, page_size=16,
                               num_kv_heads=kh, head_dim=dim,
                               dtype=tensorplay.float16, device=dev, quant=E5)
    qk, qv = quant_cache.layer(0)
    ops.rope_and_cache_write(
        query, key, value, positions, slots, cos, sin, qk, qv, active
    )

    ref_cache = PagedKVCache(num_layers=1, num_pages=16, page_size=16,
                             num_kv_heads=kh, head_dim=dim,
                             dtype=tensorplay.float16, device=dev)
    rk, rv = ref_cache.layer(0)
    fallback.rope_and_cache_write(
        query, key, value, positions, slots, cos, sin, rk, rv, active
    )
    flat_rk = rk.view(-1, kh, dim)
    flat_qk = qk.view(-1, kh, dim)
    assert flat_qk.dtype == tensorplay.float8_e5m2
    tensorplay.testing.assert_close(
        flat_qk[0].float(),
        quantize_tensor(flat_rk[0], E5, "key").float(),
        atol=0.03,
        rtol=0.05,
    )


@pytestmark_cuda
def test_triton_prefill_dequantizes_fp8_cache() -> None:
    tensorplay.manual_seed(13)
    dev = "cuda"
    pages, page_size, heads, kv_heads, dim = 8, 16, 8, 4, 64
    query = tensorplay.randn(8, heads, dim, device=dev, dtype=tensorplay.float16)
    key = tensorplay.randn(pages, page_size, kv_heads, dim, device=dev, dtype=tensorplay.float16)
    value = tensorplay.randn_like(key)
    tables = tensorplay.arange(pages, device=dev, dtype=tensorplay.int32).reshape(1, -1)
    starts = tensorplay.tensor([0, 8], device=dev, dtype=tensorplay.int32)
    positions = tensorplay.arange(56, 64, device=dev, dtype=tensorplay.int64)

    ref = PagedKVCache(num_layers=1, num_pages=pages, page_size=page_size,
                       num_kv_heads=kv_heads, head_dim=dim,
                       dtype=tensorplay.float16, device=dev)
    quant = PagedKVCache(num_layers=1, num_pages=pages, page_size=page_size,
                         num_kv_heads=kv_heads, head_dim=dim,
                         dtype=tensorplay.float16, device=dev, quant=E5)
    rk, rv = ref.layer(0)
    qk, qv = quant.layer(0)
    rk.copy_(key)
    rv.copy_(value)
    qk.copy_(quantize_tensor(key, E5, "key"))
    qv.copy_(quantize_tensor(value, E5, "value"))

    actual = paged_prefill_attention(
        query, qk, qv, tables, starts, positions,
        query_lengths=(8,), context_lengths=(64,), page_size=page_size,
    )
    # Same Triton kernel on an fp16 cache isolates the fp8 noise from
    # prefill-kernel implementation differences.
    expected = paged_prefill_attention(
        query, rk, rv, tables, starts, positions,
        query_lengths=(8,), context_lengths=(64,), page_size=page_size,
    )
    tensorplay.testing.assert_close(actual, expected, atol=0.2, rtol=0.2)


# ---------------------------------------------------------------------------
# All quantization modes (vLLM-aligned schemes)
# ---------------------------------------------------------------------------


ALL_MODES = [
    (KVQuantMode.FP8_PER_TENSOR, tensorplay.float8_e5m2),
    (KVQuantMode.FP8_PER_HEAD, tensorplay.float8_e5m2),
    (KVQuantMode.FP8_PER_TOKEN_HEAD, tensorplay.float8_e5m2),
    (KVQuantMode.INT4_PER_TOKEN_HEAD, tensorplay.uint8),
    (KVQuantMode.NVFP4, tensorplay.uint8),
]


def test_all_modes_layout_and_inline_scale(tmp_path: Path) -> None:
    for mode, dtype in ALL_MODES:
        kw = {}
        if mode == KVQuantMode.FP8_PER_HEAD:
            kw = {"num_kv_heads": 4, "key_scale": (1., 2., 3., 4.),
                  "value_scale": (1., 2., 3., 4.)}
        quant = KVQuantConfig(dtype=dtype, mode=mode, **kw)
        cache = PagedKVCache(
            num_layers=2, num_pages=8, page_size=16, num_kv_heads=4,
            head_dim=64, dtype=tensorplay.float16, device="cpu", quant=quant,
        )
        assert cache.key.shape[-1] == quant.content_dim(64)
        key, _ = cache.layer(0)
        scale = kv_scale_of(key)
        if quant.has_inline_scale:
            assert scale is not None
            assert tuple(scale.shape) == (8, 16, 4)
            assert scale.dtype == tensorplay.float32
        else:
            assert scale is None
        assert mode.value in quant.scale_signature()


def test_engine_config_all_modes() -> None:
    from megalm.runtime.engine import _resolve_kv_quant

    for name in (
        "fp8_e5m2", "fp8_per_head", "fp8_per_token_head",
        "int4_per_token_head", "nvfp4",
    ):
        config = EngineConfig(
            model_path=Path("/tmp/model"), enable_cuda_graph=False,
            kv_cache_dtype=name,
        )
        quant = _resolve_kv_quant(config, 4)
        assert quant is not None
        assert quant.mode.value in quant.scale_signature()


@pytestmark_cuda
@pytest.mark.parametrize("mode,dtype", ALL_MODES)
def test_all_modes_write_and_decode(mode, dtype) -> None:
    from megalm.kernels.qwen35 import active_kv_write

    tensorplay.manual_seed(5)
    pages, page_size, heads, kv_heads, dim = 4, 8, 4, 2, 64
    tokens = pages * page_size
    key = tensorplay.randn(tokens, kv_heads, dim, device="cuda",
                           dtype=tensorplay.float16) * 0.5
    value = tensorplay.randn_like(key)
    slots = tensorplay.arange(tokens, device="cuda", dtype=tensorplay.int32)
    active = tensorplay.ones(tokens, device="cuda", dtype=tensorplay.bool)

    kw = {"num_kv_heads": kv_heads} if mode == KVQuantMode.FP8_PER_HEAD else {}
    quant = KVQuantConfig(dtype=dtype, mode=mode, **kw)
    cache = PagedKVCache(
        num_layers=1, num_pages=pages, page_size=page_size,
        num_kv_heads=kv_heads, head_dim=dim, dtype=tensorplay.float16,
        device="cuda", quant=quant,
    )
    ck, cv = cache.layer(0)
    # ``view`` drops attached Python metadata, so re-attach the quant config
    # and inline scale views just like the model wrappers do.
    flat_k = attach_kv_quant(ck.view(-1, kv_heads, ck.shape[-1]), quant)
    flat_v = attach_kv_quant(cv.view(-1, kv_heads, cv.shape[-1]), quant)
    attach_kv_scale(flat_k, kv_scale_of(ck))
    attach_kv_scale(flat_v, kv_scale_of(cv))
    active_kv_write(key, value, flat_k, flat_v, slots, active)
    ref = PagedKVCache(
        num_layers=1, num_pages=pages, page_size=page_size,
        num_kv_heads=kv_heads, head_dim=dim, dtype=tensorplay.float16,
        device="cuda",
    )
    rk, rv = ref.layer(0)
    rk.view(-1, kv_heads, dim).copy_(key)
    rv.view(-1, kv_heads, dim).copy_(value)

    query = tensorplay.randn(2, heads, dim, device="cuda", dtype=tensorplay.float16)
    tables = tensorplay.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], device="cuda",
                               dtype=tensorplay.int32)
    lengths = tensorplay.tensor([31, 29], device="cuda", dtype=tensorplay.int32)
    actual = ops.paged_decode_attention(query, ck, cv, tables, lengths, page_size)
    expected = ops.paged_decode_attention(query, rk, rv, tables, lengths, page_size)
    tensorplay.cuda.synchronize()
    assert bool(tensorplay.isfinite(actual).all())
    tolerance = 0.2 if mode.is_fp8 else 1.0  # 4-bit without RHT
    tensorplay.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytestmark_cuda
def test_per_head_static_scale_decode() -> None:
    tensorplay.manual_seed(17)
    pages, page_size, heads, kv_heads, dim = 4, 8, 4, 2, 64
    key = tensorplay.randn(pages * page_size, kv_heads, dim, device="cuda",
                           dtype=tensorplay.float16) * 0.5
    value = tensorplay.randn_like(key)
    quant = KVQuantConfig(
        dtype=tensorplay.float8_e5m2,
        mode=KVQuantMode.FP8_PER_HEAD,
        num_kv_heads=kv_heads,
        key_scale=(1.0, 2.0),
        value_scale=(0.5, 1.5),
    )
    cache = PagedKVCache(
        num_layers=1, num_pages=pages, page_size=page_size,
        num_kv_heads=kv_heads, head_dim=dim, dtype=tensorplay.float16,
        device="cuda", quant=quant,
    )
    ck, cv = cache.layer(0)
    flat_k = ck.view(-1, kv_heads, dim)
    flat_v = cv.view(-1, kv_heads, dim)
    flat_k.copy_(quantize_tensor(key, quant, "key"))
    flat_v.copy_(quantize_tensor(value, quant, "value"))

    query = tensorplay.randn(1, heads, dim, device="cuda", dtype=tensorplay.float16)
    tables = tensorplay.arange(pages, device="cuda", dtype=tensorplay.int32).reshape(1, -1)
    lengths = tensorplay.tensor([31], device="cuda", dtype=tensorplay.int32)
    actual = ops.paged_decode_attention(query, ck, cv, tables, lengths, page_size)
    ref = PagedKVCache(
        num_layers=1, num_pages=pages, page_size=page_size,
        num_kv_heads=kv_heads, head_dim=dim, dtype=tensorplay.float16,
        device="cuda",
    )
    rk, rv = ref.layer(0)
    rk.reshape(-1, kv_heads, dim).copy_(key)
    rv.reshape(-1, kv_heads, dim).copy_(value)
    expected = ops.paged_decode_attention(query, rk, rv, tables, lengths, page_size)
    tensorplay.cuda.synchronize()
    tensorplay.testing.assert_close(actual, expected, atol=0.2, rtol=0.2)


def test_q_quant_config_and_native_gate() -> None:
    from megalm.runtime.engine import EngineError, _resolve_kv_quant

    config = EngineConfig(
        model_path=Path("/tmp/model"), enable_cuda_graph=False,
        kv_cache_dtype="fp8_e5m2", kv_cache_q_dtype="fp8_e5m2",
        kv_cache_q_scale=0.7, kv_cache_q_group="per_kv_head",
    )
    quant = _resolve_kv_quant(config, 4)
    assert quant.q_dtype == tensorplay.float8_e5m2
    assert quant.q_scale == (0.7, 0.7, 0.7, 0.7)
    assert quant.q_group == "per_kv_head"

    config = EngineConfig(
        model_path=Path("/tmp/model"), enable_cuda_graph=False,
        kv_cache_dtype="fp8_e5m2", native_fp8_attn=True,
    )
    with pytest.raises(EngineError, match="SM89"):
        _resolve_kv_quant(config, 4)


# ---------------------------------------------------------------------------
# INT4 randomized Hadamard rotation (RHT)
# ---------------------------------------------------------------------------


def test_rht_matrix_and_roundtrip() -> None:
    from megalm.kernels.kv_quant_triton import (
        get_rht_matrices,
        rht_enabled,
        rht_output,
        rht_query,
    )

    quant = KVQuantConfig(
        dtype=tensorplay.uint8,
        mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
        int4_use_rht=True,
    )
    assert rht_enabled(quant)
    assert not rht_enabled(KVQuantConfig(
        dtype=tensorplay.uint8,
        mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
    ))
    head_dim = 64
    matrix, signs = get_rht_matrices(head_dim, "cpu")
    assert tuple(matrix.shape) == (head_dim, head_dim)
    identity = matrix @ matrix / head_dim
    tensorplay.testing.assert_close(
        identity, tensorplay.eye(head_dim, dtype=tensorplay.float32)
    )
    query = tensorplay.randn(2, 4, head_dim, dtype=tensorplay.float16)
    rotated = rht_query(query, quant, head_dim)
    restored = rotated.float().reshape(-1, head_dim)
    restored = (restored @ matrix) * signs / head_dim
    tensorplay.testing.assert_close(
        restored.reshape(query.shape), query.float(), atol=0.01, rtol=0.01
    )
    out = tensorplay.randn(2, 4, head_dim, dtype=tensorplay.float16)
    rht_output(out, quant, head_dim)
    assert bool(tensorplay.isfinite(out).all())

    with pytest.raises(ValueError, match="power-of-two"):
        get_rht_matrices(63, "cpu")


def test_per_head_checkpoint_scales_override_cli() -> None:
    from megalm.runtime.engine import _resolve_kv_quant

    config = EngineConfig(
        model_path=Path("/tmp/model"), enable_cuda_graph=False,
        kv_cache_dtype="fp8_per_head", kv_cache_key_scale=0.5,
        kv_cache_value_scale=0.5, kv_cache_q_dtype="fp8_e5m2",
        kv_cache_q_group="per_kv_head", kv_cache_q_scale=0.5,
    )
    scales = {
        "key_scale": (1.0, 2.0, 3.0, 4.0),
        "value_scale": (0.1, 0.2, 0.3, 0.4),
        "q_scale": (0.7, 0.8, 0.9, 1.0),
    }
    quant = _resolve_kv_quant(config, 4, checkpoint_scales=scales)
    assert quant is not None
    assert quant.key_scale == scales["key_scale"]
    assert quant.value_scale == scales["value_scale"]
    assert quant.q_scale == scales["q_scale"]
    assert "key_scale=1,2,3,4" in quant.scale_signature()


def test_rht_engine_gate() -> None:
    from megalm.runtime.engine import EngineError, _resolve_kv_quant

    config = EngineConfig(
        model_path=Path("/tmp/model"), enable_cuda_graph=False,
        kv_cache_dtype="int4_per_token_head", int4_use_rht=True,
    )
    quant = _resolve_kv_quant(config, 2)
    assert quant is not None and quant.int4_use_rht
    assert "int4_use_rht=True" in quant.scale_signature()

    config = EngineConfig(
        model_path=Path("/tmp/model"), enable_cuda_graph=False,
        kv_cache_dtype="fp8_e5m2", int4_use_rht=True,
    )
    with pytest.raises(EngineError, match="int4_use_rht"):
        _resolve_kv_quant(config, 2)


@pytestmark_cuda
def test_int4_rht_reduces_outlier_quantization_error() -> None:
    from megalm.kernels.qwen35 import active_kv_write
    from megalm.runtime.kv_quant import attach_kv_quant, attach_kv_scale

    tensorplay.manual_seed(7)
    pages, page_size, heads, kv_heads, dim = 4, 8, 4, 2, 64
    tokens = pages * page_size
    # Outlier-heavy KV: RHT spreads the outliers and improves INT4 accuracy.
    key = tensorplay.randn(
        tokens, kv_heads, dim, device="cuda", dtype=tensorplay.float16
    ) * 0.2
    key[:, :, 3] += 6.0
    key[:, :, 17] -= 5.0
    value = tensorplay.randn_like(key) * 0.2
    value[:, :, 9] += 7.0
    slots = tensorplay.arange(tokens, device="cuda", dtype=tensorplay.int32)
    active = tensorplay.ones(tokens, device="cuda", dtype=tensorplay.bool)

    reference = PagedKVCache(
        num_layers=1, num_pages=pages, page_size=page_size,
        num_kv_heads=kv_heads, head_dim=dim, dtype=tensorplay.float16,
        device="cuda",
    )
    rk, rv = reference.layer(0)
    rk.view(-1, kv_heads, dim).copy_(key)
    rv.view(-1, kv_heads, dim).copy_(value)
    query = tensorplay.randn(2, heads, dim, device="cuda", dtype=tensorplay.float16)
    tables = tensorplay.arange(pages, device="cuda", dtype=tensorplay.int32)
    tables = tables.reshape(1, -1).expand(2, -1).contiguous()
    lengths = tensorplay.tensor([31, 29], device="cuda", dtype=tensorplay.int32)
    expected = ops.paged_decode_attention(query, rk, rv, tables, lengths, page_size)

    errors = {}
    for use_rht in (False, True):
        quant = KVQuantConfig(
            dtype=tensorplay.uint8,
            mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
            int4_use_rht=use_rht,
        )
        cache = PagedKVCache(
            num_layers=1, num_pages=pages, page_size=page_size,
            num_kv_heads=kv_heads, head_dim=dim, dtype=tensorplay.float16,
            device="cuda", quant=quant,
        )
        ck, cv = cache.layer(0)
        flat_k = attach_kv_quant(ck.view(-1, kv_heads, ck.shape[-1]), quant)
        flat_v = attach_kv_quant(cv.view(-1, kv_heads, cv.shape[-1]), quant)
        attach_kv_scale(flat_k, kv_scale_of(ck))
        attach_kv_scale(flat_v, kv_scale_of(cv))
        active_kv_write(key, value, flat_k, flat_v, slots, active)
        actual = ops.paged_decode_attention(query, ck, cv, tables, lengths, page_size)
        tensorplay.cuda.synchronize()
        assert bool(tensorplay.isfinite(actual).all())
        errors[use_rht] = float(
            tensorplay.max(tensorplay.abs(actual.float() - expected.float()))
        )
    assert errors[True] < errors[False]
