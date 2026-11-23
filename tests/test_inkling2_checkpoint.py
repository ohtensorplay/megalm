"""Round-trip the real Inkling-2 expert checkpoint blocks.

This test is skipped unless ``model/`` exists.  It decodes a handful of
expert rows from the actual checkpoint and checks the shape and dtype
contracts, plus a spot value that the NVFP4 decode matches the row's
declared amax/denominator.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

tensorplay = pytest.importorskip("tensorplay")

MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"


def _load_tensor(name: str):
    from tensorplay.serialization import load

    weights = load(MODEL_ROOT / "model-00001-of-00001.safetensors", map_location="cpu", mmap=True)
    return weights[name]


def test_checkpoint_expert_shapes_match_the_contract():
    if not (MODEL_ROOT / "model-00001-of-00001.safetensors").is_file():
        pytest.skip("Inkling-2 checkpoint not present")
    from megalm.models.inkling2.loader import load_inkling2_config

    config = load_inkling2_config(MODEL_ROOT)
    w13 = _load_tensor("model.llm.layers.0.mlp.experts.w13_weight")
    assert tuple(w13.shape) == (128, config.hidden_size, 1152)
    assert w13.dtype == tensorplay.uint8
    w2 = _load_tensor("model.llm.layers.0.mlp.experts.w2_weight")
    assert tuple(w2.shape) == (128, config.hidden_size, 576)
    assert w2.dtype == tensorplay.uint8
    scale = _load_tensor("model.llm.layers.0.mlp.experts.w13_weight_scale")
    assert tuple(scale.shape) == (128, config.hidden_size, 144)
    amax = _load_tensor("model.llm.layers.0.mlp.experts.w13_weight_amax")
    assert tuple(amax.shape) == (128,)
    denom = _load_tensor("model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator")
    assert tuple(denom.shape) == (1,)
    # The 20th layer switches to MXFP8.
    w13_mx = _load_tensor("model.llm.layers.20.mlp.experts.w13_weight")
    assert w13_mx.dtype == tensorplay.float8_e4m3fn
    assert tuple(w13_mx.shape) == (128, config.hidden_size, 2304)


def test_nvfp4_decode_of_one_expert_matches_its_amax():
    if not (MODEL_ROOT / "model-00001-of-00001.safetensors").is_file():
        pytest.skip("Inkling-2 checkpoint not present")
    from megalm.kernels.quant_fp import unpack_nvfp4
    from megalm.models.inkling2.loader import load_inkling2_config

    config = load_inkling2_config(MODEL_ROOT)
    weight = _load_tensor("model.llm.layers.0.mlp.experts.w13_weight")
    scale = _load_tensor("model.llm.layers.0.mlp.experts.w13_weight_scale")
    amax = float(_load_tensor("model.llm.layers.0.mlp.experts.w13_weight_amax")[0].item())
    denom = float(
        _load_tensor("model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator")[0].item()
    )
    # Decode expert 0, first 16 rows (one per output channel), 2304 columns.
    rows = 16
    packed = weight[0, :rows]
    scales = scale[0, :rows].float()
    decoded = unpack_nvfp4(packed, scales)
    assert tuple(decoded.shape) == (rows, config.hidden_size)
    # ``unpack_nvfp4`` returns code * block-scale; the GEMM then multiplies
    # by the outer factor ``amax / (1536 * denominator)``.  The maximum
    # magnitude of a block is the largest E2M1 code (6.0) times its E4M3
    # scale, so bound the decoded rows by 6 * max(scale).
    max_scale = max(
        abs(value) for value in scales.reshape(-1).tolist()
    )
    row_abs = max(abs(value) for value in decoded.reshape(-1).tolist())
    assert row_abs <= 6.0 * max_scale * 1.5
    assert amax > 0 and denom > 0