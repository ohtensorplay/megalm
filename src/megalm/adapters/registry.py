from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from megalm.adapters.generic_causal_lm import load_generic_causal_lm_spec
from megalm.adapters.qwen3 import qwen3_model_spec_from_config
from megalm.adapters.qwen3_5 import qwen3_5_model_spec_from_config
from megalm.adapters.qwen3_vl import qwen3_vl_model_spec_from_config
from megalm.adapters.oxr import oxr_model_spec_from_config
from megalm.adapters.qwen2_5_omni import (
    qwen2_5_omni_model_spec_from_config,
)
from megalm.adapters.inkling2 import load_inkling2_model_spec
from megalm.contracts.model import ModelContractError, ModelSpec


def load_model_spec(model_path: str | Path) -> ModelSpec:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelContractError(f"missing model config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelContractError(f"invalid JSON model config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ModelContractError("model config must be a JSON object")
    model_type = payload.get("model_type")
    if model_type in {"qwen3", "qwen3_moe"}:
        return qwen3_model_spec_from_config(payload)
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        return qwen3_5_model_spec_from_config(payload)
    if model_type in {"qwen3_vl", "qwen3_vl_moe"}:
        return qwen3_vl_model_spec_from_config(payload)
    if model_type == "oxr":
        return oxr_model_spec_from_config(payload)
    if model_type == "qwen2_5_omni":
        return qwen2_5_omni_model_spec_from_config(payload)
    if model_type in {
        "llama",
        "mistral",
        "qwen2",
        "qwen2_moe",
        "mixtral",
        "yi",
        "deepseek",
        "phi3",
        "internlm2",
        "internlm3",
        "seed_oss",
        "granite",
        "ernie4_5",
    }:
        return load_generic_causal_lm_spec(path)
    if model_type in {"inkling2_loop", "inkling2"}:
        # Re-load from the directory so checkpoint precision metadata (NVFP4/MXFP8
        # tensor counts from hf_quant_config.json) is folded into the digest.
        return load_inkling2_model_spec(path)
    raise ModelContractError(f"unsupported model_type={model_type!r}")
