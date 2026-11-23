from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tensorplay

from megalm.kernels.bindings import KernelBindings
from megalm.kernels.ops import KernelPolicy
from megalm.loading import WeightLoadReport
from megalm.models.generic_causal_lm import GenericCausalLM
from megalm.models.qwen3 import Qwen3ForCausalLM
from megalm.models.qwen3_5 import Qwen35ForConditionalGeneration
from megalm.models.qwen3_vl import Qwen3VLForConditionalGeneration
from megalm.models.oxr import OXRForConditionalGeneration
from megalm.models.qwen2_5_omni import Qwen25OmniThinker
from megalm.models.inkling2 import Inkling2ForCausalLM


class NativeModelError(ValueError):
    """Raised when no sealed first-party implementation owns a checkpoint."""


def load_native_model(
    model_path: str | Path,
    *,
    device: tensorplay.device | str,
    dtype: tensorplay.dtype,
    kernel_policy: KernelPolicy,
    kernel_bindings: KernelBindings,
    enable_mtp: bool = False,
) -> tuple[tensorplay.nn.Module, WeightLoadReport]:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeModelError(
            f"cannot load native model config from {config_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise NativeModelError("native model config must be an object")
    model_type = payload.get("model_type")
    if enable_mtp and model_type not in {"qwen3_5", "inkling2", "inkling2_loop"}:
        raise NativeModelError(
            "native MTP is available only for Qwen3.5 and Inkling-2 checkpoints"
        )
    if model_type in {"qwen3", "qwen3_moe"}:
        return Qwen3ForCausalLM.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
    if model_type in {"qwen3_vl", "qwen3_vl_moe"}:
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        return Qwen35ForConditionalGeneration.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
            enable_mtp=enable_mtp,
        )
    if model_type == "oxr":
        return OXRForConditionalGeneration.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
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
        return GenericCausalLM.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
    if model_type == "qwen2_5_omni":
        return Qwen25OmniThinker.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
    if model_type in {"inkling2", "inkling2_loop"}:
        return Inkling2ForCausalLM.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
            enable_mtp=enable_mtp,
        )
    raise NativeModelError(f"unsupported native model_type={model_type!r}")


__all__ = ["NativeModelError", "load_native_model"]
