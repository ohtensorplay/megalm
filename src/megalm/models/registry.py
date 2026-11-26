from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tensorplay

from megalm.kernels.bindings import KernelBindings
from megalm.kernels.ops import KernelPolicy
from megalm.loading import WeightLoadReport
from megalm.models.diffusiongemma import DiffusionGemmaForBlockDiffusion
from megalm.models.gemma4 import Gemma4ForCausalLM
from megalm.models.generic_causal_lm import GenericCausalLM
from megalm.models.inkling2 import Inkling2ForCausalLM
from megalm.models.lfm3 import Lfm3ForCausalLM
from megalm.models.nanbeige import NanbeigeForCausalLM
from megalm.models.oxr import OXRForConditionalGeneration
from megalm.models.qwen2_5_omni import Qwen25OmniThinker
from megalm.models.qwen3 import Qwen3ForCausalLM
from megalm.models.qwen3_5 import Qwen35ForConditionalGeneration
from megalm.models.qwen3_vl import Qwen3VLForConditionalGeneration


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
    mtp_model_path: str | Path | None = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    collective: Any = None,
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
    if not model_type:
        architectures = payload.get("architectures", ())
        if (
            isinstance(architectures, (list, tuple))
            and architectures
            and str(architectures[0]) == "MiniCPMForCausalLM"
        ):
            model_type = "minicpm"
    _mtp_models = {
        "qwen3_5",
        "inkling2",
        "inkling2_loop",
        "gemma4",
        "gemma4_unified",
        "gemma4_text",
        "gemma4_unified_text",
    }
    if enable_mtp and model_type not in _mtp_models:
        raise NativeModelError(
            "native MTP is not available for model_type=" + repr(model_type)
        )
    generic_dense = {
        "llama",
        "mistral",
        "qwen2",
        "yi",
        "olmo2",
        "deepseek",
        "phi3",
        "internlm2",
        "internlm3",
        "seed_oss",
        "granite",
        "ernie4_5",
        "minicpm",
        "nanbeige",
    }
    if tp_size > 1 and model_type not in generic_dense and model_type not in {
        "gemma4",
        "gemma4_unified",
        "gemma4_text",
        "gemma4_unified_text",
    }:
        raise NativeModelError(
            f"TP>1 is not implemented for model_type={model_type!r}; "
            "only generic dense causal LMs are supported"
        )
    if model_type in {
        "gemma4",
        "gemma4_unified",
        "gemma4_text",
        "gemma4_unified_text",
    }:
        if tp_size > 1:
            raise NativeModelError("TP>1 is not implemented for Gemma-4")
        return Gemma4ForCausalLM.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
            enable_mtp=enable_mtp,
            mtp_model_path=mtp_model_path,
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
        "olmo2",
        "deepseek",
        "phi3",
        "internlm2",
        "internlm3",
        "seed_oss",
        "granite",
        "ernie4_5",
        "minicpm",
    }:
        return GenericCausalLM.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
            tp_size=tp_size,
            tp_rank=tp_rank,
            collective=collective,
        )
    if model_type == "qwen2_5_omni":
        return Qwen25OmniThinker.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
    if model_type == "lfm3":
        return Lfm3ForCausalLM.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
    if model_type == "nanbeige":
        return NanbeigeForCausalLM.from_pretrained(
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
    if model_type == "diffusion_gemma":
        raise NativeModelError(
            "diffusion_gemma is a block-diffusion model; load it with "
            "load_diffusion_model() and drive it with DiffusionGemmaGenerator "
            "instead of the autoregressive engine"
        )
    raise NativeModelError(f"unsupported native model_type={model_type!r}")


def load_diffusion_model(
    model_path: str | Path,
    *,
    device: tensorplay.device | str,
    dtype: tensorplay.dtype,
    kernel_policy: KernelPolicy,
    kernel_bindings: KernelBindings | None = None,
) -> DiffusionGemmaForBlockDiffusion:
    """Load a DiffusionGemma checkpoint for block-diffusion generation.

    DiffusionGemma has no autoregressive ``forward``; it is driven by
    :class:`~megalm.models.diffusiongemma.DiffusionGemmaGenerator`, so it is
    deliberately not routed through :func:`load_native_model`.
    """

    model, _ = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_path,
        device=device,
        dtype=dtype,
        kernel_policy=kernel_policy,
        kernel_bindings=kernel_bindings,
    )
    return model


__all__ = ["NativeModelError", "load_diffusion_model", "load_native_model"]
