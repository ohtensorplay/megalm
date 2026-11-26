"""DiffusionGemma checkpoint configuration.

DiffusionGemma wraps a Gemma-4 text backbone in a block-diffusion loop: an
*encoder* pass runs the prefix with causal attention and fills the KV cache,
then a *decoder* pass denoises a canvas of ``canvas_length`` tokens with
bidirectional attention that reads (but never writes) the encoder cache.

The checkpoint layout mirrors that split (``model.decoder.*`` holds the single
weight stack; the encoder reuses it), so this module resolves the HF config
into the native Gemma-4 text geometry plus the diffusion knobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megalm.models.gemma4.config import Gemma4ConfigError, Gemma4TextConfig

DECODER_PREFIX = "model.decoder"
DEFAULT_CANVAS_LENGTH = 256


def _full_layers_share_kv(path: Path, text_payload: dict[str, Any]) -> bool:
    """Whether full-attention layers omit ``v_proj`` (value == key).

    The released DiffusionGemma configs drop ``attention_k_eq_v`` from the
    text config even though the global layers have no value projection.  A
    checkpoint that stores ``v_proj`` for a full layer uses a separate value;
    one that does not reuses the key projection.
    """

    layer_types = text_payload.get("layer_types") or ()
    full = next(
        (
            index
            for index, kind in enumerate(layer_types)
            if kind == "full_attention"
        ),
        None,
    )
    if full is None:
        return False
    names: set[str] = set()
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            names = set(payload["weight_map"])
        except (OSError, ValueError, KeyError, TypeError):
            names = set()
    else:
        shards = sorted(path.glob("*.safetensors"))
        if shards:
            try:
                from safetensors import safe_open

                with safe_open(str(shards[0]), "pt") as handle:
                    names = set(handle.keys())
            except Exception:  # noqa: BLE001 - layout detection is best effort
                names = set()
    if not names:
        return False
    prefix = f"model.decoder.layers.{full}.self_attn.v_proj."
    return not any(name.startswith(prefix) for name in names)


@dataclass(frozen=True, slots=True)
class DiffusionQuantSpec:
    """Weight-only int4 metadata for a ``compressed-tensors`` checkpoint."""

    group_size: int
    num_bits: int
    ignore: tuple[str, ...]

    def is_quantized(self, module_name: str) -> bool:
        """Whether ``module_name`` keeps packed int4 weights.

        ``module_name`` uses the checkpoint's own prefix (``model.decoder.…``).
        ``quantization_config.ignore`` entries are prefix matches, exactly like
        compressed-tensors applies them.
        """

        return not any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in self.ignore
        )

    @classmethod
    def from_payload(cls, payload: Any) -> DiffusionQuantSpec | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise Gemma4ConfigError("quantization_config must be an object")
        method = str(payload.get("quant_method", "compressed-tensors"))
        if method != "compressed-tensors":
            raise Gemma4ConfigError(
                f"unsupported DiffusionGemma quant_method {method!r}"
            )
        if str(payload.get("format", "")) != "pack-quantized":
            raise Gemma4ConfigError(
                "DiffusionGemma int4 needs the pack-quantized checkpoint format"
            )
        groups = payload.get("config_groups") or {}
        if not isinstance(groups, dict) or not groups:
            raise Gemma4ConfigError("quantization_config.config_groups is empty")
        group = next(iter(groups.values()))
        weights = (group or {}).get("weights") or {}
        num_bits = int(weights.get("num_bits", 4))
        if num_bits != 4:
            raise Gemma4ConfigError(
                f"only 4-bit pack-quantized weights are supported, got {num_bits}"
            )
        if not weights.get("symmetric", True):
            raise Gemma4ConfigError("DiffusionGemma int4 requires symmetric weights")
        group_size = int(weights.get("group_size", 0) or 0)
        if group_size <= 0:
            raise Gemma4ConfigError("int4 weights need a positive group_size")
        ignore = payload.get("ignore") or ()
        if not isinstance(ignore, (list, tuple)):
            raise Gemma4ConfigError("quantization_config.ignore must be a list")
        return cls(
            group_size=group_size,
            num_bits=num_bits,
            ignore=tuple(str(entry) for entry in ignore),
        )


@dataclass(frozen=True, slots=True)
class DiffusionGemmaConfig:
    """Resolved DiffusionGemma geometry and sampling defaults."""

    text: Gemma4TextConfig
    canvas_length: int
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    quant: DiffusionQuantSpec | None = None

    @property
    def vocab_size(self) -> int:
        return self.text.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text.num_hidden_layers

    def module_is_quantized(self, suffix: str) -> bool:
        """Whether a decoder module (``layers.0.self_attn.q_proj``) is int4."""

        if self.quant is None:
            return False
        return self.quant.is_quantized(f"{DECODER_PREFIX}.{suffix}")

    @classmethod
    def from_path(cls, model_path: str | Path) -> DiffusionGemmaConfig:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise Gemma4ConfigError(f"missing model config: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise Gemma4ConfigError(f"invalid JSON model config: {config_path}") from exc
        if not isinstance(payload, dict):
            raise Gemma4ConfigError("model config must be a JSON object")
        text_payload = payload.get("text_config")
        if (
            isinstance(text_payload, dict)
            and "attention_k_eq_v" not in text_payload
            and path.is_dir()
        ):
            # Some releases omit the flag even though the full-attention
            # layers store no ``v_proj``.  The checkpoint itself is the
            # authoritative source: without ``v_proj`` the value is the key.
            text_payload = dict(text_payload)
            text_payload["attention_k_eq_v"] = _full_layers_share_kv(
                path, text_payload
            )
            payload = dict(payload)
            payload["text_config"] = text_payload
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DiffusionGemmaConfig:
        if not isinstance(payload, dict):
            raise Gemma4ConfigError("model config must be a JSON object")
        model_type = payload.get("model_type")
        if model_type != "diffusion_gemma":
            raise Gemma4ConfigError(
                f"unsupported DiffusionGemma model_type {model_type!r}"
            )
        text_payload = payload.get("text_config")
        if not isinstance(text_payload, dict):
            raise Gemma4ConfigError("DiffusionGemma needs a text_config object")
        text_payload = dict(text_payload)
        text_payload["model_type"] = "gemma4_text"
        # DiffusionGemma's full-attention layers share K and V (no v_proj in
        # the checkpoint) and therefore take the global KV-head geometry.
        text_payload.setdefault("attention_k_eq_v", True)
        # The released configs omit ``enable_moe_block`` even though every
        # layer carries an expert block; infer it from the expert count.
        if "enable_moe_block" not in text_payload:
            text_payload["enable_moe_block"] = bool(
                int(text_payload.get("num_experts") or 0)
            )
        text = Gemma4TextConfig.from_dict(text_payload)

        canvas_length = int(payload.get("canvas_length") or DEFAULT_CANVAS_LENGTH)
        if canvas_length <= 0:
            raise Gemma4ConfigError("canvas_length must be positive")
        eos = payload.get("eos_token_id", 1)
        if isinstance(eos, (list, tuple)):
            eos_ids = tuple(int(value) for value in eos)
        else:
            eos_ids = (int(eos),)
        if not eos_ids:
            raise Gemma4ConfigError("DiffusionGemma needs at least one eos token")
        pad = payload.get("pad_token_id")
        pad_id = int(pad) if pad is not None else int(text_payload.get("pad_token_id", 0))
        return cls(
            text=text,
            canvas_length=canvas_length,
            eos_token_ids=eos_ids,
            pad_token_id=pad_id,
            quant=DiffusionQuantSpec.from_payload(payload.get("quantization_config")),
        )


__all__ = [
    "DECODER_PREFIX",
    "DiffusionGemmaConfig",
    "DiffusionQuantSpec",
]
