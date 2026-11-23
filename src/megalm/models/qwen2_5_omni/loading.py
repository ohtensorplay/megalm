from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import tensorplay

from megalm.loading import SafetensorsSource, WeightLoadReport

if TYPE_CHECKING:
    from megalm.models.qwen2_5_omni.model import Qwen25OmniThinker


class OmniWeightError(RuntimeError):
    """Raised when official Omni tensors do not fill the packed model."""


@tensorplay.no_grad()
def load_thinker_weights(
    model: Qwen25OmniThinker,
    model_path: str | Path,
) -> WeightLoadReport:
    parameters = dict(model.named_parameters())
    loaded: set[str] = set()
    packed: dict[str, set[str]] = {}
    qkv = re.compile(
        r"thinker\.model\.layers\.(?P<layer>\d+)\.self_attn\."
        r"(?P<part>q_proj|k_proj|v_proj)\.(?P<kind>weight|bias)"
    )
    gate_up = re.compile(
        r"thinker\.model\.layers\.(?P<layer>\d+)\.mlp\."
        r"(?P<part>gate_proj|up_proj)\.weight"
    )
    vision_qkv = re.compile(
        r"thinker\.visual\.blocks\.(?P<layer>\d+)\.attn\."
        r"(?P<part>q|k|v)\.(?P<kind>weight|bias)"
    )
    vision_gate_up = re.compile(
        r"thinker\.visual\.blocks\.(?P<layer>\d+)\.mlp\."
        r"(?P<part>gate_proj|up_proj)\.(?P<kind>weight|bias)"
    )
    audio_qkv = re.compile(
        r"thinker\.audio_tower\.layers\.(?P<layer>\d+)\.self_attn\."
        r"(?P<part>q_proj|k_proj|v_proj)\.(?P<kind>weight|bias)"
    )

    def copy(target_name: str, tensor: tensorplay.Tensor) -> None:
        target = parameters.get(target_name)
        if target is None:
            raise OmniWeightError(
                f"Omni tensor has no native target {target_name!r}"
            )
        if tuple(target.shape) != tuple(tensor.shape):
            raise OmniWeightError(
                f"shape mismatch for {target_name}: "
                f"{tuple(target.shape)} != {tuple(tensor.shape)}"
            )
        target.copy_(tensor.to(device=model.device, dtype=model.dtype))
        loaded.add(target_name)

    def consume(source_name: str, tensor: tensorplay.Tensor) -> None:
        match = qkv.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            part = match.group("part")
            kind = match.group("kind")
            attention = model.model.layers[layer_index].self_attn
            target_name = (
                f"model.layers.{layer_index}.self_attn.qkv_{kind}"
            )
            widths = {
                "q_proj": attention.query_width,
                "k_proj": attention.kv_width,
                "v_proj": attention.kv_width,
            }
            starts = {
                "q_proj": 0,
                "k_proj": attention.query_width,
                "v_proj": attention.query_width + attention.kv_width,
            }
            start = starts[part]
            target = parameters[target_name]
            view = target[start : start + widths[part]]
            if tuple(view.shape) != tuple(tensor.shape):
                raise OmniWeightError(
                    f"shape mismatch for packed {source_name}"
                )
            view.copy_(tensor.to(device=model.device, dtype=model.dtype))
            packed.setdefault(target_name, set()).add(part)
            if packed[target_name] == {
                "q_proj",
                "k_proj",
                "v_proj",
            }:
                loaded.add(target_name)
            return
        match = gate_up.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            part = match.group("part")
            target_name = (
                f"model.layers.{layer_index}.mlp.gate_up_weight"
            )
            target = parameters[target_name]
            width = model.config.intermediate_size
            start = 0 if part == "gate_proj" else width
            view = target[start : start + width]
            if tuple(view.shape) != tuple(tensor.shape):
                raise OmniWeightError(
                    f"shape mismatch for packed {source_name}"
                )
            view.copy_(tensor.to(device=model.device, dtype=model.dtype))
            packed.setdefault(target_name, set()).add(part)
            if packed[target_name] == {"gate_proj", "up_proj"}:
                loaded.add(target_name)
            return
        match = vision_qkv.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            part = match.group("part")
            kind = match.group("kind")
            target_name = (
                f"visual.blocks.{layer_index}.attn.qkv.{kind}"
            )
            width = model.config.vision_config.hidden_size
            start = {"q": 0, "k": width, "v": 2 * width}[part]
            target = parameters[target_name][start : start + width]
            if tuple(target.shape) != tuple(tensor.shape):
                raise OmniWeightError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(tensor.to(device=model.device, dtype=model.dtype))
            packed.setdefault(target_name, set()).add(part)
            if packed[target_name] == {"q", "k", "v"}:
                loaded.add(target_name)
            return
        match = vision_gate_up.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            part = match.group("part")
            kind = match.group("kind")
            target_name = (
                f"visual.blocks.{layer_index}.mlp.gate_up_{kind}"
            )
            width = model.config.vision_config.intermediate_size
            start = 0 if part == "gate_proj" else width
            target = parameters[target_name][start : start + width]
            if tuple(target.shape) != tuple(tensor.shape):
                raise OmniWeightError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(tensor.to(device=model.device, dtype=model.dtype))
            packed.setdefault(target_name, set()).add(part)
            if packed[target_name] == {"gate_proj", "up_proj"}:
                loaded.add(target_name)
            return
        match = audio_qkv.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            part = match.group("part")
            kind = match.group("kind")
            width = model.config.audio_config.hidden_size
            start = {
                "q_proj": 0,
                "k_proj": width,
                "v_proj": 2 * width,
            }[part]
            target_name = (
                f"audio_tower.layers.{layer_index}.self_attn."
                f"qkv_{kind}"
            )
            target = parameters[target_name][start : start + width]
            if tuple(target.shape) != tuple(tensor.shape):
                raise OmniWeightError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(tensor.to(device=model.device, dtype=model.dtype))
            packed.setdefault(target_name, set()).add(part)
            required = (
                {"q_proj", "k_proj", "v_proj"}
                if kind == "weight"
                else {"q_proj", "v_proj"}
            )
            if packed[target_name] == required:
                loaded.add(target_name)
            return
        stripped = source_name.removeprefix("thinker.")
        target_name = {
            "lm_head.weight": "lm_head_weight",
            "visual.patch_embed.proj.weight": (
                "visual.patch_embed_weight"
            ),
            "visual.merger.ln_q.weight": "visual.merger.norm.weight",
            "visual.merger.mlp.0.weight": (
                "visual.merger.linear1_weight"
            ),
            "visual.merger.mlp.0.bias": "visual.merger.linear1_bias",
            "visual.merger.mlp.2.weight": (
                "visual.merger.linear2_weight"
            ),
            "visual.merger.mlp.2.bias": "visual.merger.linear2_bias",
        }.get(stripped, stripped)
        target_name = target_name.replace(
            ".self_attn.o_proj.weight",
            ".self_attn.o_proj_weight",
        ).replace(
            ".mlp.down_proj.weight",
            ".mlp.down_weight",
        ).replace(
            ".mlp.down_proj.bias",
            ".mlp.down_bias",
        )
        if target_name == "visual.patch_embed_weight":
            tensor = tensor.flatten(1)
        copy(target_name, tensor)

    source = SafetensorsSource(
        model_path,
        include_prefixes=(
            "thinker.model.",
            "thinker.lm_head.",
            "thinker.visual.",
            "thinker.audio_tower.",
        ),
    )
    report = source.stream(consume)
    missing = sorted(set(parameters) - loaded)
    if missing:
        raise OmniWeightError(
            f"Omni checkpoint did not initialize parameters: {missing[:8]}"
        )
    return report


__all__ = ["OmniWeightError", "load_thinker_weights"]
