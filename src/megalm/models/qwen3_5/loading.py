from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import tensorplay

from megalm.loading import SafetensorsSource, WeightLoadReport
from megalm.models.qwen3_5.moe import Qwen35SparseMoeBlock

if TYPE_CHECKING:
    from megalm.models.qwen3_5.model import Qwen35ForConditionalGeneration


class Qwen35WeightError(RuntimeError):
    """Raised when a checkpoint cannot initialize the native packed model."""


def load_qwen35_weights(
    model: Qwen35ForConditionalGeneration,
    model_path: str | Path,
) -> WeightLoadReport:
    parameters = dict(model.named_parameters())
    loaded: set[str] = set()
    packed_parts: dict[str, set[str]] = {}
    full_attention = re.compile(
        r"model\.language_model\.layers\.(?P<layer>\d+)\."
        r"self_attn\.(?P<projection>q_proj|k_proj|v_proj)\.weight"
    )
    gdn_projection = re.compile(
        r"model\.language_model\.layers\.(?P<layer>\d+)\."
        r"linear_attn\.(?P<projection>in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)\.weight"
    )
    dense_projection = re.compile(
        r"model\.language_model\.layers\.(?P<layer>\d+)\."
        r"mlp\.(?P<projection>gate_proj|up_proj)\.weight"
    )
    expert_projection = re.compile(
        r"model\.language_model\.layers\.(?P<layer>\d+)\."
        r"mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<projection>gate_proj|up_proj|down_proj)\.weight"
    )
    shared_expert_projection = re.compile(
        r"model\.language_model\.layers\.(?P<layer>\d+)\."
        r"mlp\.shared_expert\."
        r"(?P<projection>gate_proj|up_proj|down_proj)\.weight"
    )
    mtp_full_attention = re.compile(
        r"mtp\.layers\.(?P<layer>\d+)\."
        r"self_attn\.(?P<projection>q_proj|k_proj|v_proj)\.weight"
    )
    mtp_dense_projection = re.compile(
        r"mtp\.layers\.(?P<layer>\d+)\."
        r"mlp\.(?P<projection>gate_proj|up_proj)\.weight"
    )

    def copy(target_name: str, tensor: tensorplay.Tensor) -> None:
        try:
            target = parameters[target_name]
        except KeyError as exc:
            raise Qwen35WeightError(
                f"checkpoint tensor has no native target: {target_name}"
            ) from exc
        if tuple(target.shape) != tuple(tensor.shape):
            raise Qwen35WeightError(
                f"shape mismatch for {target_name}: expected "
                f"{tuple(target.shape)}, got {tuple(tensor.shape)}"
            )
        target.copy_(
            tensor.to(device=target.device, dtype=target.dtype)
        )
        loaded.add(target_name)

    @tensorplay.no_grad()
    def consume(source_name: str, tensor: tensorplay.Tensor) -> None:
        if source_name.startswith("mtp."):
            if model.mtp is None:
                return
            match = mtp_full_attention.fullmatch(source_name)
            if match is not None:
                layer_index = int(match.group("layer"))
                projection = match.group("projection")
                try:
                    attention = model.mtp.layers[layer_index].self_attn
                except IndexError as exc:
                    raise Qwen35WeightError(
                        f"checkpoint supplied unexpected MTP layer {layer_index}"
                    ) from exc
                assert attention is not None
                target_name = (
                    f"mtp.layers.{layer_index}.self_attn.qkv_proj_weight"
                )
                offsets = {
                    "q_proj": (0, 2 * attention.query_width),
                    "k_proj": (
                        2 * attention.query_width,
                        2 * attention.query_width + attention.kv_width,
                    ),
                    "v_proj": (
                        2 * attention.query_width + attention.kv_width,
                        2 * attention.query_width + 2 * attention.kv_width,
                    ),
                }
                start, end = offsets[projection]
                target = attention.qkv_proj_weight[start:end]
                if tuple(target.shape) != tuple(tensor.shape):
                    raise Qwen35WeightError(
                        f"shape mismatch for packed {source_name}"
                    )
                target.copy_(
                    tensor.to(device=target.device, dtype=target.dtype)
                )
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                }:
                    loaded.add(target_name)
                return

            match = mtp_dense_projection.fullmatch(source_name)
            if match is not None:
                layer_index = int(match.group("layer"))
                projection = match.group("projection")
                try:
                    mlp = model.mtp.layers[layer_index].mlp
                except IndexError as exc:
                    raise Qwen35WeightError(
                        f"checkpoint supplied unexpected MTP layer {layer_index}"
                    ) from exc
                target_name = f"mtp.layers.{layer_index}.mlp.gate_up_weight"
                intermediate = model.config.text_config.intermediate_size
                start = 0 if projection == "gate_proj" else intermediate
                target = mlp.gate_up_weight[start : start + intermediate]
                if tuple(target.shape) != tuple(tensor.shape):
                    raise Qwen35WeightError(
                        f"shape mismatch for packed {source_name}"
                    )
                target.copy_(
                    tensor.to(device=target.device, dtype=target.dtype)
                )
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {"gate_proj", "up_proj"}:
                    loaded.add(target_name)
                return

            target_name = source_name
            replacements = (
                ("mtp.fc.weight", "mtp.fc_weight"),
                (".self_attn.o_proj.weight", ".self_attn.o_proj_weight"),
                (".mlp.down_proj.weight", ".mlp.down_proj_weight"),
            )
            for source_suffix, target_suffix in replacements:
                if target_name == source_suffix:
                    target_name = target_suffix
                    break
                if target_name.endswith(source_suffix):
                    target_name = (
                        target_name[: -len(source_suffix)] + target_suffix
                    )
                    break
            copy(target_name, tensor)
            return
        match = full_attention.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            projection = match.group("projection")
            attention = model.model.language_model.layers[
                layer_index
            ].self_attn
            if attention is None:
                raise Qwen35WeightError(
                    f"full-attention weight supplied for GDN layer {layer_index}"
                )
            target_name = (
                f"model.language_model.layers.{layer_index}."
                "self_attn.qkv_proj_weight"
            )
            offsets = {
                "q_proj": (0, 2 * attention.query_width),
                "k_proj": (
                    2 * attention.query_width,
                    2 * attention.query_width + attention.kv_width,
                ),
                "v_proj": (
                    2 * attention.query_width + attention.kv_width,
                    2 * attention.query_width + 2 * attention.kv_width,
                ),
            }
            start, end = offsets[projection]
            target = attention.qkv_proj_weight[start:end]
            if tuple(target.shape) != tuple(tensor.shape):
                raise Qwen35WeightError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(
                tensor.to(device=target.device, dtype=target.dtype)
            )
            packed_parts.setdefault(target_name, set()).add(projection)
            if packed_parts[target_name] == {
                "q_proj",
                "k_proj",
                "v_proj",
            }:
                loaded.add(target_name)
            return

        match = gdn_projection.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            projection = match.group("projection")
            gdn = model.model.language_model.layers[
                layer_index
            ].linear_attn
            if gdn is None:
                raise Qwen35WeightError(
                    f"GDN weight supplied for full layer {layer_index}"
                )
            if projection in {"in_proj_qkv", "in_proj_z"}:
                target_name = (
                    f"model.language_model.layers.{layer_index}."
                    "linear_attn.in_proj_qkvz_weight"
                )
                start = 0 if projection == "in_proj_qkv" else gdn.conv_width
                target = gdn.in_proj_qkvz_weight[
                    start : start + tensor.shape[0]
                ]
                expected = {"in_proj_qkv", "in_proj_z"}
            else:
                target_name = (
                    f"model.language_model.layers.{layer_index}."
                    "linear_attn.in_proj_ba_weight"
                )
                start = 0 if projection == "in_proj_b" else gdn.num_value_heads
                target = gdn.in_proj_ba_weight[
                    start : start + tensor.shape[0]
                ]
                expected = {"in_proj_b", "in_proj_a"}
            if tuple(target.shape) != tuple(tensor.shape):
                raise Qwen35WeightError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(
                tensor.to(device=target.device, dtype=target.dtype)
            )
            packed_parts.setdefault(target_name, set()).add(projection)
            if packed_parts[target_name] == expected:
                loaded.add(target_name)
            return

        match = expert_projection.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            expert_index = int(match.group("expert"))
            projection = match.group("projection")
            mlp = model.model.language_model.layers[layer_index].mlp
            if not isinstance(mlp, Qwen35SparseMoeBlock):
                raise Qwen35WeightError(
                    f"expert weight supplied for dense layer {layer_index}"
                )
            config = model.config.text_config
            if projection == "down_proj":
                target_name = (
                    f"model.language_model.layers.{layer_index}."
                    "mlp.down_weight"
                )
                target = mlp.down_weight[expert_index]
                expected_parts = config.num_experts
            else:
                target_name = (
                    f"model.language_model.layers.{layer_index}."
                    "mlp.gate_up_weight"
                )
                offset = (
                    0
                    if projection == "gate_proj"
                    else config.moe_intermediate_size
                )
                target = mlp.gate_up_weight[
                    expert_index,
                    offset : offset + config.moe_intermediate_size,
                ]
                expected_parts = config.num_experts * 2
            if tuple(target.shape) != tuple(tensor.shape):
                raise Qwen35WeightError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(
                tensor.to(device=target.device, dtype=target.dtype)
            )
            packed_parts.setdefault(target_name, set()).add(
                f"{expert_index}:{projection}"
            )
            relevant = {
                part
                for part in packed_parts[target_name]
                if (
                    projection == "down_proj"
                    and part.endswith(":down_proj")
                )
                or (
                    projection != "down_proj"
                    and not part.endswith(":down_proj")
                )
            }
            if len(relevant) == expected_parts:
                loaded.add(target_name)
            return

        match = shared_expert_projection.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            projection = match.group("projection")
            mlp = model.model.language_model.layers[layer_index].mlp
            if not isinstance(mlp, Qwen35SparseMoeBlock) or (
                mlp.shared_expert is None
            ):
                raise Qwen35WeightError(
                    f"shared expert weight supplied for layer {layer_index}"
                )
            intermediate = (
                model.config.text_config.shared_expert_intermediate_size
            )
            if projection == "down_proj":
                target_name = (
                    f"model.language_model.layers.{layer_index}."
                    "mlp.shared_expert.down_proj_weight"
                )
                target = mlp.shared_expert.down_proj_weight
                if tuple(target.shape) != tuple(tensor.shape):
                    raise Qwen35WeightError(
                        f"shape mismatch for {source_name}"
                    )
                target.copy_(
                    tensor.to(device=target.device, dtype=target.dtype)
                )
                loaded.add(target_name)
                return
            target_name = (
                f"model.language_model.layers.{layer_index}."
                "mlp.shared_expert.gate_up_weight"
            )
            start = 0 if projection == "gate_proj" else intermediate
            target = mlp.shared_expert.gate_up_weight[
                start : start + intermediate
            ]
            if tuple(target.shape) != tuple(tensor.shape):
                raise Qwen35WeightError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(
                tensor.to(device=target.device, dtype=target.dtype)
            )
            packed_parts.setdefault(target_name, set()).add(projection)
            if packed_parts[target_name] == {"gate_proj", "up_proj"}:
                loaded.add(target_name)
            return

        match = dense_projection.fullmatch(source_name)
        if match is not None:
            layer_index = int(match.group("layer"))
            projection = match.group("projection")
            mlp = model.model.language_model.layers[layer_index].mlp
            target_name = (
                f"model.language_model.layers.{layer_index}."
                "mlp.gate_up_weight"
            )
            intermediate = model.config.text_config.intermediate_size
            start = 0 if projection == "gate_proj" else intermediate
            target = mlp.gate_up_weight[start : start + intermediate]
            if tuple(target.shape) != tuple(tensor.shape):
                raise Qwen35WeightError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(
                tensor.to(device=target.device, dtype=target.dtype)
            )
            packed_parts.setdefault(target_name, set()).add(projection)
            if packed_parts[target_name] == {"gate_proj", "up_proj"}:
                loaded.add(target_name)
            return

        target_name = source_name
        replacements = (
            (".self_attn.o_proj.weight", ".self_attn.o_proj_weight"),
            (".linear_attn.conv1d.weight", ".linear_attn.conv1d_weight"),
            (".linear_attn.out_proj.weight", ".linear_attn.out_proj_weight"),
            (".mlp.down_proj.weight", ".mlp.down_proj_weight"),
        )
        for source_suffix, target_suffix in replacements:
            if target_name.endswith(source_suffix):
                target_name = (
                    target_name[: -len(source_suffix)] + target_suffix
                )
                break
        if source_name == "lm_head.weight":
            target_name = "lm_head_weight"
        copy(target_name, tensor)

    source = SafetensorsSource(model_path)
    report = source.stream(consume)
    missing = sorted(set(parameters) - loaded)
    if missing:
        raise Qwen35WeightError(
            f"checkpoint did not initialize native parameters: {missing}"
        )
    return report
