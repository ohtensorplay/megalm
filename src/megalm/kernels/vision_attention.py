from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import tensorplay
import tensorplay.nn.functional as F


@dataclass(frozen=True, slots=True)
class VarlenAttentionGroup:
    """One equal-length subset of a packed variable-length batch."""

    length: int
    batch_size: int
    flat_indices: tensorplay.Tensor | None


@dataclass(frozen=True, slots=True)
class GroupedVarlenAttentionLayout:
    """Segment metadata reused by every layer in one encoder launch."""

    token_count: int
    device: tensorplay.device
    groups: tuple[VarlenAttentionGroup, ...]


def build_grouped_varlen_attention_layout(
    cu_seqlens: tensorplay.Tensor,
) -> GroupedVarlenAttentionLayout:
    """Group packed segments once instead of synchronizing every layer."""

    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 2
        or cu_seqlens.dtype not in {tensorplay.int32, tensorplay.int64}
    ):
        raise ValueError(
            "attention cumulative lengths must be an integer vector"
        )
    boundaries = tuple(int(value) for value in cu_seqlens.tolist())
    if boundaries[0] != 0:
        raise ValueError("attention cumulative lengths must start at zero")
    grouped_starts: dict[int, list[int]] = defaultdict(list)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            raise ValueError("attention segments must be non-empty")
        grouped_starts[end - start].append(start)

    token_count = boundaries[-1]
    groups: list[VarlenAttentionGroup] = []
    for length, starts in grouped_starts.items():
        direct = (
            len(grouped_starts) == 1
            and starts == list(range(0, token_count, length))
        )
        flat_indices = None
        if not direct:
            flat_indices = (
                tensorplay.tensor(
                    starts,
                    dtype=tensorplay.long,
                    device=cu_seqlens.device,
                )[:, None]
                + tensorplay.arange(
                    length,
                    device=cu_seqlens.device,
                )[None, :]
            ).reshape(-1)
        groups.append(
            VarlenAttentionGroup(
                length=length,
                batch_size=len(starts),
                flat_indices=flat_indices,
            )
        )
    return GroupedVarlenAttentionLayout(
        token_count=token_count,
        device=cu_seqlens.device,
        groups=tuple(groups),
    )


def grouped_varlen_attention(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    layout: GroupedVarlenAttentionLayout | tensorplay.Tensor,
    *,
    scale: float,
) -> tensorplay.Tensor:
    """Run one Flash-SDPA launch per distinct packed segment length."""

    if isinstance(layout, tensorplay.Tensor):
        layout = build_grouped_varlen_attention_layout(layout)
    if (
        query.ndim != 3
        or key.shape != query.shape
        or value.shape != query.shape
        or query.shape[0] != layout.token_count
        or query.device != layout.device
    ):
        raise ValueError(
            "packed attention tensors do not match the prepared layout"
        )

    if (
        len(layout.groups) == 1
        and layout.groups[0].flat_indices is None
    ):
        group = layout.groups[0]
        return _sdpa_group(
            query,
            key,
            value,
            length=group.length,
            batch_size=group.batch_size,
            scale=scale,
        )

    output = tensorplay.empty_like(query)
    for group in layout.groups:
        flat_indices = group.flat_indices
        if flat_indices is None:
            raise ValueError(
                "direct attention layout cannot be mixed with indexed groups"
            )
        attended = _sdpa_group(
            query.index_select(0, flat_indices),
            key.index_select(0, flat_indices),
            value.index_select(0, flat_indices),
            length=group.length,
            batch_size=group.batch_size,
            scale=scale,
        )
        output.index_copy_(
            0,
            flat_indices,
            attended,
        )
    return output


def _sdpa_group(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    *,
    length: int,
    batch_size: int,
    scale: float,
) -> tensorplay.Tensor:
    heads = query.shape[1]
    head_dim = query.shape[2]

    def packed(source: tensorplay.Tensor) -> tensorplay.Tensor:
        return source.reshape(
            batch_size,
            length,
            heads,
            head_dim,
        ).transpose(1, 2)

    attended = F.scaled_dot_product_attention(
        packed(query),
        packed(key),
        packed(value),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return attended.transpose(1, 2).reshape(
        batch_size * length,
        heads,
        head_dim,
    )


__all__ = [
    "GroupedVarlenAttentionLayout",
    "VarlenAttentionGroup",
    "build_grouped_varlen_attention_layout",
    "grouped_varlen_attention",
]
