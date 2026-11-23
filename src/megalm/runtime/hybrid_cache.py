from __future__ import annotations

from dataclasses import dataclass

import tensorplay

from megalm.runtime.cache import CacheGeometry, KVCacheError, PagedKVCache


@dataclass(frozen=True, slots=True)
class GDNStateGeometry:
    layer_indices: tuple[int, ...]
    num_slots: int
    conv_width: int
    conv_history: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int


class GDNStateCache:
    """Stable-address per-request GDN state arena owned by one Replica."""

    def __init__(
        self,
        *,
        layer_indices: tuple[int, ...],
        num_slots: int,
        conv_width: int,
        conv_kernel_dim: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        dtype: tensorplay.dtype,
        device: tensorplay.device | str,
    ) -> None:
        normalized_layers = tuple(layer_indices)
        values = (
            len(normalized_layers),
            num_slots,
            conv_width,
            conv_kernel_dim,
            num_value_heads,
            key_head_dim,
            value_head_dim,
        )
        if any(value <= 0 for value in values):
            raise KVCacheError("GDN cache dimensions must be positive")
        if (
            len(set(normalized_layers)) != len(normalized_layers)
            or any(index < 0 for index in normalized_layers)
        ):
            raise KVCacheError("GDN cache layer mapping is invalid")
        self.geometry = GDNStateGeometry(
            layer_indices=normalized_layers,
            num_slots=num_slots,
            conv_width=conv_width,
            conv_history=conv_kernel_dim - 1,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
        )
        self._layer_rows = {
            layer_index: row
            for row, layer_index in enumerate(normalized_layers)
        }
        self.conv = tensorplay.zeros(
            (
                len(normalized_layers),
                num_slots,
                conv_width,
                conv_kernel_dim - 1,
            ),
            dtype=dtype,
            device=device,
        )
        self.recurrent = tensorplay.zeros(
            (
                len(normalized_layers),
                num_slots,
                num_value_heads,
                key_head_dim,
                value_head_dim,
            ),
            dtype=tensorplay.float32,
            device=device,
        )

    @property
    def nbytes(self) -> int:
        return (
            self.conv.numel() * self.conv.element_size()
            + self.recurrent.numel() * self.recurrent.element_size()
        )

    def layer(
        self,
        layer_index: int,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        try:
            row = self._layer_rows[layer_index]
        except KeyError as exc:
            raise KVCacheError(f"invalid GDN cache layer {layer_index}") from exc
        return self.conv[row], self.recurrent[row]

    @tensorplay.no_grad()
    def copy_state(
        self,
        source: tuple[int, int],
        target: tuple[int, int],
    ) -> None:
        source_conv, source_recurrent = source
        target_conv, target_recurrent = target
        for index in (
            source_conv,
            target_conv,
            source_recurrent,
            target_recurrent,
        ):
            if index < 0 or index >= self.geometry.num_slots:
                raise KVCacheError("GDN state copy slot is out of range")
        if source_conv != target_conv:
            self.conv[:, target_conv].copy_(self.conv[:, source_conv])
        if source_recurrent != target_recurrent:
            self.recurrent[:, target_recurrent].copy_(
                self.recurrent[:, source_recurrent]
            )


class HybridCache:
    """One physical cache facade for Qwen3.5 full attention and GDN state."""

    def __init__(
        self,
        *,
        kv: PagedKVCache,
        gdn: GDNStateCache,
    ) -> None:
        if kv.device != gdn.conv.device:
            raise KVCacheError("hybrid cache arenas are on different devices")
        self.kv = kv
        self.gdn = gdn

    @property
    def geometry(self) -> CacheGeometry:
        return self.kv.geometry

    @property
    def device(self) -> tensorplay.device:
        return self.kv.device

    @property
    def nbytes(self) -> int:
        return self.kv.nbytes + self.gdn.nbytes

    def layer(self, layer_index: int) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        return self.kv.layer(layer_index)
