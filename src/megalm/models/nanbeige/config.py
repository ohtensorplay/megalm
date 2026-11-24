"""Runtime configuration for the Nanbeige 4.2 looped transformer."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from megalm.adapters.nanbeige import _reject_unsupported_features
from megalm.models.generic_causal_lm.config import (
    GenericCausalLMConfig,
    GenericCausalLMConfigError,
)


@dataclass(frozen=True, slots=True)
class NanbeigeConfig(GenericCausalLMConfig):
    """Llama-family geometry plus the full-stack loop schedule."""

    num_loops: int = 1
    skip_loop_final_norm: bool = False

    @classmethod
    def from_path(cls, model_path: str | Path) -> NanbeigeConfig:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        try:
            payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GenericCausalLMConfigError(
                f"cannot load Nanbeige config from {config_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise GenericCausalLMConfigError("Nanbeige config must be an object")
        if str(payload.get("model_type", "")) != "nanbeige":
            raise GenericCausalLMConfigError(
                "Nanbeige config requires model_type='nanbeige'"
            )
        _reject_unsupported_features(payload)

        # Reuse the Llama-family geometry parser, then carry its frozen fields
        # into this subclass.  Calling it on the base class avoids ``super()``
        # binding quirks in classmethods of slotted dataclasses.
        base = GenericCausalLMConfig.from_path(model_path)
        try:
            num_loops = int(payload.get("num_loops", 1))
        except (TypeError, ValueError) as exc:
            raise GenericCausalLMConfigError("Nanbeige num_loops must be an integer") from exc
        if num_loops < 1:
            raise GenericCausalLMConfigError("Nanbeige num_loops must be positive")
        geometry = {
            field.name: getattr(base, field.name)
            for field in fields(GenericCausalLMConfig)
        }
        return cls(
            **geometry,
            num_loops=num_loops,
            skip_loop_final_norm=bool(payload.get("skip_loop_final_norm", False)),
        )


__all__ = ["NanbeigeConfig"]
