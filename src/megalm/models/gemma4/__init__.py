from __future__ import annotations

from megalm.models.gemma4.config import (
    Gemma4AssistantConfig,
    Gemma4Config,
    Gemma4ConfigError,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from megalm.models.gemma4.model import Gemma4ForCausalLM

from megalm.models.gemma4.mtp import (
    Gemma4MultiTokenPredictor,
    load_gemma4_mtp_weights,
)

__all__ = [
    "Gemma4AssistantConfig",
    "Gemma4Config",
    "Gemma4ConfigError",
    "Gemma4ForCausalLM",
    "Gemma4MultiTokenPredictor",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "load_gemma4_mtp_weights",
]
