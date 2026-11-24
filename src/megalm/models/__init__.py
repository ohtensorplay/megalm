"""First-party model implementations."""

from megalm.models.generic_causal_lm import (
    GenericCausalLM,
    GenericCausalLMConfig,
)
from megalm.models.inkling2 import Inkling2Config, Inkling2ForCausalLM
from megalm.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM
from megalm.models.oxr import (
    OXRConfig,
    OXRForConditionalGeneration,
    OXRImageProcessor,
)
from megalm.models.qwen2_5_omni import Qwen25OmniConfig, Qwen25OmniThinker
from megalm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from megalm.models.qwen3_5 import (
    Qwen35Config,
    Qwen35ForConditionalGeneration,
)
from megalm.models.qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)
from megalm.models.registry import NativeModelError, load_native_model

__all__ = [
    "GenericCausalLM",
    "GenericCausalLMConfig",
    "Inkling2Config",
    "Inkling2ForCausalLM",
    "NanbeigeConfig",
    "NanbeigeForCausalLM",
    "NativeModelError",
    "OXRConfig",
    "OXRForConditionalGeneration",
    "OXRImageProcessor",
    "Qwen25OmniConfig",
    "Qwen25OmniThinker",
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "Qwen35Config",
    "Qwen35ForConditionalGeneration",
    "Qwen3VLConfig",
    "Qwen3VLForConditionalGeneration",
    "load_native_model",
]
