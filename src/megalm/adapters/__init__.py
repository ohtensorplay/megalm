"""Cold-path model adapters that translate model facts into ModelSpec."""

from megalm.adapters.generic_causal_lm import (
    generic_causal_lm_spec_from_config,
    load_generic_causal_lm_spec,
)
from megalm.adapters.inkling2 import (
    inkling2_model_spec_from_config,
    load_inkling2_model_spec,
)
from megalm.adapters.qwen2_5_omni import (
    qwen2_5_omni_model_spec_from_config,
)
from megalm.adapters.qwen3 import (
    load_qwen3_model_spec,
    qwen3_model_spec_from_config,
)
from megalm.adapters.qwen3_5 import (
    load_qwen3_5_model_spec,
    qwen3_5_model_spec_from_config,
)
from megalm.adapters.qwen3_vl import (
    load_qwen3_vl_model_spec,
    qwen3_vl_model_spec_from_config,
)
from megalm.adapters.oxr import (
    load_oxr_model_spec,
    oxr_model_spec_from_config,
)
from megalm.adapters.registry import load_model_spec

__all__ = [
    "generic_causal_lm_spec_from_config",
    "inkling2_model_spec_from_config",
    "load_generic_causal_lm_spec",
    "load_inkling2_model_spec",
    "load_model_spec",
    "load_oxr_model_spec",
    "load_qwen3_5_model_spec",
    "load_qwen3_model_spec",
    "load_qwen3_vl_model_spec",
    "qwen2_5_omni_model_spec_from_config",
    "qwen3_5_model_spec_from_config",
    "oxr_model_spec_from_config",
    "qwen3_model_spec_from_config",
    "qwen3_vl_model_spec_from_config",
]
