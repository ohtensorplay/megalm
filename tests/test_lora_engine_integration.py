"""Engine-level multi-LoRA integration on a tiny synthetic generic checkpoint.

These tests exercise the full path (adapter loading, bank binding, per-token
slot ABI, Triton/reference kernel, scheduler admission, pin/unpin and metrics)
without downloading a real checkpoint.  They require CUDA because the engine's
dtype contract is fp16/bf16 and the LoRA hot path uses the Triton kernel there.
"""

from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import pytest

tensorplay = pytest.importorskip("tensorplay")
from tensorplay.serialization import save

from megalm.contracts.lora import LoRARequest
from megalm.contracts.request import SamplingParams
from megalm.kernels.ops import KernelPolicy
from megalm.models.generic_causal_lm.config import GenericCausalLMConfig
from megalm.models.generic_causal_lm.model import GenericCausalLM
from megalm.runtime.engine import Engine, EngineConfig
from megalm.runtime.scheduler import SchedulerLoRAError

pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(),
    reason="first-party engine LoRA integration requires CUDA",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _release_gpu_between_tests() -> None:
    yield
    gc.collect()
    if tensorplay.cuda.is_available():
        tensorplay.cuda.empty_cache()
TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
GEOMETRY = {
    "q_proj": (32, 16),
    "k_proj": (16, 16),
    "v_proj": (16, 16),
    "o_proj": (16, 32),
    "gate_proj": (16, 16),
    "up_proj": (16, 16),
    "down_proj": (16, 16),
}
RANK = 4


def _tiny_model() -> GenericCausalLM:
    config = GenericCausalLMConfig(
        model_type="llama",
        vocab_size=201024,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000,
        tie_word_embeddings=False,
        qkv_bias=False,
        qk_norm=False,
        num_experts=0,
        experts_per_token=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        decoder_sparse_step=1,
        mlp_only_layers=(),
    )
    return GenericCausalLM(
        config,
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )


def _synthetic_sources(model: GenericCausalLM) -> dict[str, tensorplay.Tensor]:
    """Map native parameter names onto the llama source checkpoint names."""

    sources: dict[str, tensorplay.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name == "model.layers.0.self_attn.qkv_proj.weight":
            sources["model.layers.0.attention.wqkv.weight"] = (
                tensorplay.ones_like(parameter)
            )
            continue
        if name == "model.layers.0.mlp.gate_up_weight":
            gate, up = parameter.split(
                (model.config.intermediate_size, model.config.intermediate_size),
                dim=0,
            )
            sources["model.layers.0.feed_forward.w1.weight"] = tensorplay.ones_like(
                gate
            )
            sources["model.layers.0.feed_forward.w3.weight"] = tensorplay.ones_like(
                up
            )
            continue
        source_name = (
            "lm_head.weight" if name == "lm_head_weight" else name
        )
        sources[source_name] = tensorplay.ones_like(parameter)
    return sources


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("lora_engine_checkpoint")
    model = _tiny_model()
    save(
        {key: value.to("cpu") for key, value in _synthetic_sources(model).items()},
        str(root / "model.safetensors"),
    )
    config = {
        "model_type": "llama",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "vocab_size": 201024,
        "max_position_embeddings": 64,
        "intermediate_size": 16,
        "rope_theta": 10000,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "generation_config.json").write_text("{}", encoding="utf-8")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        source = PROJECT_ROOT / "model" / name
        if source.exists():
            shutil.copy(source, root / name)
    return root


def _write_adapter(
    root: Path,
    *,
    name: str,
    magnitude: float = 1.0,
    alpha: int = 8,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tensorplay.Tensor] = {}
    for target in TARGETS:
        out_features, in_features = GEOMETRY[target]
        stem = f"base_model.model.model.layers.0.{target}"
        tensors[f"{stem}.lora_A.weight"] = tensorplay.full(
            (RANK, in_features), magnitude, dtype=tensorplay.float32
        )
        tensors[f"{stem}.lora_B.weight"] = tensorplay.full(
            (out_features, RANK), magnitude, dtype=tensorplay.float32
        )
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": RANK,
                "lora_alpha": alpha,
                "target_modules": list(TARGETS),
                "bias": "none",
                "use_rslora": False,
            }
        ),
        encoding="utf-8",
    )
    save(tensors, str(root / "adapter_model.safetensors"))
    return root


def _engine(
    checkpoint: Path,
    adapter: Path | None,
    *,
    max_loras: int = 2,
) -> Engine:
    return Engine(
        EngineConfig(
            model_path=checkpoint,
            device="cuda",
            dtype="bfloat16",
            gpu_memory_utilization=0.6,
            min_runtime_memory_bytes=64 * 1024**2,
            max_model_len=64,
            max_num_sequences=2,
            max_num_batched_tokens=64,
            prefill_chunk_size=32,
            mixed_prefill_chunk_size=32,
            enable_cuda_graph=False,
            enable_prefix_cache=True,
            enable_lora=True,
            lora_paths=() if adapter is None else (str(adapter),),
            max_loras=max_loras,
            max_lora_rank=16,
            lora_dtype="float16",
        )
    )


def _run(engine: Engine, lora: LoRARequest | None = None) -> tuple[int, ...]:
    request = engine.add_request(
        [1, 2, 3, 4, 5],
        max_new_tokens=3,
        sampling=SamplingParams(temperature=0.0),
        lora_request=lora,
    )
    result = engine.run_until_complete((request,))[0]
    return tuple(result.token_ids)


def test_engine_lora_end_to_end_and_pin_lifecycle(
    checkpoint: Path, tmp_path: Path
) -> None:
    adapter = _write_adapter(tmp_path / "adapter", name="adapter")
    engine = _engine(checkpoint, adapter)
    descriptor = engine.descriptor()["lora"]
    assert descriptor["enabled"] is True
    assert descriptor["registered"] == 1
    assert descriptor["adapters"][0]["name"] == "adapter"
    assert descriptor["adapters"][0]["slot"] == 0

    lora_tokens = _run(
        engine,
        LoRARequest(lora_name="adapter", lora_int_id=1),
    )
    base_tokens = _run(engine)
    assert len(lora_tokens) == len(base_tokens) == 3
    stats = engine.lora_manager.stats()
    assert stats["pins"] == 1
    assert stats["activations"] == 1
    assert stats["pinned_slots"] == 0  # released when the request finished
    snapshot = engine.metrics.snapshot()
    assert snapshot["lora_requests"] == 1
    assert snapshot["lora_steps"] >= 1
    assert snapshot["lora_tokens"] >= 1
    assert snapshot["lora_eager_steps"] == 0  # graphs disabled in this fixture


def test_engine_lora_bank_is_activated_only_for_lora_requests(
    checkpoint: Path, tmp_path: Path
) -> None:
    engine = _engine(checkpoint, None)
    state = engine.lora_manager.state
    assert state is not None
    assert bool(tensorplay.all(state.banks["qkv_proj"].b[0] == 0))
    _run(engine)  # base request must not activate a bank
    assert engine.metrics.snapshot()["lora_steps"] == 0
    assert bool(tensorplay.all(state.banks["qkv_proj"].b[0] == 0))
    engine.load_lora(
        str(_write_adapter(tmp_path / "strong", name="strong", magnitude=4.0)),
        name="strong",
        adapter_id=1,
    )
    assert not bool(tensorplay.all(state.banks["qkv_proj"].b[0] == 0))
    _run(engine)  # base request with a loaded adapter still uses base weights
    assert engine.metrics.snapshot()["lora_steps"] == 0
    _run(engine, LoRARequest(lora_name="strong", lora_int_id=1))
    assert engine.metrics.snapshot()["lora_steps"] >= 1


def test_engine_rejects_unknown_adapter_with_client_code(
    checkpoint: Path,
) -> None:
    engine = _engine(checkpoint, None)
    with pytest.raises(SchedulerLoRAError) as info:
        engine.add_request(
            [1, 2, 3],
            max_new_tokens=2,
            lora_request=LoRARequest(lora_name="missing"),
        )
    assert info.value.code == "lora_adapter_not_found"
    assert info.value.retryable is False
    assert engine.metrics.snapshot()["lora_requests"] == 0
    assert engine.scheduler.metrics_snapshot()["lora_rejections"] == 1
    terminal = engine.scheduler.drain_events()[-1]
    assert terminal.error is not None
    assert terminal.error.code == "lora_adapter_not_found"


def test_engine_lora_slot_exhaustion_is_retryable(
    checkpoint: Path, tmp_path: Path
) -> None:
    first = _write_adapter(tmp_path / "first", name="first")
    second = _write_adapter(tmp_path / "second", name="second")
    engine = _engine(checkpoint, None, max_loras=1)
    engine.load_lora(str(first), name="first", adapter_id=1)
    engine.load_lora(str(second), name="second", adapter_id=2, )
    first_request = engine.add_request(
        [1, 2, 3],
        max_new_tokens=2,
        lora_request=LoRARequest(lora_name="first", lora_int_id=1),
    )
    with pytest.raises(SchedulerLoRAError) as info:
        engine.add_request(
            [1, 2, 3],
            max_new_tokens=2,
            lora_request=LoRARequest(lora_name="second", lora_int_id=2),
        )
    assert info.value.code == "lora_slots_exhausted"
    assert info.value.retryable is True
    engine.run_until_complete((first_request,))
    # Once the first request releases its pin the second adapter can activate.
    retry = engine.add_request(
        [1, 2, 3],
        max_new_tokens=2,
        lora_request=LoRARequest(lora_name="second", lora_int_id=2),
    )
    assert len(engine.run_until_complete((retry,))) == 1


def test_engine_dynamic_load_and_unload(
    checkpoint: Path, tmp_path: Path
) -> None:
    engine = _engine(checkpoint, None)
    adapter = _write_adapter(tmp_path / "dynamic", name="dynamic")
    outcome = engine.load_lora(str(adapter), name="dynamic", adapter_id=7)
    assert outcome["id"] == 7
    assert engine.descriptor()["lora"]["registered"] == 1
    request = engine.add_request(
        [1, 2, 3],
        max_new_tokens=2,
        lora_request=LoRARequest(lora_name="dynamic", lora_int_id=7),
    )
    with pytest.raises(Exception, match="still used"):
        engine.unload_lora("dynamic")
    engine.run_until_complete((request,))
    assert engine.unload_lora("dynamic")["unloaded"] is True
    assert engine.descriptor()["lora"]["registered"] == 0
    assert engine.lora_manager.stats()["unloads"] == 1


def test_engine_allowlist_covers_startup_paths_but_blocks_others(
    checkpoint: Path, tmp_path: Path
) -> None:
    from megalm.runtime.engine import EngineLoRAError

    startup = _write_adapter(tmp_path / "startup", name="startup")
    outside = _write_adapter(tmp_path / "outside", name="outside")
    engine = Engine(
        EngineConfig(
            model_path=checkpoint,
            device="cuda",
            dtype="bfloat16",
            gpu_memory_utilization=0.6,
            min_runtime_memory_bytes=64 * 1024**2,
            max_model_len=64,
            max_num_sequences=2,
            max_num_batched_tokens=64,
            prefill_chunk_size=32,
            enable_cuda_graph=False,
            enable_lora=True,
            lora_paths=(str(startup),),
            lora_allowed_roots=(str(tmp_path / "unrelated"),),
            max_loras=2,
            max_lora_rank=16,
            lora_dtype="float16",
        )
    )
    assert engine.descriptor()["lora"]["registered"] == 1
    with pytest.raises(EngineLoRAError) as info:
        engine.load_lora(str(outside), name="outside", adapter_id=2)
    assert info.value.code == "lora_invalid"
