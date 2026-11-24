"""Microbenchmark for Multi-LoRA admission and forward overhead.

The script can benchmark any text-only generic causal-LM checkpoint.  When
``--model-path`` is omitted it builds a tiny synthetic llama checkpoint so the
benchmark runs without downloads; that mode measures relative overhead only,
not absolute tokens/s of a production model.

Example::

    python scripts/bench_lora.py --model-path /models/llama3-8b \\
        --adapter /adapters/sql --adapter /adapters/style \\
        --max-loras 2 --tokens 64 --iters 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import tensorplay
from tensorplay.serialization import save

from megalm.contracts.lora import LoRARequest
from megalm.contracts.request import SamplingParams
from megalm.kernels.ops import KernelPolicy
from megalm.models.generic_causal_lm.config import GenericCausalLMConfig
from megalm.models.generic_causal_lm.model import GenericCausalLM
from megalm.runtime.engine import Engine, EngineConfig

TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
RANK = 8


def _synthetic_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = GenericCausalLMConfig(
        model_type="llama",
        vocab_size=201024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
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
    model = GenericCausalLM(
        config,
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )
    sources: dict[str, tensorplay.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name == "model.layers.0.self_attn.qkv_proj.weight":
            sources["model.layers.0.attention.wqkv.weight"] = (
                tensorplay.ones_like(parameter)
            )
            continue
        if name == "model.layers.0.mlp.gate_up_weight":
            gate, up = parameter.split(
                (config.intermediate_size, config.intermediate_size), dim=0
            )
            sources["model.layers.0.feed_forward.w1.weight"] = tensorplay.ones_like(
                gate
            )
            sources["model.layers.0.feed_forward.w3.weight"] = tensorplay.ones_like(
                up
            )
            continue
        source_name = "lm_head.weight" if name == "lm_head_weight" else name
        sources[source_name] = tensorplay.ones_like(parameter)
    save({k: v.to("cpu") for k, v in sources.items()}, str(root / "model.safetensors"))
    native = {
        "model_type": "llama",
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "intermediate_size": config.intermediate_size,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": False,
    }
    (root / "config.json").write_text(json.dumps(native), encoding="utf-8")
    (root / "generation_config.json").write_text("{}", encoding="utf-8")
    bundled = Path(__file__).resolve().parents[1] / "model"
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        source = bundled / name
        if source.exists():
            (root / name).write_bytes(source.read_bytes())
    return root


def _adapter_geometry(engine: Engine) -> dict[str, tuple[int, int]]:
    config = engine.model.config
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    return {
        "q_proj": (q, hidden),
        "k_proj": (kv, hidden),
        "v_proj": (kv, hidden),
        "o_proj": (hidden, q),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }


def _synthetic_adapter(root: Path, geometry: dict[str, tuple[int, int]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tensorplay.Tensor] = {}
    for target in TARGETS:
        out_features, in_features = geometry[target]
        stem = f"base_model.model.model.layers.0.{target}"
        tensors[f"{stem}.lora_A.weight"] = tensorplay.randn(RANK, in_features) * 0.01
        tensors[f"{stem}.lora_B.weight"] = tensorplay.randn(out_features, RANK) * 0.01
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": RANK,
                "lora_alpha": 16,
                "target_modules": list(TARGETS),
                "bias": "none",
            }
        ),
        encoding="utf-8",
    )
    save(tensors, str(root / "adapter_model.safetensors"))
    return root


def _run_once(engine: Engine, lora: LoRARequest | None, tokens: int) -> Any:
    request = engine.add_request(
        [1, 2, 3, 4, 5, 6, 7, 8],
        max_new_tokens=tokens,
        sampling=SamplingParams(temperature=0.0, ignore_eos=True),
        lora_request=lora,
    )
    return engine.run_until_complete((request,))[0]


def _measure(
    engine: Engine, lora: LoRARequest | None, tokens: int, iters: int
) -> dict[str, Any]:
    latencies: list[float] = []
    emitted: list[int] = []
    # One untimed warmup keeps Triton/JIT compilation out of the samples.
    _run_once(engine, lora, tokens)
    for _ in range(iters):
        started = time.perf_counter()
        result = _run_once(engine, lora, tokens)
        latencies.append((time.perf_counter() - started) * 1000)
        emitted.append(len(result.token_ids))
    return {
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3),
        "tokens": emitted[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--max-loras", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not tensorplay.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")

    workdir = Path("/tmp/opencode/bench_lora")
    workdir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or _synthetic_checkpoint(workdir / "model")

    probe = Engine(
        EngineConfig(
            model_path=model_path,
            device="cuda",
            dtype="bfloat16",
            max_model_len=256,
            max_num_sequences=4,
            max_num_batched_tokens=256,
            prefill_chunk_size=128,
            enable_cuda_graph=False,
            enable_lora=False,
        )
    )
    geometry = _adapter_geometry(probe)
    del probe
    adapters = [Path(path) for path in args.adapter]
    if not adapters:
        adapters = [
            _synthetic_adapter(
                workdir / f"adapter_{index}", geometry
            )
            for index in range(args.max_loras)
        ]

    config = lambda: EngineConfig(
        model_path=model_path,
        device="cuda",
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        min_runtime_memory_bytes=64 * 1024**2,
        max_model_len=256,
        max_num_sequences=4,
        max_num_batched_tokens=256,
        prefill_chunk_size=128,
        enable_cuda_graph=False,
        enable_lora=True,
        lora_paths=tuple(str(path) for path in adapters[: args.max_loras]),
        max_loras=args.max_loras,
        max_lora_rank=32,
        lora_dtype="float16",
    )

    report: dict[str, Any] = {
        "model_path": str(model_path),
        "max_loras": args.max_loras,
        "tokens": args.tokens,
        "iters": args.iters,
        "disabled": _measure(Engine(replace_disable(config())), None, args.tokens, args.iters),
    }
    enabled = Engine(config())
    report["enabled_base"] = _measure(enabled, None, args.tokens, args.iters)
    for index, adapter in enumerate(adapters[: args.max_loras]):
        report[f"enabled_adapter_{index}"] = _measure(
            enabled,
            LoRARequest(lora_name=adapter.name, lora_int_id=index + 1),
            args.tokens,
            args.iters,
        )
    report["lora_metrics"] = {
        key: value
        for key, value in enabled.metrics.snapshot().items()
        if key.startswith("lora_")
    }
    report["manager_stats"] = enabled.lora_manager.stats()
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")


def replace_disable(config: EngineConfig) -> EngineConfig:
    """Return a copy of ``config`` with LoRA disabled."""

    from dataclasses import replace

    return replace(config, enable_lora=False, lora_paths=())


if __name__ == "__main__":
    main()
