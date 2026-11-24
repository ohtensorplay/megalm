from __future__ import annotations

import argparse
import json

from megalm.contracts.lora import LoRARequest
from megalm.model_source import add_model_source_args, model_path_from_args
from megalm.observability import configure_logging
from megalm.runtime.engine import Engine, EngineConfig


def _parse_compile_buckets(value: str) -> tuple[int, ...]:
    tokens = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not tokens:
        raise argparse.ArgumentTypeError("compile buckets cannot be empty")
    return tokens


def _parse_compile_regions(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the embedded MegaLM engine.")
    parser.add_argument("--model", required=True, help="local path or Hub repo id")
    add_model_source_args(parser)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--max-cache-tokens",
        type=int,
        default=0,
        help="page-aligned KV token cap; 0 profiles capacity from free GPU memory",
    )
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument(
        "--cuda-graph-batches",
        default="1,2,4,8,16",
        help="Comma-separated decode graph buckets.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=(
            "auto",
            "fp8_e4m3fn",
            "fp8_e5m2",
            "fp8_per_head",
            "fp8_per_token_head",
            "int4_per_token_head",
            "nvfp4",
        ),
        default="auto",
    )
    parser.add_argument("--kv-cache-key-scale", type=float, default=None)
    parser.add_argument("--kv-cache-value-scale", type=float, default=None)
    parser.add_argument(
        "--kv-cache-q-dtype", choices=("auto", "fp8_e4m3fn", "fp8_e5m2"), default="auto"
    )
    parser.add_argument("--kv-cache-q-scale", type=float, default=None)
    parser.add_argument(
        "--kv-cache-q-group",
        choices=("per_tensor", "per_kv_head"),
        default="per_tensor",
    )
    parser.add_argument("--native-fp8-attn", action="store_true")
    parser.add_argument("--nvfp4-global-scale", type=float, default=1.0)
    parser.add_argument("--int4-use-rht", action="store_true")
    parser.add_argument("--kv-host-cache-bytes", type=int, default=0)
    parser.add_argument("--kv-disk-cache-dir", default=None)
    parser.add_argument("--kv-disk-cache-bytes", type=int, default=0)
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable decode CUDA Graph capture (needed by eager executors).",
    )
    parser.add_argument(
        "--compile-backend",
        choices=("none", "stax"),
        default="none",
        help="cold-path Stax lowering for static text regions",
    )
    parser.add_argument(
        "--compile-regions",
        type=_parse_compile_regions,
        default=("dense_mlp",),
        help="comma-separated Stax regions: dense_mlp,attn_proj,vision_mlp",
    )
    parser.add_argument(
        "--compile-min-tokens",
        type=int,
        default=1024,
        help="do not compile buckets below this token count (Stax loses there)",
    )
    parser.add_argument(
        "--compile-buckets",
        type=_parse_compile_buckets,
        default=(),
        help="exact prefill token counts to compile (default: prefill chunk size)",
    )
    parser.add_argument(
        "--compile-max-artifact-bytes",
        type=int,
        default=2 * 1024**3,
        help="stop compiling when artifacts exceed this device memory budget (0 = no limit)",
    )
    parser.add_argument(
        "--compile-startup-budget",
        type=float,
        default=0.0,
        help="stop compiling after this many seconds (0 means no budget)",
    )
    parser.add_argument(
        "--compile-min-speedup",
        type=float,
        default=1.05,
        help="reject a compiled region below this measured speedup (0 admits every parity-clean bucket; diagnostic only)",
    )
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--lora-path", action="append", default=[])
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument(
        "--lora-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    args = parser.parse_args()
    configure_logging()
    model_path = model_path_from_args(args.model, args)
    graph_sizes = tuple(
        int(value) for value in args.cuda_graph_batches.split(",") if value
    )
    engine = Engine(
        EngineConfig(
            model_path=str(model_path),
            max_cache_tokens=(
                None if args.max_cache_tokens == 0 else args.max_cache_tokens
            ),
            max_model_len=args.max_model_len,
            max_num_sequences=args.max_sequences,
            enable_prefix_cache=not args.no_prefix_cache,
            kv_cache_dtype=args.kv_cache_dtype,
            kv_cache_key_scale=args.kv_cache_key_scale,
            kv_cache_value_scale=args.kv_cache_value_scale,
            kv_cache_q_dtype=args.kv_cache_q_dtype,
            kv_cache_q_scale=args.kv_cache_q_scale,
            kv_cache_q_group=args.kv_cache_q_group,
            native_fp8_attn=args.native_fp8_attn,
            nvfp4_global_scale=args.nvfp4_global_scale,
            int4_use_rht=args.int4_use_rht,
            kv_host_cache_bytes=args.kv_host_cache_bytes,
            kv_disk_cache_dir=args.kv_disk_cache_dir,
            kv_disk_cache_bytes=args.kv_disk_cache_bytes,
            cuda_graph_batch_sizes=graph_sizes,
            enable_cuda_graph=not args.no_cuda_graph,
            compile_backend=args.compile_backend,
            compile_regions=args.compile_regions,
            compile_min_speedup=args.compile_min_speedup,
            compile_buckets=args.compile_buckets,
            compile_min_tokens=args.compile_min_tokens,
            compile_max_artifact_bytes=args.compile_max_artifact_bytes,
            compile_startup_budget_seconds=args.compile_startup_budget,
            enable_lora=args.enable_lora,
            lora_paths=tuple(args.lora_path),
            max_loras=args.max_loras,
            max_lora_rank=args.max_lora_rank,
            lora_dtype=args.lora_dtype,
        )
    )
    lora_request = None
    if args.lora_path:
        adapters = engine.list_loras()
        if adapters:
            lora_request = LoRARequest(
                lora_name=str(adapters[0]["name"]),
                lora_int_id=int(adapters[0]["id"]),
            )
    results = engine.generate_chat(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        lora_request=lora_request,
    )
    cache_stats = engine.cache_manager.stats()
    prefix_stats = cache_stats.prefix
    print(
        json.dumps(
            {
                "results": [
                    {
                        "request_id": result.request_id,
                        "token_ids": result.token_ids,
                        "text": result.text,
                        "finish_reason": result.finish_reason.value,
                    }
                    for result in results
                ],
                "compile": engine.compile_report.as_dict(),
                "metrics": {
                    "steps": engine.metrics.steps,
                    "eager_steps": engine.metrics.eager_steps,
                    "graph_steps": engine.metrics.graph_steps,
                    "scheduled_tokens": engine.metrics.scheduled_tokens,
                    "emitted_tokens": engine.metrics.emitted_tokens,
                    "structured_requests": engine.metrics.structured_requests,
                    "structured_degraded": engine.metrics.structured_degraded,
                    "structured_mask_steps": engine.metrics.structured_mask_steps,
                    "structured_masked_rows": engine.metrics.structured_masked_rows,
                    "structured_rejected": engine.metrics.structured_rejected,
                    "structured_grammar_errors": (
                        engine.metrics.structured_grammar_errors
                    ),
                    "graph_captures": (
                        0
                        if engine.graph_runner is None
                        else engine.graph_runner.metrics.captures
                    ),
                    "graph_replays": (
                        0
                        if engine.graph_runner is None
                        else engine.graph_runner.metrics.replays
                    ),
                    "plan_digest": engine.execution_plan.plan_digest,
                    "graph_memory_bytes": (
                        0
                        if engine.graph_runner is None
                        else engine.graph_runner.memory_bytes
                    ),
                    "cache": {
                        "free_pages": cache_stats.free_pages,
                        "reclaimable_pages": cache_stats.reclaimable_pages,
                        "prefix": (
                            None
                            if prefix_stats is None
                            else {
                                "cached_pages": prefix_stats.cached_pages,
                                "flattened_page_refs": (
                                    prefix_stats.flattened_page_refs
                                ),
                                "request_hit_rate": prefix_stats.request_hit_rate,
                                "token_hit_rate": prefix_stats.token_hit_rate,
                                "evictions": prefix_stats.evictions,
                            }
                        ),
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
