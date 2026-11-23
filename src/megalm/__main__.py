from __future__ import annotations

import argparse
import json
from pathlib import Path

from megalm.observability import configure_logging
from megalm.runtime.engine import Engine, EngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the embedded MegaLM engine.")
    parser.add_argument("--model", type=Path, required=True)
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
    parser.add_argument("--kv-host-cache-bytes", type=int, default=0)
    parser.add_argument("--kv-disk-cache-dir", default=None)
    parser.add_argument("--kv-disk-cache-bytes", type=int, default=0)
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable decode CUDA Graph capture (needed by eager executors).",
    )
    args = parser.parse_args()
    configure_logging()
    graph_sizes = tuple(
        int(value)
        for value in args.cuda_graph_batches.split(",")
        if value
    )
    engine = Engine(
        EngineConfig(
            model_path=args.model,
            max_cache_tokens=(
                None if args.max_cache_tokens == 0 else args.max_cache_tokens
            ),
            max_model_len=args.max_model_len,
            max_num_sequences=args.max_sequences,
            enable_prefix_cache=not args.no_prefix_cache,
            kv_host_cache_bytes=args.kv_host_cache_bytes,
            kv_disk_cache_dir=args.kv_disk_cache_dir,
            kv_disk_cache_bytes=args.kv_disk_cache_bytes,
            cuda_graph_batch_sizes=graph_sizes,
            enable_cuda_graph=not args.no_cuda_graph,
        )
    )
    results = engine.generate_chat(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
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
