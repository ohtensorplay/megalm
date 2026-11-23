from __future__ import annotations

import argparse
import json
from pathlib import Path

from megalm.contracts._canonical import canonical_value
from megalm.observability import configure_logging
from megalm.runtime.engine import Engine, EngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load the first-party Qwen3 executor and print its immutable "
            "production execution plan."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--max-cache-tokens",
        type=int,
        default=0,
        help="page-aligned KV token cap; 0 profiles capacity from free GPU memory",
    )
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=32_768)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument(
        "--cuda-graph-batches",
        default="1,2,4,8,16",
    )
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-prefix-cache", action="store_true")
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
            device=args.device,
            dtype=args.dtype,
            page_size=args.page_size,
            max_cache_tokens=(
                None if args.max_cache_tokens == 0 else args.max_cache_tokens
            ),
            max_model_len=args.max_sequence_length,
            max_num_sequences=args.max_batch_size,
            max_num_batched_tokens=args.max_batched_tokens,
            prefill_chunk_size=args.prefill_chunk_size,
            enable_prefix_cache=not args.no_prefix_cache,
            enable_cuda_graph=not args.no_cuda_graph,
            cuda_graph_batch_sizes=graph_sizes,
        )
    )
    output = {
        "milestone": "M1+M2-core",
        "executable_engine": True,
        "model": {
            "architecture": engine.model_spec.architecture,
            "model_type": engine.model_spec.model_type,
            "layers": engine.model_spec.num_layers,
            "model_digest": engine.model_spec.digest,
            "weight_tensors": engine.weight_report.tensor_count,
            "source_weight_bytes": engine.weight_report.total_bytes,
        },
        "plan": canonical_value(engine.execution_plan),
        "cache": {
            "backend": engine.execution_plan.cache.prefix_backend,
            "copy_on_write": engine.execution_plan.cache.copy_on_write,
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
