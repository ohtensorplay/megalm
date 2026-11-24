"""Qwen3-VL-8B AWQ int4 throughput in one model load (greedy, one sync).

Same protocol as ``bench_qwen3vl_awq_sglang.py``: a 512-token prompt repeated
``--batches`` times, ``--gen`` greedy tokens, single sync after the batch
finishes.  ``MEGALM_MARLIN=1`` selects the Marlin W4A16 kernel; unset/0 keeps
the Triton int4 path.

    PYTHONPATH=<tensorplay>:<megaLM/src> python bench_qwen3vl_awq.py \
        --batches 1,8,16 --reps 3
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import torch
import tensorplay
import triton

from megalm.contracts.request import SamplingParams
from megalm.runtime.engine import Engine, EngineConfig

DEFAULT_MODEL = (
    "/home/bluemoon/projects/jianjie-project/models/Qwen3-VL-8B-Instruct-AWQ-4bit"
)
PARA = (
    "The history of computing is a history of abstraction. Each generation of "
    "engineers hides the previous generation's complexity behind a new interface, "
    "and each new interface makes the next layer of machinery possible. A compiler "
    "hides registers; an operating system hides memory; a virtual machine hides the "
    "instruction set. When the abstraction is good, the machine feels simpler than "
    "it is, and the programmer can think about the problem instead of the substrate. "
    "When the abstraction leaks, the programmer is forced back down the stack, and "
    "the illusion collapses. "
)


def _metadata(model: str) -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return {
        "host": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "tensorplay": getattr(tensorplay, "__version__", None),
        "gpu": gpu,
        "model": model,
        "marlin": os.environ.get("MEGALM_MARLIN", "0"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batches", type=str, default="1,8,16")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen", type=int, default=129)
    ap.add_argument("--gens", type=str, default="",
                    help="comma list; overrides --gen for a prefill/decode split")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-batched", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--cache", type=int, default=32768)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--decode-chunk", type=int, default=1)
    args = ap.parse_args()

    cfg = EngineConfig(
        model_path=args.model,
        max_model_len=4096,
        max_cache_tokens=args.cache,
        max_num_sequences=max(16, int(args.batches.split(",")[-1])),
        max_num_batched_tokens=max(args.max_batched, args.chunk),
        prefill_chunk_size=args.chunk,
        mixed_prefill_chunk_size=args.chunk,
        gpu_memory_utilization=args.util,
        enable_cuda_graph=True,
        cuda_graph_batch_sizes=(1, 2, 4, 8, 16),
        enable_prefix_cache=False,
        strict_cuda_graph=True,
        decode_chunk_steps=args.decode_chunk,
    )
    t0 = time.perf_counter()
    engine = Engine(cfg)
    tensorplay.cuda.synchronize()
    load_s = time.perf_counter() - t0
    mem = tensorplay.cuda.memory_allocated() / 2**30
    print(f"# load_s={load_s:.1f} resident_alloc_gib={mem:.2f}", flush=True)

    base = engine.tokenizer.encode(PARA)
    prompt = (tuple(base) * (args.prompt_tokens // len(base) + 1))[
        : args.prompt_tokens
    ]

    out = {}
    gens = [int(x) for x in args.gens.split(",")] if args.gens else [args.gen]
    for batch in [int(x) for x in args.batches.split(",")]:
        for gen in gens:
            for rep in range(-args.warmup, args.reps):
                sampling = SamplingParams(temperature=0.0, ignore_eos=True)
                t_start = time.perf_counter()
                reqs = [
                    engine.add_request(
                        prompt, max_new_tokens=gen, sampling=sampling,
                        request_id=f"b{batch}-g{gen}-{rep}-{i}",
                    )
                    for i in range(batch)
                ]
                while not all(engine.scheduler.is_finished(r) for r in reqs):
                    engine.step()
                tensorplay.cuda.synchronize()
                total = time.perf_counter() - t_start
                ntok = sum(len(engine.scheduler.result_tokens(r)) for r in reqs)
                if rep >= 0:
                    out.setdefault(f"b{batch}_g{gen}", []).append(
                        {"total_s": round(total, 4), "tok_s": round(ntok / total, 1)}
                    )
                for r in reqs:
                    engine.scheduler.forget(r)
                print(f"# b{batch} gen{gen} rep{rep}: {ntok} tok in {total:.3f}s "
                      f"= {ntok / total:.1f} tok/s", flush=True)

    for batch in [int(x) for x in args.batches.split(",")]:
        if f"b{batch}_g1" in out and f"b{batch}_g129" in out:
            t1 = out[f"b{batch}_g1"][-1]["total_s"]
            t129 = out[f"b{batch}_g129"][-1]["total_s"]
            step = (t129 - t1) / 128
            print(f"# b{batch}: prefill+1step={t1 * 1e3:.1f}ms  "
                  f"decode_step={step * 1e3:.2f}ms  "
                  f"prefill~={(t1 - step) * 1e3:.1f}ms")

    print(json.dumps({
        "engine": "megalm",
        "metadata": _metadata(args.model),
        "gen": args.gen,
        "prompt_tokens": args.prompt_tokens,
        "chunk": args.chunk,
        "decode_chunk_steps": args.decode_chunk,
        "load_s": round(load_s, 1),
        "resident_alloc_gib": round(mem, 2),
        "results": out,
    }))


if __name__ == "__main__":
    main()
