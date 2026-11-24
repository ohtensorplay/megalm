"""SGLang baseline for Qwen3-VL-8B AWQ int4, same protocol as the MegaLM run.

Needs an environment where ``sglang`` imports (here: the ``llm`` conda env).
The prompt, batch sizes, greedy settings and timing rule match
``bench_qwen3vl_awq.py`` exactly.

    python bench_qwen3vl_awq_sglang.py --batches 1,8,16 --reps 3
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import sglang as sgl
from transformers import AutoTokenizer

DEFAULT_MODEL = os.environ.get("MEGALM_QWEN3VL_MODEL", "")
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
    import torch

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return {
        "host": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "sglang": sgl.__version__,
        "gpu": gpu,
        "model": model,
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
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--max-prefill", type=int, default=16384)
    ap.add_argument("--mem-frac", type=float, default=0.88)
    ap.add_argument("--defaults", action="store_true",
                    help="leave SGLang's own chunking/radix defaults alone")
    args = ap.parse_args()

    kw = dict(
        model_path=args.model,
        dtype="bfloat16",
        context_length=4096,
        mem_fraction_static=args.mem_frac,
        max_running_requests=max(16, int(args.batches.split(",")[-1])),
        log_level="error",
    )
    if not args.defaults:
        kw.update(
            chunked_prefill_size=args.chunk,
            max_prefill_tokens=args.max_prefill,
            disable_radix_cache=True,
        )

    t0 = time.perf_counter()
    engine = sgl.Engine(**kw)
    load_s = time.perf_counter() - t0
    print(f"# load_s={load_s:.1f}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    base = tok.encode(PARA)
    prompt = (base * (args.prompt_tokens // len(base) + 1))[: args.prompt_tokens]

    out = {}
    gens = [int(x) for x in args.gens.split(",")] if args.gens else [args.gen]
    for batch in [int(x) for x in args.batches.split(",")]:
        for gen in gens:
            sp = {"max_new_tokens": gen, "temperature": 0.0, "ignore_eos": True}
            for rep in range(-args.warmup, args.reps):
                t = time.perf_counter()
                outs = engine.generate(input_ids=[prompt] * batch,
                                       sampling_params=sp)
                dt = time.perf_counter() - t
                ntok = sum(len(o["output_ids"]) for o in outs)
                if rep >= 0:
                    out.setdefault(f"b{batch}_g{gen}", []).append(
                        {"total_s": round(dt, 4), "tok_s": round(ntok / dt, 1)}
                    )
                print(f"# b{batch} gen{gen} rep{rep}: {ntok} tok in {dt:.3f}s "
                      f"= {ntok / dt:.1f} tok/s", flush=True)

    for batch in [int(x) for x in args.batches.split(",")]:
        if f"b{batch}_g1" in out and f"b{batch}_g129" in out:
            t1 = out[f"b{batch}_g1"][-1]["total_s"]
            t129 = out[f"b{batch}_g129"][-1]["total_s"]
            step = (t129 - t1) / 128
            print(f"# b{batch}: prefill+1step={t1 * 1e3:.1f}ms  "
                  f"decode_step={step * 1e3:.2f}ms  "
                  f"prefill~={(t1 - step) * 1e3:.1f}ms")

    print(json.dumps({
        "engine": "sglang",
        "metadata": _metadata(args.model),
        "gen": args.gen,
        "prompt_tokens": args.prompt_tokens,
        "chunk": None if args.defaults else args.chunk,
        "load_s": round(load_s, 1),
        "results": out,
    }))
    engine.shutdown()


if __name__ == "__main__":
    main()
