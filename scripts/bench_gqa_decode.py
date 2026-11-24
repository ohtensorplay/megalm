#!/usr/bin/env python3
"""Microbenchmark + gold check for grouped-query paged decode attention.

Measures ``gqa_decode_attention`` on one GPU across batch sizes that exercise
both the split-K + reduce path (few programs) and the direct output path
(``batch * num_kv_heads >= 128``).  With ``--baseline-file`` the previous
implementation can be loaded out of tree for an interleaved A/B.

Run:
    PYTHONPATH=src:../tensorplay-main \
      python scripts/bench_gqa_decode.py --output /tmp/opencode/gqa-bench.json

Every result carries the provenance required by docs/benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tensorplay as tp
from triton.testing import do_bench

from megalm.kernels import paged_decode_gqa as gqa


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def provenance(root: Path, kernel_path: Path) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    def nvidia(query: str) -> str:
        try:
            return subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                text=True,
            ).strip().splitlines()[0]
        except (OSError, IndexError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "commit": git("rev-parse", "HEAD"),
        "source_sha256": _sha256(kernel_path),
        "kernel_file": str(kernel_path),
        "driver": nvidia("driver_version"),
        "hardware": nvidia("name"),
        "compute_capability": nvidia("compute_cap"),
        "python": platform.python_version(),
        "torch_version": __import__("torch").__version__,
        "tensorplay_version": tp.__version__,
        "triton_version": __import__("triton").__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def reference(query, key_cache, value_cache, tables, lengths, page_size, scale):
    batch, heads, head_dim = query.shape
    kv_heads = key_cache.shape[-2]
    qpkv = heads // kv_heads
    out = tp.empty((batch, heads, head_dim), dtype=query.dtype, device=query.device)
    for b in range(batch):
        length = int(lengths[b].item())
        pages = tables[b, : (length + page_size - 1) // page_size].long()
        keys = key_cache[pages].reshape(-1, kv_heads, head_dim).float()[:length]
        values = value_cache[pages].reshape(-1, kv_heads, head_dim).float()[:length]
        q = query[b].float().reshape(kv_heads, qpkv, head_dim)
        scores = tp.einsum("hqd,htd->hqt", q, keys.transpose(0, 1)) * scale
        probabilities = tp.softmax(scores, dim=-1)
        gathered = tp.einsum("hqt,htd->hqd", probabilities, values.transpose(0, 1))
        out[b] = gathered.reshape(heads, head_dim).to(query.dtype)
    return out


def measure(batch: int, dtype, ctx: int, page_size: int, head_dim: int, qpkv: int, baseline=None):
    kv_heads = 8
    heads = kv_heads * qpkv
    num_pages = (ctx + page_size - 1) // page_size
    query = tp.randn((batch, heads, head_dim), dtype=dtype, device="cuda") * 0.1
    key_cache = tp.randn(
        (num_pages + 2, page_size, kv_heads, head_dim), dtype=dtype, device="cuda"
    ) * 0.1
    value_cache = tp.randn_like(key_cache) * 0.1
    tables = tp.stack(
        [tp.arange(1, num_pages + 3, dtype=tp.int32, device="cuda") for _ in range(batch)]
    )
    lengths = tp.full((batch,), ctx, dtype=tp.int32, device="cuda")
    output = tp.empty_like(query)
    scale = head_dim**-0.5

    def run(module):
        module.gqa_decode_attention(
            query, key_cache, value_cache, tables, lengths, page_size,
            output=output, scale=scale,
        )

    run(gqa)
    expected = reference(query, key_cache, value_cache, tables, lengths, page_size, scale)
    max_error = float((output.float() - expected.float()).abs().max().item())

    record = {
        "batch": batch,
        "dtype": str(dtype),
        "context_tokens": ctx,
        "page_size": page_size,
        "head_dim": head_dim,
        "query_heads": heads,
        "kv_heads": kv_heads,
        "max_abs_error_vs_reference": max_error,
        "direct_path": batch * kv_heads >= 128,
    }
    # Interleave A/B so clock drift affects both equally.
    current = sorted(do_bench(lambda: run(gqa)) * 1000 for _ in range(3))[1]
    record["current_us"] = round(current, 3)
    if baseline is not None:
        before = sorted(do_bench(lambda: run(baseline)) * 1000 for _ in range(3))[1]
        record["baseline_us"] = round(before, 3)
        record["speedup"] = round(before / current, 3)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/opencode/gqa-bench.json"))
    parser.add_argument("--baseline-file", type=Path, default=None)
    parser.add_argument("--batches", default="1,4,8,16,32")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    kernel_path = root / "src/megalm/kernels/paged_decode_gqa.py"
    baseline = _load_module(str(args.baseline_file), "gqa_baseline") if args.baseline_file else None

    results = [
        measure(batch, tp.bfloat16, args.context, 16, args.head_dim, 4, baseline)
        for batch in (int(v) for v in args.batches.split(","))
    ]
    payload = {
        "benchmark": "gqa_decode_attention",
        "provenance": provenance(root, kernel_path),
        "configuration": {
            "dtype": "bfloat16",
            "page_size": 16,
            "kv_heads": 8,
            "qpkv": 4,
            "timing": "triton.testing.do_bench, median of 3, warmup+rep defaults",
            "baseline_file_sha256": (
                _sha256(args.baseline_file) if args.baseline_file else None
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
