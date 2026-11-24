"""Fair HTTP-level benchmark: MegaLM vs SGLang on Nanbeige4.2-3B.

Streams OpenAI-compatible chat completions from both servers with identical
prompts, greedy sampling and output budget, and reports TTFT / E2E / decode
throughput at several concurrency levels.  Uses only the Python stdlib so it
can run from either conda environment.

Example::

    python tests/bench_nanbeige_compare.py \
        --megalm http://127.0.0.1:8000 --sglang http://127.0.0.1:30000 \
        --concurrency 1,2,4 --max-tokens 64 --trials 3 --out /tmp/opencode/nb_bench.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

DEFAULT_PROMPT = "你好，请用一句话介绍你自己。"
PREFIX = (
    "请仔细阅读下面的背景材料，然后回答问题。背景材料：人工智能助手需要"
    "在处理长上下文时保持稳定，同时兼顾首 token 延迟与整体吞吐。"
)


@dataclass(frozen=True)
class Sample:
    ttft_ms: float
    e2e_ms: float
    output_tokens: int
    prompt_tokens: int = 0
    error: str = ""


def _stream_once(url: str, prompt: str, max_tokens: int, *, shared: bool) -> Sample:
    payload = {
        "model": "nanbeige4.2-3b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    prompt_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw in response:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data in ("", "[DONE]"):
                    continue
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        if ttft is None:
                            ttft = (time.perf_counter() - started) * 1000
                        tokens += 1
                usage = chunk.get("usage") or {}
                reported = usage.get("completion_tokens")
                if isinstance(reported, int) and reported > 0:
                    tokens = reported
                prompt_reported = usage.get("prompt_tokens")
                if isinstance(prompt_reported, int) and prompt_reported > 0:
                    prompt_tokens = prompt_reported
        e2e = (time.perf_counter() - started) * 1000
        if ttft is None:
            return Sample(0, e2e, tokens, prompt_tokens, "no content delta received")
        return Sample(ttft, e2e, tokens, prompt_tokens)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return Sample(0, (time.perf_counter() - started) * 1000, 0, 0, str(exc))


def _run_scenario(
    url: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    trials: int,
) -> dict[str, object]:
    # Warm-up once (not measured) to bracket clock drift between the engines.
    warm = _stream_once(url, prompt, max_tokens, shared=True)
    samples: list[Sample] = []
    for _ in range(trials):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_stream_once, url, prompt, max_tokens, shared=True)
                for _ in range(concurrency)
            ]
            samples.extend(future.result() for future in futures)
    errors = [sample.error for sample in samples if sample.error]
    good = [sample for sample in samples if not sample.error]
    if not good:
        return {"concurrency": concurrency, "error": errors[0] if errors else "no samples"}
    total_tokens = sum(sample.output_tokens for sample in good)
    wall_seconds = max(sample.e2e_ms for sample in good) / 1000
    return {
        "concurrency": concurrency,
        "requests": len(good),
        "errors": len(errors),
        "ttft_ms_median": round(statistics.median(s.ttft_ms for s in good), 1),
        "ttft_ms_p95": round(sorted(s.ttft_ms for s in good)[int(0.95 * (len(good) - 1))], 1),
        "e2e_ms_median": round(statistics.median(s.e2e_ms for s in good), 1),
        "decode_tok_s": round(total_tokens / wall_seconds, 2),
        "output_tokens": total_tokens,
        "output_tokens_per_request": round(total_tokens / len(good), 1),
        "prompt_tokens_per_request": round(
            statistics.median(s.prompt_tokens for s in good), 1
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megalm", default="http://127.0.0.1:8000")
    parser.add_argument("--sglang", default="http://127.0.0.1:30000")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", default="")
    parser.add_argument("--only", choices=("megalm", "sglang"), default=None,
                        help="benchmark a single engine (run them one at a time on one GPU)")
    args = parser.parse_args()

    levels = [int(value) for value in args.concurrency.split(",") if value.strip()]
    engines = {"megalm": args.megalm, "sglang": args.sglang}
    if args.only:
        engines = {args.only: engines[args.only]}
    report: dict[str, object] = {
        "model": "Nanbeige/Nanbeige4.2-3B",
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "prompt": args.prompt,
        "results": {},
    }
    for name, url in engines.items():
        print(f"== {name} @ {url} ==", flush=True)
        rows = []
        for level in levels:
            row = _run_scenario(url, args.prompt, args.max_tokens, level, args.trials)
            rows.append(row)
            print(f"  {row}", flush=True)
        report["results"][name] = rows  # type: ignore[index]

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
