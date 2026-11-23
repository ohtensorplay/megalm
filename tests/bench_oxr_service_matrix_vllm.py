"""vLLM counterpart of ``bench_oxr_service_matrix.py``.

Requires a running OXR vLLM service (see ``bench_oxr_vllm.py`` for startup).
Usage::

    python tests/bench_oxr_service_matrix_vllm.py \
        --output /tmp/megalm-oxr-vllm-service.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

TOKEN_COUNTS = (16, 32, 64, 128)
CONCURRENCY = (1, 2, 4, 8)


def _make_case() -> str:
    image = Image.new("RGB", (320, 480), "white")
    draw = ImageDraw.Draw(image)
    for index in range(20):
        draw.text(
            (10, 10 + index * 20),
            f"Line {index}: hello OXR 1234",
            fill="black",
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="oxr")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/megalm-oxr-vllm-service.json")
    )
    args = parser.parse_args()

    data_url = _make_case()

    def payload(tokens: int) -> dict:
        return {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {
                            "type": "text",
                            "text": "What is written in this document?",
                        },
                    ],
                }
            ],
            "max_tokens": tokens,
            "min_tokens": tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    def one(tokens: int) -> tuple[float, float, int]:
        request = urllib.request.Request(
            args.url,
            data=json.dumps(payload(tokens)).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        first = None
        completion = None
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    completion = chunk["usage"].get("completion_tokens")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content and first is None:
                    first = time.perf_counter()
        ended = time.perf_counter()
        return (first or ended) - started, ended - started, completion or tokens

    one(2)  # warmup

    results: dict[str, dict] = {"single": {}, "concurrent": {}}
    ttft = statistics.median([one(1)[0] for _ in range(5)])
    for count in TOKEN_COUNTS:
        samples = [one(count) for _ in range(3)]
        total = statistics.median(item[1] for item in samples)
        results["single"][count] = {
            "ttft_ms": ttft * 1000,
            "total_s": total,
            "decode_tok_s": (count - 1) / (total - ttft),
        }
    for n in CONCURRENCY:
        def batch() -> float:
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                list(pool.map(lambda _: one(32), range(n)))
            return time.perf_counter() - started

        total = statistics.median([batch() for _ in range(3)])
        results["concurrent"][n] = {
            "total_s": total,
            "aggregate_tok_s": n * 32 / total,
        }

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
