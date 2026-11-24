"""Measure the OXR vLLM service with the same workload as tests/bench_oxr.py.

Start the official service first (from the OXR checkout)::

    cd "$OXR_CHECKOUT" && bash start_all.sh

Or just the model server::

    MODEL_PATH="$MEGALM_OXR_MODEL" \
      python services/oxr/run_vllm_local.py serve \
      "$MEGALM_OXR_MODEL" \
      --trust-remote-code --served-model-name oxr \
      --chat-template "$MEGALM_OXR_MODEL/chat_template.jinja" \
      --dtype bfloat16 --max-model-len 16384 \
      --attention-config '{"backend":"TRITON_ATTN"}' \
      --gpu-memory-utilization 0.70 --max-num-seqs 16 \
      --host 127.0.0.1 --port 8080

Then::

    python tests/bench_oxr_vllm.py --tokens 32
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


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
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="oxr")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/megalm-oxr-vllm-bench.json")
    )
    args = parser.parse_args()

    data_url = _make_case()
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "What is written in this document?"},
                ],
            }
        ],
        "max_tokens": args.tokens,
        "min_tokens": args.tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    def run() -> dict[str, float]:
        request = urllib.request.Request(
            args.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        first = None
        completion_tokens = None
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
                    completion_tokens = chunk["usage"].get("completion_tokens")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content and first is None:
                    first = time.perf_counter()
        ended = time.perf_counter()
        return {
            "ttft_s": (first or ended) - started,
            "total_s": ended - started,
            "completion_tokens": completion_tokens or args.tokens,
        }

    run()
    samples = [run() for _ in range(3)]
    ttft = statistics.median(item["ttft_s"] for item in samples)
    total = statistics.median(item["total_s"] for item in samples)
    tokens = statistics.median(item["completion_tokens"] for item in samples)
    result = {
        "tokens": args.tokens,
        "vllm": {
            "ttft_ms": ttft * 1000,
            "total_s": total,
            "decode_tok_s": (tokens - 1) / (total - ttft),
        },
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
