"""Comprehensive MegaLM OXR service-scenario benchmark matrix.

Measures single-stream decode at several token counts and aggregate throughput
under 1/2/4/8 concurrent requests. The counterpart for vLLM is
``bench_oxr_service_matrix_vllm.py``; both emit the same JSON shape.

Usage::

    PYTHONPATH=src:../tensorplay-main python tests/bench_oxr_service_matrix.py \
        --model /home/bluemoon/projects/OXR/models/OXR-1.0 \
        --output /tmp/megalm-oxr-native-service.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tensorplay-main"))

TOKEN_COUNTS = (16, 32, 64, 128)
CONCURRENCY = (1, 2, 4, 8)


def _make_case():
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
    return image, buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/bluemoon/projects/OXR/models/OXR-1.0")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/megalm-oxr-native-service.json")
    )
    args = parser.parse_args()

    import tensorplay as tp
    from megalm.contracts.request import ImageInput, SamplingParams
    from megalm.runtime.engine import Engine, EngineConfig

    image, raw = _make_case()
    image_input = ImageInput(
        content_digest=hashlib.sha256(raw).hexdigest(),
        mime_type="image/png",
        payload=raw,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": "What is written in this document?"},
            ],
        }
    ]
    cfg = EngineConfig(
        model_path=args.model,
        device="cuda",
        dtype="bfloat16",
        max_model_len=2048,
        max_cache_tokens=4096,
        max_num_sequences=8,
        max_num_batched_tokens=2048,
        prefill_chunk_size=1024,
        enable_cuda_graph=True,
        cuda_graph_batch_sizes=(1, 2, 4, 8),
        use_triton=True,
        gpu_memory_utilization=args.gpu_util,
        vision_cache_bytes=256 * 1024**2,
    )
    engine = Engine(cfg)

    def generate(tokens: int, nreq: int) -> float:
        tp.cuda.synchronize()
        started = time.perf_counter()
        request_ids = [
            engine.add_multimodal_chat(
                messages,
                images=(image_input,),
                max_new_tokens=tokens,
                sampling=SamplingParams(temperature=0, ignore_eos=True),
            )
            for _ in range(nreq)
        ]
        engine.run_until_complete(tuple(request_ids))
        tp.cuda.synchronize()
        return time.perf_counter() - started

    generate(2, 1)  # warmup

    results: dict[str, dict] = {"single": {}, "concurrent": {}}
    ttft = statistics.median([generate(1, 1) for _ in range(5)])
    for count in TOKEN_COUNTS:
        total = statistics.median([generate(count, 1) for _ in range(3)])
        results["single"][count] = {
            "ttft_ms": ttft * 1000,
            "total_s": total,
            "decode_tok_s": (count - 1) / (total - ttft),
        }
    for n in CONCURRENCY:
        total = statistics.median([generate(32, n) for _ in range(3)])
        results["concurrent"][n] = {
            "total_s": total,
            "aggregate_tok_s": n * 32 / total,
        }

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
