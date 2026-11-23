"""Measure native MegaLM OXR with the same workload as ``bench_oxr_vllm.py``.

Run with::

    PYTHONPATH=src:../tensorplay-main python tests/bench_oxr_native.py --tokens 32

This produces the same JSON shape as ``bench_oxr_vllm.py`` so the two numbers
can be compared directly. The two scripts are kept separate because vLLM runs
in its own conda environment and process.
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
    raw = buffer.getvalue()
    return image, raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument(
        "--model", default="/home/bluemoon/projects/OXR/models/OXR-1.0"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/megalm-oxr-native-bench.json")
    )
    parser.add_argument("--gpu-util", type=float, default=0.85)
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
    cfg = EngineConfig(
        model_path=args.model,
        device="cuda",
        dtype="bfloat16",
        max_model_len=2048,
        max_cache_tokens=2048,
        max_num_sequences=1,
        max_num_batched_tokens=1024,
        prefill_chunk_size=1024,
        enable_cuda_graph=True,
        cuda_graph_batch_sizes=(1,),
        use_triton=True,
        gpu_memory_utilization=args.gpu_util,
        vision_cache_bytes=128 * 1024**2,
    )
    engine = Engine(cfg)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is written in this document?"},
            ],
        }
    ]

    def generate(tokens: int) -> float:
        tp.cuda.synchronize()
        started = time.perf_counter()
        request_id = engine.add_multimodal_chat(
            messages,
            images=(image_input,),
            max_new_tokens=tokens,
            sampling=SamplingParams(temperature=0, ignore_eos=True),
        )
        engine.run_until_complete((request_id,))
        tp.cuda.synchronize()
        return time.perf_counter() - started

    generate(2)  # warmup
    firsts = [generate(1) for _ in range(3)]
    totals = [generate(args.tokens) for _ in range(3)]
    ttft = statistics.median(firsts)
    total = statistics.median(totals)
    result = {
        "tokens": args.tokens,
        "native": {
            "ttft_ms": ttft * 1000,
            "total_s": total,
            "decode_tok_s": (args.tokens - 1) / (total - ttft),
        },
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
