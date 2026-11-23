"""Local OXR speed comparison: native MegaLM vs the official transformers path.

Usage::

    PYTHONPATH=src:../tensorplay-main python tests/bench_oxr.py \
        --oxr-model /home/bluemoon/projects/OXR/models/OXR-1.0 \
        --tokens 64

The script builds one deterministic synthetic document, warms both runtimes,
then reports first-token latency and decode-only throughput.  The official
reference is imported from the checkpoint directory through a symlinked
package so its remote-code module keeps working with modern transformers.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def _reference_package(model_dir: Path, root: Path) -> Path:
    package = root / "oxr_model"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for source in model_dir.iterdir():
        target = package / source.name
        if not target.exists():
            target.symlink_to(source)
    return root


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
    payload = buffer.getvalue()
    return image, payload


def _measure(fn, tokens: int) -> dict[str, float]:
    fn(2)
    first = [fn(1) for _ in range(3)]
    full = [fn(tokens) for _ in range(3)]
    ttft = float(statistics.median(first))
    total = float(statistics.median(full))
    decode = (tokens - 1) / max(total - ttft, 1e-9)
    return {
        "ttft_ms": ttft * 1000,
        "total_s": total,
        "decode_tok_s": decode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oxr-model",
        type=Path,
        default=Path("/home/bluemoon/projects/OXR/models/OXR-1.0"),
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("/tmp/megalm-oxr-bench.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    tensorplay_root = Path(__file__).resolve().parents[2] / "tensorplay-main"
    sys.path.insert(0, str(tensorplay_root))

    import tensorplay as tp

    from megalm.contracts.request import ImageInput, SamplingParams
    from megalm.runtime.engine import Engine, EngineConfig

    image, payload = _make_case()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": "What is written in this document?"},
            ],
        }
    ]
    image_input = ImageInput(
        content_digest=hashlib.sha256(payload).hexdigest(),
        mime_type="image/png",
        payload=payload,
    )

    config = EngineConfig(
        model_path=args.oxr_model,
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
        gpu_memory_utilization=0.85,
        vision_cache_bytes=128 * 1024**2,
    )
    engine = Engine(config)

    def run_native(tokens: int) -> float:
        tp.cuda.synchronize()
        start = time.perf_counter()
        request_id = engine.add_multimodal_chat(
            messages,
            images=(image_input,),
            max_new_tokens=tokens,
            sampling=SamplingParams(temperature=0, ignore_eos=True),
        )
        engine.run_until_complete((request_id,))
        tp.cuda.synchronize()
        return time.perf_counter() - start

    native = _measure(run_native, args.tokens)
    engine = None  # release the GPU arena before loading the reference model
    gc.collect()
    tp.cuda.empty_cache()

    with tempfile.TemporaryDirectory() as tmp:
        _reference_package(args.oxr_model, Path(tmp))
        sys.path.insert(0, tmp)
        from oxr_model.modeling_oxr import OXR

        ref_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is written in this document?"},
                ],
            }
        ]
        reference = OXR.from_pretrained(
            str(args.oxr_model), torch_dtype=torch.bfloat16
        ).cuda().eval()
        # transformers 5.19 mis-materializes this non-persistent buffer.
        reference.indicator_token_indices = torch.arange(
            reference.config.visual_vocab_size - 4,
            reference.config.visual_vocab_size,
            dtype=torch.long,
            device="cuda",
        )
        input_ids, pixel_values, grid_thws = reference.preprocess_inputs(
            ref_messages
        )
        ids = input_ids.cuda()
        pixels = pixel_values.cuda()
        grids = grid_thws.cuda()

        def run_reference(tokens: int) -> float:
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                reference.generate(
                    inputs=ids,
                    pixel_values=pixels,
                    grid_thws=grids,
                    max_new_tokens=tokens,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            return time.perf_counter() - start

        official = _measure(run_reference, args.tokens)

    result = {
        "tokens": args.tokens,
        "native": native,
        "official": official,
        "decode_speedup": native["decode_tok_s"] / official["decode_tok_s"],
        "e2e_speedup": official["total_s"] / native["total_s"],
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
