"""Asset-free model fixtures.

The pattern mirrors ``refs/vllm/tests/renderers/test_inkling.py``: a tiny
tokenizer that encodes one character per token, dict-based adapter configs and
in-memory media, so frontend tests never need a downloaded checkpoint.
"""

from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path

from tokenizers import Regex, Tokenizer, models, pre_tokenizers

ASSETS = Path(__file__).resolve().parent / "assets"

INKLING_SPECIAL_TOKENS = (
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_image|>",
    "<|unused_200054|>",
    "<|content_invoke_tool_json|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
    "<|image_pad|>",
    "<|image_start|>",
    "<|image_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|AUDIO|>",
    "<|audio_bos|>",
    "<|audio_eos|>",
)

INKLING2_CONFIG = {
    "model_type": "inkling2_loop",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "text_config": {
        "model_type": "inkling2_loop_text",
        "model_max_length": 1048576,
        "hidden_size": 2304,
        "num_hidden_layers": 24,
        "vocab_size": 201024,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "n_routed_experts": 128,
        "num_experts_per_tok": 8,
        "intermediate_size": 1152,
        "local_layer_ids": [
            0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20,
            21, 22,
        ],
        "sliding_window_size": 1024,
        "n_layers_per_attn_res_block": 6,
        "d_rel": 16,
        "rel_extent": 1024,
        "sconv_kernel_size": 4,
        "loop_config": {
            "core_start": 6,
            "core_end": 18,
            "num_passes": 3,
            "gate_rank": 64,
        },
    },
}

OXR_CONFIG = {
    "model_type": "oxr",
    "architectures": ["OXR"],
    "hidden_size": 1024,
    "vocab_size": 151936,
    "visual_vocab_size": 65536,
    "llm_config": {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": True,
    },
    "vit_config": {
        "patch_size": 16,
        "hidden_stride": 2,
        "temporal_patch_size": 1,
        "num_channels": 3,
        "image_size": 512,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "layer_norm_eps": 1e-6,
        "hidden_act": "gelu_pytorch_tanh",
        "window_size": 112,
        "use_rope": True,
        "preserve_original_pe": True,
        "num_patches": 1024,
    },
}


def write_json(root: Path, name: str, payload: object) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_tiny_tokenizer(root: Path, *, vocab: dict[str, int] | None = None) -> Path:
    """Write a self-contained ``tokenizer.json`` into *root* and return it."""

    root.mkdir(parents=True, exist_ok=True)
    if vocab is None:
        vocab = {"<unk>": 0, **{chr(code): code for code in range(32, 127)}}
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern=Regex("(.)"), behavior="isolated"
        )
        tokenizer.add_special_tokens(list(INKLING_SPECIAL_TOKENS))
    else:
        tokenizer = Tokenizer(
            models.WordLevel(vocab=vocab, unk_token="<|endoftext|>")
        )
        tokenizer.add_special_tokens(
            [
                token
                for token in vocab
                if token.startswith("<|") and token != "<|endoftext|>"
            ]
        )
    path = root / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def write_tiny_model(
    root: Path,
    *,
    chat_template: str | None = None,
    eos_token: str = "<|content_model_end_sampling|>",
    extra_config: dict | None = None,
) -> Path:
    """Build a minimal model directory accepted by ``ModelTokenizer``."""

    write_tiny_tokenizer(root)
    template = (
        chat_template
        if chat_template is not None
        else (ASSETS / "inkling_chat_template.jinja").read_text(encoding="utf-8")
    )
    (root / "chat_template.jinja").write_text(template, encoding="utf-8")
    write_json(root, "tokenizer_config.json", {"eos_token": eos_token, "pad_token": None})
    config = {"model_type": "inkling2_loop", "eos_token_id": []}
    if extra_config:
        config.update(extra_config)
    write_json(root, "config.json", config)
    return root


def write_tiny_shard(root: Path) -> Path:
    """Write one tiny safetensors shard (header-only inspection is enough)."""

    import tensorplay
    from tensorplay.serialization import save

    root.mkdir(parents=True, exist_ok=True)
    path = root / "model.safetensors"
    save(
        {
            "model.llm.attn_res.attn_queries.0": tensorplay.ones(2, 2),
            "model.llm.attn_res.attn_keys.0": tensorplay.ones(2, 3),
        },
        str(path),
    )
    return path


def wav_bytes(*, frames: int = 8800, rate: int = 16000) -> bytes:
    """PCM s16le mono WAV; 8800 frames at 16 kHz is the 0.55 s test clip."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", 0) for _ in range(frames)))
    return buffer.getvalue()
