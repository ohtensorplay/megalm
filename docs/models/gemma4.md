# Gemma 4 in megaLM

This document describes the first-party Gemma 4 executor in
`src/megalm/models/gemma4/`. It is a native TensorPlay implementation; no
PyTorch/Transformers model is invoked in the serving path.

## Supported checkpoint families

| Family | `model_type` | Status |
| --- | --- | --- |
| Text / multimodal conditional | `gemma4`, `gemma4_text` | Native decoder, YOCO KV sharing, PLE, MoE |
| Unified (encoder-free) | `gemma4_unified`, `gemma4_unified_text` | Native decoder + direct media embedders |
| Assistant (MTP) | `gemma4_assistant` | Native Q-only draft predictor |

## Text decoder

- Scaled word embeddings (`sqrt(hidden_size)`) and per-layer embeddings (PLE).
- Sandwich RMSNorm with post-norm residual adds (Gemma convention, *not* the
  Qwen-style fused add-then-norm kernel).
- Heterogeneous per-layer geometry: sliding layers use `head_dim`, full
  attention layers use `global_head_dim`. The paged arena is allocated at the
  largest geometry and each layer takes a narrowed view.
- Dual RoPE: standard for sliding layers and proportional / partial for full
  layers. Non-owning (YOCO) layers rotate Q only and read K/V from their owner.
- QK norm with unit attention scale; final logit soft-capping.
- Optional dense + sparse MoE branch: softmax routing over all experts,
  top-k renormalisation, per-expert scale, added to the dense MLP output.

Numerical parity is covered by `tests/test_gemma4.py` against
`transformers.Gemma4ForCausalLM` on a small synthetic checkpoint, for both the
dense and MoE paths (max logit diff < 1e-4).

## Multi-token prediction

`Gemma4MultiTokenPredictor` is a Q-only draft decoder that reads K/V from the
target model's paged cache. Each assistant layer is bound to the last
non-shared target layer of the same attention type. The shared engine contract
(graph-stable hidden buffer, `compute_logits`, optional centroid-masked logits)
lives in `src/megalm/models/components/mtp.py` and is also used by Qwen 3.5.

To serve with an assistant checkpoint, pass `--speculative-mode mtp` and
`--speculative-model-path <assistant-checkpoint>`.

## Multimodal

- Tower-based: `Gemma4VisionModel` (patch projection with 2-D position tables,
  QK-normalised bidirectional attention with axial RoPE, position-based
  pooling) and `Gemma4AudioModel` (strided Conv2d subsampling, relative
  sinusoidal position encoding, blocked local attention with soft-cap,
  feed-forward / depthwise-conv blocks).
- Encoder-free: `Gemma4UnifiedVisionEmbedder` and
  `Gemma4UnifiedAudioEmbedder` project patches / waveform frames directly into
  the text embedding space.
- The `MultimodalEmbedder` connector is shared between vision and audio.
- Token-count arithmetic for prompt replacement follows the reference: image
  soft tokens are capped at `max_soft_tokens`, and audio length is derived from
  mel framing and two stride-2 subsampling stages.

End-to-end admission is wired: `Gemma4ImageProcessor` and
`Gemma4AudioProcessor` are Transformers-free frontends whose mel and patch
geometry match the reference processors, the engine selects them by model type,
and the tokenizer expands Gemma-4's bare `<|image|>` / `<|audio|>` placeholders
into `<|image>` / `<image|>` and `<|audio>` / `<audio|>` boundaries with the
computed number of soft tokens. `build_media_admission` uses plain 1-D RoPE
positions for Gemma-4 instead of Qwen-style mRoPE, and the model exposes
`encode_images` / `encode_audios` so cached tower output is scattered into the
prompt rows. Tower weights load from the real `Gemma4ForConditionalGeneration`
checkpoint. Vision features match HuggingFace on a real checkpoint: the native
pooler uses the reference k×k spatial clusters and the checkpoint's per-linear
input/output clipping metadata is loaded and applied (cosine 1.0, relative
error ~0.0018 on `gemma-4-E2B-it`).

## Reasoning and tool calls

- The tokenizer reports the `gemma4` output grammar, so the gateway preserves
  special tokens and keeps `<|channel>` / `<channel|>` for the Rust stream
  parser. The `thought\n` prefix is stripped from streamed reasoning.
- Tool calls use the native `<|tool_call>call:name{...}<tool_call|>` grammar
  with `<|"|>` string delimiters; the Rust `gemma4` tool parser converts the
  custom arguments to JSON and applies the tool schema's type coercion.
- Chat templates support `enable_thinking` and `preserve_thinking`. A prompt
  ending inside an open channel is treated as continuing reasoning.

## Validation

The `google/gemma-4-E2B-it` checkpoint (35 layers, 15 KV owners, 256/512 head
dims, PLE, final soft-cap 30) loads through `load_weights` and produces finite
logits on CPU. GPU/Triton paths require a CUDA-enabled TensorPlay build.
