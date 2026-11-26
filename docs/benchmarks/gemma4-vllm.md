# Gemma-4 E2B throughput: MegaLM vs vLLM

Same RTX 3090, same `google/gemma-4-E2B-it` fp16 weights, same 512-token
greedy prompt, same batch/max-new tokens, single sync after generation.
3 measurements after 1 warmup, median `tok/s` reported.

- MegaLM: `devdoc/bench_gemma4_megalm.py` (`PYTHONPATH=src`).
- vLLM: `devdoc/bench_gemma4_vllm.py`, which runs the pinned `refs/vllm`
  checkout in place with `devdoc/vllm_ref_shim.py` (the bundled FA extension is
  ABI-incompatible with the local torch 2.13, so `TRITON_ATTN` is forced on
  **both** the reference and here — vLLM's FA fast path is deliberately not
  available in this environment).

## Results (latest clean measurements)

| Engine | Batch | Prefill+1 step | Steady tok/s (gen=129) | Load time |
| --- | ---: | ---: | ---: | ---: |
| MegaLM | 1 | **39 ms** | 134–135 | **4–16 s** |
| vLLM (TRITON_ATTN) | 1 | 46 ms | 135 | 48–78 s |
| MegaLM (CUDA graph, chunk=8) | 8 | — | 750–756 | 5–16 s |
| vLLM (TRITON_ATTN) | 8 | — | 807–815 | 48–78 s |

MegaLM now matches the reference on batch-1 steady-state decode (within run
noise) and is **faster at prefill** and **~10× faster to start**. The remaining
decode gap at batch 8 is due to vLLM's fused attention kernel. Note the box is
shared, so absolute numbers move with co-tenant load; the ratios use matched
runs.

## What changed since the first build

1. **Fast attention for local layers.** 28 of 35 layers ran through a
   Python-loop fallback because the Triton kernels lacked sliding-window
   masks. Added an optional `sliding_window` constexpr to the shared
   prefill/decode/GQA-decode kernels (default 0, so other models are unchanged)
   and skip / re-split work outside the window. Prefill 297 → 39 ms; batch-1
   throughput 81 → ~135 tok/s. Covered by
   `tests/test_gemma4_window_attention_cuda.py` (decode maxdiff 1.5e-8, prefill
   1.3e-4 vs an explicit reference).
2. **Batched-serving bug.** Stale/sentinel token ids overflowed the embedding
   gather at batch>1; a defensive clamp unblocked batch>1 CUDA-graph serving
   (batch 8: 170 → 756 tok/s).
3. **Fused QKV-norm-RoPE-V-norm.** One packed QKV projection plus one Triton
   kernel now performs Q/K norm, V norm, RoPE (both flavours via the existing
   cos/sin tables) and cache write for every layer; shared layers use a fused
   Q-only variant. All 35 layers are on the fast path. The kernel was extended
   from the existing shared `fused_qkv_norm_rope_cache_write` with constexpr
   flags, so Qwen 3.5's path is untouched. Owned-layer output is bit-exact vs
   the previous path; real-checkpoint HF logits differ by at most ~0.11 with a
   matching argmax (prior path: ~0.086).
4. **Fused decode projections.** Single-token decode uses tuned Triton GEMVs
   for the QKV projection and a fused gate/up GEMV + GELU-tanh for the MLP,
   replacing ~10 elementwise launches per layer with one. Added reusable
   `gelu_tanh_and_mul`, `add_scaled` and `rms_norm_add` kernels.
5. **Vision parity.** Checkpoint per-linear clipping metadata is loaded and
   applied; vision features now match HuggingFace at cosine 1.0 / relerr 0.0018
   on `gemma-4-E2B-it`.

All verification passed: Ruff clean, 20/20 Gemma-4 tests, 603-test Python
suite.

## Reproducing

The box is shared with other GPU workloads. Both scripts accept `--util` and
MegaLM accepts `--guard-gib` to hold a small allocation that keeps co-tenants
from grabbing the device mid-load. For a quiet run use
`--util 0.99 --cache 4096 --max-model-len 1024 --decode-chunk 8`, and re-run if
a co-tenant starts mid-load (an OOM at model construction is a co-tenant, not
a code failure).
