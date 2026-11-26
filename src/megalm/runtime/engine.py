from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import logging
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import tensorplay
import triton

from megalm.adapters import load_model_spec
from megalm.contracts.lora import (
    GENERIC_LORA_MODEL_TYPES,
    LoRAConfig,
    LoRARequest,
)
from megalm.contracts.model import AttentionKind
from megalm.contracts.plan import (
    CompiledPlan,
    HardwareSpec,
    TopologySpec,
    WorkloadProfile,
)
from megalm.runtime.collectives import (
    CollectiveGroup,
    DistributedCollectiveGroup,
    NOOP_COLLECTIVE,
    TPGroupConfig,
)
from megalm.contracts.request import (
    AudioChunk,
    AudioInput,
    AudioOutputOptions,
    EngineError as ContractEngineError,
    EngineEvent,
    EventKind,
    FinishReason,
    ImageInput,
    InferenceRequest,
    OutputModality,
    RequestState,
    SamplingParams,
    VideoInput,
)
from megalm.kernels.attention import (
    AttentionConfig,
    AttentionConfigError,
)
from megalm.kernels.bindings import (
    bind_kernel_plan,
    discover_kernel_plan,
)
from megalm.kernels.grammar_mask import apply_grammar_bitmask
from megalm.runtime.structured import GrammarManager, resolve_structured_output
from megalm.kernels.ops import KernelPolicy
from megalm.loading import (
    KV_SCALE_SUFFIXES,
    SafetensorsSource,
    TensorInfo,
    read_kv_scale_vectors,
)
from megalm.models import load_native_model
from megalm.models.qwen2_5_omni import (
    Qwen25OmniAudioProcessor,
    Qwen25OmniImageProcessor,
    Qwen25OmniVideoProcessor,
)
from megalm.models.qwen3_5 import (
    Qwen35ImageProcessor,
    Qwen35VideoProcessor,
)
from megalm.models.oxr import OXRImageProcessor
from megalm.models.gemma4.processors import (
    Gemma4AudioProcessor,
    Gemma4ImageProcessor,
)
from megalm.observability import log_event
from megalm.planning import build_execution_plan
from megalm.providers.registry import (
    DEFAULT_REGISTRY as HARDWARE_PROVIDERS,
    ProviderEvidence,
)
from megalm.runtime.cache import KVCacheManager, PagedKVCache, PrefixStateStore
from megalm.runtime.hybrid_cache import GDNStateCache, HybridCache
from megalm.runtime.kv_offload import (
    DiskKVTier,
    HostKVTier,
    KVOffloadError,
    TensorCopier,
    TieredKVOffload,
)
from megalm.runtime.kv_quant import KVQuantConfig, KVQuantMode
from megalm.runtime.lora import LoRAError, LoRAManager
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.compile_backend import (
    CompileBackendError,
    CompilePolicy,
    CompileReport,
    compile_model,
    compiled_identity_digest,
)
from megalm.runtime.audio_pipeline import AudioPipeline, ResolvedAudios
from megalm.runtime.graphs import (
    CUDAGraphError,
    DecodeGraphRunner,
    MTPUpdateGraphRunner,
)
from megalm.runtime.media_admission import (
    MediaState,
    MultimodalAdmission,
    build_media_admission,
)
from megalm.runtime.omni_output import (
    OmniOutputError,
    OmniSpeechStage,
    OmniSpeechTask,
    TransformersOmniSpeechBackend,
)
from megalm.runtime.sampler import Sampler
from megalm.runtime.speculative import DraftSource, build_draft_source
from megalm.runtime.scheduler import (
    Completion,
    ContinuousBatchScheduler,
    ScheduledBatch,
    SchedulerError,
    SchedulerLoRAError,
)
from megalm.runtime.tokenizer import ModelTokenizer
from megalm.runtime.vision_cache import (
    GPUVisionCache,
)
from megalm.runtime.vision_pipeline import (
    ResolvedImages,
    ResolvedVideos,
    VisionPipeline,
)
from megalm.startup import (
    StartupPhase,
    StartupTracker,
    build_plan_draft,
    estimate_gdn_state_bytes,
    estimate_kv_page_bytes,
    kv_cache_layer_count,
)

_LOGGER = logging.getLogger("megalm.engine")


def _merge_completions(completions: list[Completion]) -> Completion:
    """Splice per-token completions of one decode chunk into a single step."""

    if len(completions) == 1:
        return completions[0]
    events: tuple[EngineEvent, ...] = ()
    finished: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()
    tokens: tuple[int, ...] = ()
    reasons: tuple[FinishReason | None, ...] = ()
    logprobs: tuple[float | None, ...] = ()
    top_logprobs: tuple[tuple[tuple[int, float], ...], ...] = ()
    label_ids: tuple[tuple[int, ...], ...] = ()
    label_logprobs: tuple[tuple[float, ...], ...] = ()
    for completion in completions:
        events += completion.events
        finished += completion.finished_request_ids
        request_ids += completion.output_request_ids
        tokens += completion.output_token_ids
        reasons += completion.finish_reasons
        logprobs += completion.output_logprobs
        top_logprobs += completion.output_top_logprobs
        label_ids += completion.output_label_ids
        label_logprobs += completion.output_label_logprobs
    return Completion(
        events=events,
        finished_request_ids=finished,
        output_request_ids=request_ids,
        output_token_ids=tokens,
        finish_reasons=reasons,
        output_logprobs=logprobs,
        output_top_logprobs=top_logprobs,
        output_label_ids=label_ids,
        output_label_logprobs=label_logprobs,
    )


class EngineError(RuntimeError):
    """Raised when the embedded Replica cannot make safe forward progress."""


class EngineLoRAError(EngineError):
    """An ``EngineError`` that preserves the LoRA client-facing code."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _checkpoint_source_dtype(manifest: Sequence[TensorInfo]) -> str:
    """Resolve the storage dtype without rejecting Inkling-2's mixed precision."""

    dtypes = {tensor.dtype for tensor in manifest}
    # Inkling-2 keeps experts as NVFP4 (U8) or MXFP8 (F8_E4M3); the base
    # dtype is still BF16.  Qwen3.5 adds FP32 state vectors.
    unsupported = dtypes - {
        "F16",
        "BF16",
        "F32",
        "F8_E4M3",
        "U8",
        "F8_E8M0",
        "I32",
        "I64",
    }
    if unsupported:
        raise EngineError(
            f"checkpoint source dtypes are not supported: {sorted(unsupported)}"
        )
    invalid_integer = tuple(
        tensor.name
        for tensor in manifest
        if (tensor.dtype == "I32" and not tensor.name.endswith(".weight_packed"))
        or (tensor.dtype == "I64" and not tensor.name.endswith(".weight_shape"))
    )
    if invalid_integer:
        raise EngineError(
            "checkpoint contains unexpected integer model tensors: "
            f"{invalid_integer[:4]}"
        )
    low_precision = dtypes & {"F16", "BF16"}
    if len(low_precision) > 1:
        raise EngineError("checkpoint mixes float16 and bfloat16 storage")
    if "F32" in dtypes and low_precision:
        invalid_fp32 = tuple(
            tensor.name
            for tensor in manifest
            if tensor.dtype == "F32"
            and not tensor.name.endswith((".A_log", ".dt_bias"))
            and not tensor.name.endswith(".linear_attn.norm.weight")
            and not tensor.name.startswith("mtp.")
            and ".attn_res." not in tensor.name
            and not tensor.name.endswith((".gate.bias", ".gate.weight"))
            and not tensor.name.endswith("middle_cycle_cross_pass_queries")
            and not tensor.name.endswith("middle_cycle_cross_pass_kv_queries")
            and not tensor.name.endswith(("_amax", "_outer_scale_denominator"))
        )
        if invalid_fp32:
            raise EngineError(
                f"checkpoint contains unexpected FP32 model tensors: {invalid_fp32[:4]}"
            )
    selected = next(iter(low_precision), "F32")
    return {
        "F16": "float16",
        "BF16": "bfloat16",
        "F32": "float32",
    }[selected]


_KV_QUANT_DTYPES = {
    "fp8_e4m3fn": (tensorplay.float8_e4m3fn, KVQuantMode.FP8_PER_TENSOR),
    "fp8_e5m2": (tensorplay.float8_e5m2, KVQuantMode.FP8_PER_TENSOR),
    "fp8_per_head": (tensorplay.float8_e5m2, KVQuantMode.FP8_PER_HEAD),
    "fp8_per_token_head": (
        tensorplay.float8_e5m2,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
    ),
    "int4_per_token_head": (
        tensorplay.uint8,
        KVQuantMode.INT4_PER_TOKEN_HEAD,
    ),
    "nvfp4": (tensorplay.uint8, KVQuantMode.NVFP4),
}


def _gpu_capability() -> tuple[int, int]:
    try:
        caps = triton.runtime.driver.active.get_current_target().capability
        return (int(caps[0]), int(caps[1]))
    except Exception:  # noqa: BLE001 - capability probe is best effort
        return (0, 0)


def _resolve_kv_quant(
    config: EngineConfig,
    num_kv_heads: int,
    *,
    checkpoint_scales: dict[str, tuple[float, ...]] | None = None,
) -> KVQuantConfig | None:
    """Resolve the configured KV dtype and guard unsupported hardware.

    Calibrated per-head scales from a checkpoint take precedence over CLI
    scalars for ``fp8_per_head``.
    """

    if config.kv_cache_dtype == "auto":
        if config.kv_cache_q_dtype != "auto" or config.native_fp8_attn:
            raise EngineError("Q quantization requires a quantized kv_cache_dtype")
        return None
    dtype, mode = _KV_QUANT_DTYPES[config.kv_cache_dtype]
    capability = _gpu_capability() if config.device == "cuda" else (0, 0)
    if config.device == "cuda" and dtype == tensorplay.float8_e4m3fn:
        if capability[0] < 89:
            raise EngineError(
                "kv_cache_dtype=fp8_e4m3fn requires SM89+ (Ada/Hopper); "
                "use fp8_e5m2 on this GPU"
            )
    # Q quantization only makes sense with a static scale and e4m3 dot.
    q_dtype = None
    if config.kv_cache_q_dtype != "auto":
        q_dtype = {
            "fp8_e4m3fn": tensorplay.float8_e4m3fn,
            "fp8_e5m2": tensorplay.float8_e5m2,
        }[config.kv_cache_q_dtype]
    native = config.native_fp8_attn
    if config.int4_use_rht and mode != KVQuantMode.INT4_PER_TOKEN_HEAD:
        raise EngineError("int4_use_rht requires kv_cache_dtype=int4_per_token_head")
    if native:
        if capability[0] < 89:
            raise EngineError(
                "native_fp8_attn requires SM89+ (Ada/Hopper); on SM86 the "
                "dequantizing fp16 attention path is used instead"
            )
        if mode.is_dynamic:
            raise EngineError("native fp8 attention requires static KV scales")
        if q_dtype is None:
            q_dtype = tensorplay.float8_e4m3fn
        if q_dtype != tensorplay.float8_e4m3fn:
            raise EngineError("native fp8 attention requires fp8_e4m3fn Q")
    q_scale: float | tuple[float, ...] = config.kv_cache_q_scale or 1.0
    if config.kv_cache_q_group == "per_kv_head":
        if isinstance(q_scale, tuple):
            pass
        elif config.kv_cache_q_scale is not None:
            q_scale = (float(q_scale),) * num_kv_heads
        else:
            q_scale = (1.0,) * num_kv_heads
    key_scale: float | tuple[float, ...] = config.kv_cache_key_scale or 1.0
    value_scale: float | tuple[float, ...] = config.kv_cache_value_scale or 1.0
    if mode == KVQuantMode.FP8_PER_HEAD:
        if isinstance(key_scale, tuple) or isinstance(value_scale, tuple):
            # already vectors
            pass
        else:
            key_scale = (float(key_scale),) * num_kv_heads
            value_scale = (float(value_scale),) * num_kv_heads
        if checkpoint_scales:
            key_scale = checkpoint_scales.get("key_scale", key_scale)
            value_scale = checkpoint_scales.get("value_scale", value_scale)
            if config.kv_cache_q_group == "per_kv_head":
                q_scale = checkpoint_scales.get("q_scale", q_scale)
    return KVQuantConfig(
        dtype=dtype,
        key_scale=key_scale,
        value_scale=value_scale,
        mode=mode,
        num_kv_heads=num_kv_heads,
        q_dtype=q_dtype,
        q_scale=q_scale,
        q_group=config.kv_cache_q_group,
        native_fp8_dot=native,
        nvfp4_global_scale=config.nvfp4_global_scale,
        int4_use_rht=config.int4_use_rht,
    )


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model_path: Path
    device: str = "cuda"
    dtype: str = "bfloat16"
    # "generate" keeps the autoregressive scheduler. "embed" runs the native
    # encoder/pooling path and exposes only the embeddings data plane.
    task: str = "generate"
    embedding_max_batch_size: int = 64
    embedding_max_sequence_length: int | None = None
    embedding_coalesce_wait_us: int = 1_000
    # Tensor parallelism. TP=1 keeps the original single-process fast path;
    # TP>1 is a node-local process group with one rank per GPU.
    tensor_parallel: int = 1
    tp_rank: int = 0
    tp_backend: str = "nccl"
    tp_init_method: str = "env://"
    # Attention backend is fixed during DISCOVER. "causal" keeps the existing
    # hand-written kernels; the others bind a typed reference backend cold.
    attention_mode: str = "causal"
    attention_sink_logits: tuple[float, ...] = ()
    attention_sparse_top_k: int = 0
    attention_sparse_block_size: int = 16
    attention_mla_kv_lora_rank: int = 0
    attention_mla_qk_rope_head_dim: int = 0
    attention_mla_qk_nope_head_dim: int = 0
    page_size: int = 16
    max_cache_tokens: int | None = None
    gpu_memory_utilization: float = 0.9
    min_runtime_memory_bytes: int = 512 * 1024**2
    max_model_len: int | None = 32_768
    max_num_sequences: int = 16
    # A prefill step only reaches the weight-bound GEMM's efficient regime once
    # it carries a few thousand rows, so allow whole prompt blocks per step
    # instead of dribbling them out 1024 tokens at a time.  Mixed steps keep a
    # smaller budget so running sequences are not starved of decode slots.
    max_num_batched_tokens: int = 8192
    prefill_chunk_size: int = 8192
    mixed_prefill_chunk_size: int = 1024
    cpu_intraop_threads: int = 16
    vision_max_encode_patches: int = 8_192
    omni_admission_coalesce_wait_us: int = 5_000
    omni_light_admission_batch_size: int = 2
    video_admission_max_defer_ms: int = 300
    video_admission_defer_min_target: int = 8
    enable_prefix_cache: bool = True
    kv_cache_dtype: str = "auto"
    kv_cache_key_scale: float | None = None
    kv_cache_value_scale: float | None = None
    kv_cache_q_dtype: str = "auto"
    kv_cache_q_scale: float | None = None
    kv_cache_q_group: str = "per_tensor"
    native_fp8_attn: bool = False
    nvfp4_global_scale: float = 1.0
    int4_use_rht: bool = False
    kv_host_cache_bytes: int = 0
    kv_disk_cache_dir: str | None = None
    kv_disk_cache_bytes: int = 0
    kv_disk_read_threads: int = 4
    kv_disk_write_threads: int = 2
    vision_cache_bytes: int = 256 * 1024**2
    use_triton: bool = True
    triton_moe_max_tokens: int = 64
    tensorcore_moe_min_tokens: int = 8
    grouped_moe_min_tokens: int = 8
    speculative_mode: str = "off"
    # Optional assistant checkpoint for checkpoint-native MTP (Gemma-4).
    speculative_model_path: str | None = None
    num_speculative_tokens: int = 7
    prompt_lookup_min_ngram: int = 2
    prompt_lookup_max_ngram: int = 4
    prompt_lookup_min_candidate_support: int = 3
    prompt_lookup_min_candidate_agreement: float = 0.0
    prompt_lookup_probe_interval: int = 4
    enable_cuda_graph: bool = True
    cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)
    strict_cuda_graph: bool = True
    # Chained decode replays a captured graph may run before the host commits.
    # 1 keeps the classic one-launch-one-sync step; larger values amortise the
    # host scheduler/launch cost over a device-side self-loop.
    decode_chunk_steps: int = 1
    # Stax is an opt-in cold-path lowering provider for static text regions.
    # "none" keeps the hand-written kernels; "stax" admits a region only after
    # native lowering, parity and CUDA Graph capture all succeed.
    compile_backend: str = "none"
    compile_regions: tuple[str, ...] = ("dense_mlp",)
    compile_tolerance: float | None = None
    compile_min_speedup: float = 1.05
    compile_buckets: tuple[int, ...] = ()
    compile_min_tokens: int = 1024
    compile_max_artifact_bytes: int = 2 * 1024**3
    compile_startup_budget_seconds: float = 0.0
    enable_omni_audio_output: bool = True
    omni_talker_device: str | None = None
    omni_token2wav_device: str | None = None
    omni_output_max_pending: int = 32
    # LoRA is an opt-in phase-2 capability. When disabled all of its state,
    # bank allocations and hot-path hooks are absent.
    enable_lora: bool = False
    lora_paths: tuple[str, ...] = ()
    max_loras: int = 4
    max_lora_rank: int = 16
    lora_dtype: str = "auto"
    lora_allowed_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        if self.task not in {"generate", "embed"}:
            raise EngineError(f"unsupported engine task {self.task!r}")
        if self.embedding_max_batch_size <= 0:
            raise EngineError("embedding_max_batch_size must be positive")
        if self.embedding_max_sequence_length is not None and (
            self.embedding_max_sequence_length <= 0
        ):
            raise EngineError("embedding_max_sequence_length must be positive")
        if self.embedding_coalesce_wait_us < 0:
            raise EngineError("embedding_coalesce_wait_us cannot be negative")
        object.__setattr__(self, "lora_paths", tuple(self.lora_paths))
        roots = list(self.lora_allowed_roots)
        if roots:
            # Startup adapters are trusted; their parent directories are
            # implicitly allowed so an allowlist does not break boot.
            for path in self.lora_paths:
                resolved = str(Path(path).expanduser().resolve())
                if resolved not in roots:
                    roots.append(resolved)
        object.__setattr__(self, "lora_allowed_roots", tuple(roots))
        try:
            LoRAConfig(
                enable_lora=self.enable_lora,
                lora_paths=self.lora_paths,
                max_loras=self.max_loras,
                max_lora_rank=self.max_lora_rank,
                lora_dtype=self.lora_dtype,
                allowed_roots=self.lora_allowed_roots,
            )
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        graph_sizes = tuple(sorted(set(self.cuda_graph_batch_sizes)))
        object.__setattr__(self, "cuda_graph_batch_sizes", graph_sizes)
        object.__setattr__(
            self,
            "decode_chunk_steps",
            max(1, int(self.decode_chunk_steps)),
        )
        object.__setattr__(self, "compile_regions", tuple(self.compile_regions))
        object.__setattr__(
            self,
            "attention_sink_logits",
            tuple(self.attention_sink_logits),
        )
        try:
            AttentionConfig(
                mode=self.attention_mode,
                sink_tokens=len(self.attention_sink_logits),
                sparse_top_k=self.attention_sparse_top_k,
                sparse_block_size=self.attention_sparse_block_size,
                mla_kv_lora_rank=self.attention_mla_kv_lora_rank,
                mla_qk_rope_head_dim=self.attention_mla_qk_rope_head_dim,
                mla_qk_nope_head_dim=self.attention_mla_qk_nope_head_dim,
            )
        except AttentionConfigError as exc:
            raise EngineError(str(exc)) from exc
        object.__setattr__(
            self,
            "compile_buckets",
            tuple(sorted(set(self.compile_buckets))) or (self.prefill_chunk_size,),
        )
        try:
            CompilePolicy(
                backend=self.compile_backend,
                regions=self.compile_regions,
                tolerance=self.compile_tolerance,
                min_speedup=self.compile_min_speedup,
                candidate_tokens=self.compile_buckets,
                min_tokens=self.compile_min_tokens,
                max_artifact_bytes=self.compile_max_artifact_bytes,
                startup_budget_seconds=self.compile_startup_budget_seconds,
            )
        except CompileBackendError as exc:
            raise EngineError(str(exc)) from exc
        if (
            self.compile_backend != "none"
            and graph_sizes
            and min(self.compile_buckets) <= max(graph_sizes)
        ):
            raise EngineError(
                "compile buckets must exceed the largest CUDA Graph decode "
                "batch size: compiled regions are prefill-only"
            )
        if (
            self.compile_backend != "none"
            and max(self.compile_buckets) > self.max_num_batched_tokens
        ):
            raise EngineError(
                "compile buckets cannot exceed max_num_batched_tokens: the "
                "scheduler never emits those shapes"
            )
        if self.dtype not in {"float16", "bfloat16"}:
            raise EngineError("M1 supports float16 and bfloat16 only")
        if self.tensor_parallel <= 0:
            raise EngineError("tensor_parallel must be positive")
        if not 0 <= self.tp_rank < self.tensor_parallel:
            raise EngineError("tp_rank is outside the tensor-parallel world")
        if self.tensor_parallel == 1 and self.tp_rank != 0:
            raise EngineError("TP=1 requires rank 0")
        if self.tp_backend not in {"nccl", "gloo"}:
            raise EngineError("tp_backend must be nccl or gloo")
        if self.tensor_parallel > 1 and not self.tp_init_method:
            raise EngineError("tp_init_method is required for TP>1")
        if self.tensor_parallel > 1:
            if self.speculative_mode != "off":
                raise EngineError("TP>1 does not support speculative decoding")
            if self.enable_lora:
                raise EngineError("TP>1 does not support LoRA")
            if self.compile_backend != "none":
                raise EngineError("TP>1 does not support compiled artifacts yet")
        if (
            min(
                self.page_size,
                self.min_runtime_memory_bytes,
                self.max_model_len if self.max_model_len is not None else 32_768,
                self.max_num_sequences,
                self.max_num_batched_tokens,
                self.prefill_chunk_size,
                self.mixed_prefill_chunk_size,
                self.cpu_intraop_threads,
                self.vision_max_encode_patches,
                self.omni_light_admission_batch_size,
                self.video_admission_defer_min_target,
                self.triton_moe_max_tokens,
                self.tensorcore_moe_min_tokens,
                self.grouped_moe_min_tokens,
                self.num_speculative_tokens,
                self.prompt_lookup_min_ngram,
                self.prompt_lookup_max_ngram,
                self.prompt_lookup_min_candidate_support,
                self.prompt_lookup_probe_interval,
                self.omni_output_max_pending,
            )
            <= 0
        ):
            raise EngineError("engine limits must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise EngineError("gpu_memory_utilization must be in (0, 1]")
        if self.kv_cache_dtype not in {"auto", *_KV_QUANT_DTYPES}:
            raise EngineError(
                f"kv_cache_dtype must be auto or one of {sorted(_KV_QUANT_DTYPES)}"
            )
        if self.kv_cache_q_dtype not in {"auto", "fp8_e4m3fn", "fp8_e5m2"}:
            raise EngineError("kv_cache_q_dtype must be auto, fp8_e4m3fn, or fp8_e5m2")
        if self.kv_cache_q_group not in {"per_tensor", "per_kv_head"}:
            raise EngineError("kv_cache_q_group must be per_tensor or per_kv_head")
        for name, value in (
            ("kv_cache_key_scale", self.kv_cache_key_scale),
            ("kv_cache_value_scale", self.kv_cache_value_scale),
            ("kv_cache_q_scale", self.kv_cache_q_scale),
            ("nvfp4_global_scale", self.nvfp4_global_scale),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise EngineError(f"{name} must be positive and finite")
        if self.kv_cache_dtype == "auto" and (
            self.kv_cache_key_scale is not None
            or self.kv_cache_value_scale is not None
            or self.kv_cache_q_scale is not None
        ):
            raise EngineError("KV scales require a quantized kv_cache_dtype")
        if self.native_fp8_attn and self.kv_cache_dtype == "auto":
            raise EngineError("native_fp8_attn requires a quantized kv_cache_dtype")
        if self.speculative_mode not in {
            "off",
            "ngram",
            "suffix",
            "dflash",
            "mtp",
        }:
            raise EngineError(
                "speculative_mode must be off, ngram, suffix, dflash, or mtp"
            )
        if self.num_speculative_tokens > 32:
            raise EngineError(
                "num_speculative_tokens cannot exceed verifier capacity 32"
            )
        if self.speculative_mode == "mtp" and self.max_num_batched_tokens < 2:
            raise EngineError(
                "native MTP requires room for current plus one draft token"
            )
        if self.prompt_lookup_min_ngram > self.prompt_lookup_max_ngram:
            raise EngineError("prompt lookup n-gram range is invalid")
        if not 0.0 <= self.prompt_lookup_min_candidate_agreement <= 1.0:
            raise EngineError("prompt lookup candidate agreement must be in [0, 1]")
        if self.vision_cache_bytes < 0:
            raise EngineError("vision_cache_bytes must be non-negative")
        if self.kv_host_cache_bytes < 0 or self.kv_disk_cache_bytes < 0:
            raise EngineError("KV offload budgets cannot be negative")
        if self.kv_disk_cache_bytes and not self.kv_host_cache_bytes:
            raise EngineError("the disk tier requires a host staging tier")
        if self.kv_disk_cache_dir and not self.kv_disk_cache_bytes:
            raise EngineError("a disk tier directory requires a byte budget")
        if self.kv_disk_cache_bytes and not self.kv_disk_cache_dir:
            raise EngineError("a disk tier byte budget requires a directory")
        if (
            min(
                self.kv_disk_read_threads,
                self.kv_disk_write_threads,
            )
            <= 0
        ):
            raise EngineError("disk tier thread counts must be positive")
        if self.omni_admission_coalesce_wait_us < 0:
            raise EngineError("Omni admission coalesce wait must be non-negative")
        if self.video_admission_max_defer_ms < 0:
            raise EngineError("video admission deferral must be non-negative")
        if self.max_cache_tokens is not None:
            if self.max_cache_tokens < self.page_size:
                raise EngineError("KV cache must contain at least one page")
            if self.max_cache_tokens % self.page_size:
                raise EngineError("max_cache_tokens must be page aligned")
        if self.enable_cuda_graph and not self.use_triton:
            raise EngineError("production decode graphs require fixed Triton kernels")
        if self.enable_cuda_graph and not any(
            size <= self.max_num_sequences for size in self.cuda_graph_batch_sizes
        ):
            raise EngineError("no CUDA Graph bucket fits max_num_sequences")
        if self.grouped_moe_min_tokens < self.tensorcore_moe_min_tokens:
            raise EngineError("grouped MoE guard cannot precede Tensor Core MoE")


@dataclass(slots=True)
class EngineMetrics:
    steps: int = 0
    eager_steps: int = 0
    graph_steps: int = 0
    scheduled_tokens: int = 0
    emitted_tokens: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    mixed_steps: int = 0
    speculative_steps: int = 0
    speculative_draft_attempts: int = 0
    speculative_no_draft: int = 0
    speculative_draft_tokens: int = 0
    speculative_accepted_draft_tokens: int = 0
    speculative_emitted_tokens: int = 0
    # Engine-level constrained decoding (xgrammar).
    structured_requests: int = 0
    structured_degraded: int = 0
    structured_mask_steps: int = 0
    structured_masked_rows: int = 0
    structured_rejected: int = 0
    structured_grammar_errors: int = 0
    # Multi-LoRA observability (RED "errors" + USE "utilization" signals).
    lora_requests: int = 0
    lora_steps: int = 0
    lora_eager_steps: int = 0
    lora_tokens: int = 0
    lora_fallback_steps: int = 0
    # Request outcomes by finish reason (RED "errors"/outcomes signal).
    finish_reasons: Counter[str] = field(default_factory=Counter)
    # GPU utilization / memory (USE "utilization" + "saturation" signals).
    gpu_utilization: int = 0
    gpu_memory_utilization: int = 0
    gpu_memory_free_bytes: int = 0
    gpu_memory_total_bytes: int = 0
    gpu_memory_allocated_bytes: int = 0
    # Bound to the Engine's GrammarManager for snapshot enrichment.
    _grammar: Any = field(default=None, repr=False)
    step_time_ns: int = 0
    prefill_time_ns: int = 0
    decode_time_ns: int = 0
    mixed_time_ns: int = 0
    scheduler_time_ns: int = 0
    model_launch_time_ns: int = 0
    sampler_launch_time_ns: int = 0
    overlap_window_time_ns: int = 0
    completion_time_ns: int = 0
    sample_wait_time_ns: int = 0
    scheduler_commit_time_ns: int = 0
    media_admission_batches: int = 0
    media_admission_requests: int = 0
    media_admission_launch_time_ns: int = 0
    media_stream_batches: int = 0
    media_stream_submit_time_ns: int = 0
    media_stream_wait_time_ns: int = 0
    sequence_batch_histogram: Counter[int] = field(default_factory=Counter)
    token_batch_histogram: Counter[int] = field(default_factory=Counter)
    graph_bucket_histogram: Counter[int] = field(default_factory=Counter)
    media_admission_batch_histogram: Counter[int] = field(default_factory=Counter)

    def snapshot(self) -> dict[str, object]:
        result = {
            "steps": self.steps,
            "eager_steps": self.eager_steps,
            "graph_steps": self.graph_steps,
            "scheduled_tokens": self.scheduled_tokens,
            "emitted_tokens": self.emitted_tokens,
            "prefill_steps": self.prefill_steps,
            "decode_steps": self.decode_steps,
            "mixed_steps": self.mixed_steps,
            "speculative_steps": self.speculative_steps,
            "speculative_draft_attempts": (self.speculative_draft_attempts),
            "speculative_no_draft": self.speculative_no_draft,
            "speculative_draft_tokens": self.speculative_draft_tokens,
            "speculative_accepted_draft_tokens": (
                self.speculative_accepted_draft_tokens
            ),
            "speculative_emitted_tokens": (self.speculative_emitted_tokens),
            # Derived acceptance rate for same-condition speed/quality ablations.
            "speculative_acceptance_rate": (
                self.speculative_accepted_draft_tokens
                / self.speculative_draft_tokens
                if self.speculative_draft_tokens
                else 0.0
            ),
            "structured_requests": self.structured_requests,
            "structured_degraded": self.structured_degraded,
            "structured_mask_steps": self.structured_mask_steps,
            "structured_masked_rows": self.structured_masked_rows,
            "structured_rejected": self.structured_rejected,
            "structured_grammar_errors": self.structured_grammar_errors,
            "lora_requests": self.lora_requests,
            "lora_steps": self.lora_steps,
            "lora_eager_steps": self.lora_eager_steps,
            "lora_tokens": self.lora_tokens,
            "lora_fallback_steps": self.lora_fallback_steps,
            "finish_reasons": dict(sorted(self.finish_reasons.items())),
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "gpu_memory_free_bytes": self.gpu_memory_free_bytes,
            "gpu_memory_total_bytes": self.gpu_memory_total_bytes,
            "gpu_memory_allocated_bytes": self.gpu_memory_allocated_bytes,
            "step_time_ms": self.step_time_ns / 1_000_000,
            "prefill_time_ms": self.prefill_time_ns / 1_000_000,
            "decode_time_ms": self.decode_time_ns / 1_000_000,
            "mixed_time_ms": self.mixed_time_ns / 1_000_000,
            "scheduler_time_ms": self.scheduler_time_ns / 1_000_000,
            "model_launch_time_ms": self.model_launch_time_ns / 1_000_000,
            "sampler_launch_time_ms": self.sampler_launch_time_ns / 1_000_000,
            "overlap_window_time_ms": self.overlap_window_time_ns / 1_000_000,
            "completion_time_ms": self.completion_time_ns / 1_000_000,
            "sample_wait_time_ms": self.sample_wait_time_ns / 1_000_000,
            "scheduler_commit_time_ms": (self.scheduler_commit_time_ns / 1_000_000),
            "media_admission_batches": self.media_admission_batches,
            "media_admission_requests": self.media_admission_requests,
            "media_admission_launch_time_ms": (
                self.media_admission_launch_time_ns / 1_000_000
            ),
            "media_stream_batches": self.media_stream_batches,
            "media_stream_submit_time_ms": (
                self.media_stream_submit_time_ns / 1_000_000
            ),
            "media_stream_wait_time_ms": (self.media_stream_wait_time_ns / 1_000_000),
            "sequence_batch_histogram": dict(
                sorted(self.sequence_batch_histogram.items())
            ),
            "token_batch_histogram": dict(sorted(self.token_batch_histogram.items())),
            "graph_bucket_histogram": dict(sorted(self.graph_bucket_histogram.items())),
            "media_admission_batch_histogram": dict(
                sorted(self.media_admission_batch_histogram.items())
            ),
        }
        if self._grammar is not None:
            for key, value in self._grammar.stats().items():
                result[f"structured_{key}"] = value
        return result


@dataclass(frozen=True, slots=True)
class DeviceMemoryProfile:
    """Cold-path GPU budget used to size the one physical KV arena.

    The budget follows SGLang's static-fraction rule, but records the actual
    CUDA Graph allocation and validates the remaining runtime headroom after
    capture. An explicit cache-token value is only an upper bound; automatic
    mode additionally caps the arena at the configured concurrency envelope.
    """

    total_bytes: int
    initial_free_bytes: int
    free_after_weights_bytes: int
    resident_model_bytes: int
    workspace_bytes: int
    fixed_state_bytes: int
    vision_cache_capacity_bytes: int
    runtime_reserve_bytes: int
    cache_budget_bytes: int
    cache_bytes: int
    cache_tokens: int
    cache_capacity_source: str
    graph_bytes: int
    free_after_init_bytes: int

    def snapshot(self) -> dict[str, int | str]:
        return {
            "total_bytes": self.total_bytes,
            "initial_free_bytes": self.initial_free_bytes,
            "free_after_weights_bytes": self.free_after_weights_bytes,
            "resident_model_bytes": self.resident_model_bytes,
            "workspace_bytes": self.workspace_bytes,
            "fixed_state_bytes": self.fixed_state_bytes,
            "vision_cache_capacity_bytes": (self.vision_cache_capacity_bytes),
            "runtime_reserve_bytes": self.runtime_reserve_bytes,
            "cache_budget_bytes": self.cache_budget_bytes,
            "cache_bytes": self.cache_bytes,
            "cache_tokens": self.cache_tokens,
            "cache_capacity_source": self.cache_capacity_source,
            "graph_bytes": self.graph_bytes,
            "free_after_init_bytes": self.free_after_init_bytes,
        }


@dataclass(frozen=True, slots=True)
class KernelWarmupProfile:
    prefill_moe_token_counts: tuple[int, ...]
    full_prefill_sequences: int
    full_prefill_tokens: int
    greedy_sampling_batch_size: int
    duration_ns: int

    def snapshot(self) -> dict[str, object]:
        return {
            "prefill_moe_token_counts": self.prefill_moe_token_counts,
            "full_prefill_sequences": self.full_prefill_sequences,
            "full_prefill_tokens": self.full_prefill_tokens,
            "greedy_sampling_batch_size": self.greedy_sampling_batch_size,
            "duration_ms": self.duration_ns / 1_000_000,
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    token_ids: tuple[int, ...]
    text: str
    finish_reason: FinishReason


@dataclass(frozen=True, slots=True)
class PendingEngineStep:
    """One GPU batch launched on the Replica's ordered CUDA stream.

    CUDA execution is asynchronous until ``finish_step`` consumes sampled
    tokens on the host. The gap is intentionally available to the Gateway for
    relaying the previous batch without permitting a second in-flight batch.
    """

    prior_events: tuple[EngineEvent, ...]
    scheduled: ScheduledBatch | None
    sampled: tensorplay.Tensor | None
    sample_ready_event: tensorplay.cuda.Event | None
    step_started_ns: int
    scheduled_ns: int
    model_launched_ns: int
    sampler_launched_ns: int
    sampled_logprobs: tensorplay.Tensor | None = None
    # Host-side top-k (token_id, logprob) rows aligned with the scheduled rows.
    top_logprobs: tuple[tuple[tuple[int, float], ...] | None, ...] | None = None
    # Decision reads: raw next-token log-probs for the requested label ids.
    score_label_rows: tuple[int, ...] = ()
    score_logprobs: tensorplay.Tensor | None = None
    speculative_accepted: tensorplay.Tensor | None = None
    omni_input_embeddings: tensorplay.Tensor | None = None
    omni_hidden_states: tensorplay.Tensor | None = None
    # Decode steps folded into this one launch by the device-side self-loop.
    chunk_steps: int = 1


@dataclass(slots=True)
class PendingMultimodalBatch:
    """Media tower work enqueued independently from the decode stream."""

    ordered: tuple[MultimodalAdmission, ...]
    outputs: list[str | BaseException | None]
    valid: tuple[tuple[int, MultimodalAdmission], ...]
    resolved: (
        dict[
            str,
            tuple[
                ResolvedImages | None,
                ResolvedVideos | None,
                ResolvedAudios | None,
            ],
        ]
        | None
    )
    resolution_error: BaseException | None
    ready_event: tensorplay.cuda.Event | None
    started_ns: int
    consumed: bool = False


@dataclass(slots=True)
class _OmniFeatureCapture:
    prompt_token_ids: tuple[int, ...]
    audio_output: AudioOutputOptions
    deadline_ns: int | None = None
    features: list[tensorplay.Tensor] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _OmniPendingTerminal:
    finish_reason: FinishReason
    next_sequence_no: int


class Engine:
    """TP=1 embedded fast path: frontend, Leader scheduler and executor in-process."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        emit_engine_events: bool = True,
    ) -> None:
        self.config = config
        self.emit_engine_events = bool(emit_engine_events)
        self.startup = StartupTracker(model_path=str(config.model_path))
        tensorplay.set_num_threads(config.cpu_intraop_threads)
        self.cpu_intraop_threads = tensorplay.get_num_threads()
        self.device = tensorplay.device(config.device)
        if self.device.type != "cuda" or not tensorplay.cuda.is_available():
            raise EngineError("M1 requires an NVIDIA CUDA device")
        tensorplay.cuda.synchronize(self.device)
        initial_free_bytes, total_bytes = tensorplay.cuda.mem_get_info(self.device)
        properties = tensorplay.cuda.get_device_properties(self.device)
        if (properties.major, properties.minor) < (8, 0):
            raise EngineError("MegaLM v1 requires NVIDIA Ampere or newer")
        if config.tensor_parallel > 1:
            self.tp_group: CollectiveGroup = DistributedCollectiveGroup(
                TPGroupConfig(
                    rank=config.tp_rank,
                    world_size=config.tensor_parallel,
                    init_method=config.tp_init_method,
                    backend=config.tp_backend,
                )
            )
        else:
            self.tp_group = NOOP_COLLECTIVE
        (
            self.media_stream_priority,
            self.execution_stream_priority,
        ) = tensorplay.cuda.Stream.priority_range()
        dtype = {
            "float16": tensorplay.float16,
            "bfloat16": tensorplay.bfloat16,
        }[config.dtype]
        self.kernel_policy = KernelPolicy(
            use_triton=config.use_triton,
            moe_max_tokens=config.triton_moe_max_tokens,
            moe_tensorcore_min_tokens=config.tensorcore_moe_min_tokens,
            grouped_moe_min_tokens=config.grouped_moe_min_tokens,
        )
        self.hardware_spec = HardwareSpec(
            accelerator="cuda",
            device_name=properties.name,
            compute_capability=(properties.major, properties.minor),
            total_memory_bytes=properties.total_memory,
        )
        # Bind the probed hardware to its provider declaration (cold path).
        # The evidence digest lets DISCOVER and DRAFT detect capability drift.
        _hardware_provider = HARDWARE_PROVIDERS.select(device="cuda")
        self.provider_evidence = ProviderEvidence.create(
            provider=_hardware_provider.name,
            accelerator=_hardware_provider.accelerator,
            device_name=properties.name,
            compute_capability=(int(properties.major), int(properties.minor)),
            total_memory_bytes=int(properties.total_memory),
            free_memory_bytes=int(initial_free_bytes),
            supported_dtypes=_hardware_provider.supported_dtypes,
            backends=_hardware_provider.backends,
        )
        self.model_spec = load_model_spec(config.model_path)
        self._full_attention_layers = tuple(
            layer.index
            for layer in self.model_spec.layers
            if layer.attention is AttentionKind.FULL
        )
        full_attention = next(
            (
                layer
                for layer in self.model_spec.layers
                if layer.attention is AttentionKind.FULL
            ),
            None,
        )
        if full_attention is None:
            raise EngineError("model spec declares no full-attention layer")
        include_prefixes = (
            (
                "thinker.model.",
                "thinker.lm_head.",
                "thinker.visual.",
                "thinker.audio_tower.",
            )
            if self.model_spec.model_type == "qwen2_5_omni"
            else ()
        )
        source = SafetensorsSource(
            config.model_path,
            include_prefixes=include_prefixes,
            exclude_suffixes=(
                KV_SCALE_SUFFIXES if config.kv_cache_dtype != "auto" else ()
            ),
        )
        checkpoint_scales: dict[str, tuple[float, ...]] = {}
        if config.kv_cache_dtype == "fp8_per_head":
            checkpoint_scales = read_kv_scale_vectors(
                config.model_path,
                num_kv_heads=full_attention.num_key_value_heads,
                num_query_heads=full_attention.num_attention_heads,
                include_prefixes=include_prefixes,
            )
        kv_quant = _resolve_kv_quant(
            config,
            full_attention.num_key_value_heads,
            checkpoint_scales=checkpoint_scales,
        )
        if kv_quant is not None and kv_quant.int4_use_rht:
            head_dim = int(full_attention.head_dim)
            if head_dim & (head_dim - 1):
                raise EngineError(
                    f"int4_use_rht requires a power-of-two head_dim, got {head_dim}"
                )
        kv_dtype = kv_quant.dtype if kv_quant is not None else dtype
        self.kv_quant = kv_quant
        if self.model_spec.loop is not None:
            # A looped model (Inkling-2, Nanbeige4.2) replays its core: every
            # layer visit owns its own KV cache layer.
            self._full_attention_layers = tuple(
                range(kv_cache_layer_count(self.model_spec))
            )
        self._gdn_layers = tuple(
            layer.index
            for layer in self.model_spec.layers
            if layer.attention is AttentionKind.GATED_DELTA_NET
        )
        self._hybrid_state = bool(self._gdn_layers)
        self._transactional_gdn = self._hybrid_state and config.speculative_mode in {
            "ngram",
            "suffix",
            "dflash",
            "mtp",
        }
        self._draft_source: DraftSource | None = build_draft_source(
            config.speculative_mode,
            max_ngram_size=config.prompt_lookup_max_ngram,
        )
        self._prefix_cache_enabled = (
            config.enable_prefix_cache and not self._hybrid_state
        )
        # Inkling-2 keeps short-convolution history outside the paged K/V
        # arena.  The radix cache is still usable, but only for boundaries
        # whose convolution state was checkpointed; the store below carries
        # those checkpoints and is wired into the cache manager.
        self._prefix_state = (
            None
            if not self._prefix_cache_enabled
            or self.model_spec.short_convolution is None
            else PrefixStateStore(
                self._capture_prefix_state,
                self._restore_prefix_state,
            )
        )
        source_dtype = _checkpoint_source_dtype(source.manifest)
        max_model_len = min(
            config.max_model_len if config.max_model_len is not None else 32_768,
            self.model_spec.position.max_positions,
        )
        self.max_model_len = max_model_len
        workload = WorkloadProfile(
            max_batch_size=config.max_num_sequences,
            max_sequence_length=max_model_len,
            typical_prefill_tokens=min(
                config.prefill_chunk_size,
                max_model_len,
            ),
            typical_decode_tokens=min(256, max_model_len),
        )
        graph_sizes = (
            ()
            if not config.enable_cuda_graph
            else tuple(
                size
                for size in config.cuda_graph_batch_sizes
                if size <= config.max_num_sequences
            )
        )
        self.topology = TopologySpec(
            tensor_parallel=config.tensor_parallel,
            data_parallel=1,
            node_count=1,
            gpus_per_node=config.tensor_parallel,
        )
        self.attention_config = AttentionConfig(
            mode=config.attention_mode,
            sink_tokens=len(config.attention_sink_logits),
            sparse_top_k=config.attention_sparse_top_k,
            sparse_block_size=config.attention_sparse_block_size,
            mla_kv_lora_rank=config.attention_mla_kv_lora_rank,
            mla_qk_rope_head_dim=config.attention_mla_qk_rope_head_dim,
            mla_qk_nope_head_dim=config.attention_mla_qk_nope_head_dim,
        )
        binding_spec = discover_kernel_plan(
            self.kernel_policy,
            device=self.device,
            dtype=dtype,
            model_type=self.model_spec.model_type,
            attention_config=self.attention_config,
        )
        self.startup.advance(
            StartupPhase.DRAFT,
            model_type=self.model_spec.model_type,
            architecture=self.model_spec.architecture,
            device=str(self.device),
            device_name=self.hardware_spec.device_name,
            compute_capability=self.hardware_spec.compute_capability,
            source_dtype=source_dtype,
            dtype=config.dtype,
            max_model_len=max_model_len,
            max_num_sequences=config.max_num_sequences,
            max_num_batched_tokens=config.max_num_batched_tokens,
            prefix_cache=self._prefix_cache_enabled,
            cuda_graph_buckets=graph_sizes,
            model_digest=self.model_spec.digest,
            kernel_binding_digest=binding_spec.digest,
        )
        compile_regions = tuple(sorted(set(config.compile_regions)))
        compile_buckets = tuple(sorted(set(config.compile_buckets)))
        if config.compile_backend == "none":
            self.draft_compiled = CompiledPlan.disabled()
        else:
            compile_identity = compiled_identity_digest(
                backend=config.compile_backend,
                regions=compile_regions,
                buckets=compile_buckets,
                dtype=config.dtype,
                device_name=self.hardware_spec.device_name,
                model_digest=self.model_spec.digest,
                tolerance=config.compile_tolerance,
                min_speedup=config.compile_min_speedup,
            )
            self.draft_compiled = CompiledPlan(
                backend=config.compile_backend,
                regions=compile_regions,
                buckets=compile_buckets,
                digest=compile_identity,
                artifact_bytes=0,
                max_artifact_bytes=config.compile_max_artifact_bytes,
            )
        self.plan_draft = build_plan_draft(
            model_spec=self.model_spec,
            hardware=self.hardware_spec,
            source_dtype=source_dtype,
            dtype=config.dtype,
            workload=workload,
            estimated_model_bytes=source.total_bytes,
            page_size=config.page_size,
            max_batched_tokens=config.max_num_batched_tokens,
            cuda_graph_buckets=graph_sizes,
            binding_spec=binding_spec,
            enable_prefix_cache=self._prefix_cache_enabled,
            topology=self.topology,
            compiled=self.draft_compiled,
        )
        if self.tp_group.enabled:
            draft_identity = (
                self.plan_draft.model_digest,
                self.plan_draft.kernel_binding_digest,
                self.plan_draft.topology,
            )
            agreed = self.tp_group.broadcast_object(draft_identity)
            if agreed != draft_identity:
                raise EngineError("tensor-parallel ranks drafted different plans")
        self.kernel_bindings = bind_kernel_plan(
            self.kernel_policy,
            device=self.device,
            dtype=dtype,
            expected=binding_spec,
            model_type=self.model_spec.model_type,
            attention_config=self.attention_config,
            sink_logits=config.attention_sink_logits,
        )
        self.startup.advance(
            StartupPhase.LOAD_PACK,
            estimated_model_bytes=self.plan_draft.estimated_model_bytes,
            weight_tensors=len(source.manifest),
        )
        self.tokenizer = ModelTokenizer(config.model_path)
        self.model, self.weight_report = load_native_model(
            config.model_path,
            device=self.device,
            dtype=dtype,
            kernel_policy=self.kernel_policy,
            kernel_bindings=self.kernel_bindings,
            enable_mtp=config.speculative_mode == "mtp",
            mtp_model_path=config.speculative_model_path,
            tp_size=config.tensor_parallel,
            tp_rank=config.tp_rank,
            collective=self.tp_group,
        )
        if self.model_spec.short_convolution is not None:
            self.model.num_state_slots = max(
                self.model.num_state_slots, config.max_num_sequences
            )
        lora_dtype = (
            dtype
            if config.lora_dtype == "auto"
            else getattr(tensorplay, config.lora_dtype)
        )
        self.lora_manager = LoRAManager(
            LoRAConfig(
                enable_lora=config.enable_lora,
                lora_paths=config.lora_paths,
                max_loras=config.max_loras,
                max_lora_rank=config.max_lora_rank,
                lora_dtype=config.lora_dtype,
                allowed_roots=config.lora_allowed_roots,
            ),
            device=self.device,
            dtype=lora_dtype,
            use_triton=config.use_triton and self.device.type == "cuda",
        )
        try:
            self.lora_manager.bind_model(self.model)
            for lora_path in config.lora_paths:
                self.lora_manager.load(lora_path)
        except LoRAError as exc:
            raise EngineError(str(exc)) from exc
        self.compile_report = CompileReport.disabled()
        if config.compile_backend != "none":
            try:
                self.compile_report = compile_model(
                    self.model,
                    policy=CompilePolicy(
                        backend=config.compile_backend,
                        regions=config.compile_regions,
                        tolerance=config.compile_tolerance,
                        min_speedup=config.compile_min_speedup,
                        candidate_tokens=config.compile_buckets,
                        min_tokens=config.compile_min_tokens,
                        max_artifact_bytes=config.compile_max_artifact_bytes,
                        startup_budget_seconds=(config.compile_startup_budget_seconds),
                    ),
                    dtype=config.dtype,
                    device=self.device,
                    enable_lora=config.enable_lora,
                )
            except Exception as exc:  # noqa: BLE001 - cold path must not fail startup
                self.compile_report = CompileReport.failed(
                    config.compile_backend,
                    f"{type(exc).__name__}: {exc}",
                )
            log_event(
                _LOGGER,
                (
                    logging.INFO
                    if self.compile_report.accepted
                    or self.compile_report.attempted == 0
                    else logging.WARNING
                ),
                "engine.compile",
                **self.compile_report.as_dict(),
            )
        self._request_lora: dict[str, str | int] = {}
        self._grammar = GrammarManager(
            self.tokenizer,
            vocab_size=self.model.config.vocab_size,
        )
        self.omni_output: OmniSpeechStage | None = None
        self._omni_captures: dict[str, _OmniFeatureCapture] = {}
        self._omni_terminals: dict[str, _OmniPendingTerminal] = {}
        self._omni_ready_events: list[EngineEvent] = []
        if (
            config.enable_omni_audio_output
            and self.model_spec.model_type == "qwen2_5_omni"
            and bool(getattr(self.model.config, "enable_audio_output", False))
        ):
            try:
                omni_backend = TransformersOmniSpeechBackend(
                    config.model_path,
                    talker_device=(config.omni_talker_device or config.device),
                    token2wav_device=(config.omni_token2wav_device or config.device),
                    dtype=dtype,
                )
            except OmniOutputError as exc:
                raise EngineError(
                    f"failed to initialize native Omni audio output: {exc}"
                ) from exc
            self.omni_output = OmniSpeechStage(
                omni_backend,
                max_pending=config.omni_output_max_pending,
            )
        if config.speculative_mode == "mtp":
            self.model.configure_mtp_hidden_buffer(config.max_num_batched_tokens)
        self._media_requests: dict[str, MediaState] = {}
        if self.model_spec.model_type in {
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_vl",
            "qwen3_vl_moe",
        }:
            self.image_processor = Qwen35ImageProcessor(self.model.config.vision_config)
        elif self.model_spec.model_type == "oxr":
            self.image_processor = OXRImageProcessor(self.model.config.vision_config)
        elif self.model_spec.model_type == "qwen2_5_omni":
            self.image_processor = Qwen25OmniImageProcessor(
                self.model.config.vision_config
            )
        elif self.model_spec.model_type in {"gemma4", "gemma4_unified"} and getattr(
            self.model, "vision_tower", None
        ) is not None:
            self.image_processor = Gemma4ImageProcessor(
                self.model.config.vision_config
            )
        else:
            self.image_processor = None
        if self.model_spec.model_type in {
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_vl",
            "qwen3_vl_moe",
        }:
            self.video_processor = Qwen35VideoProcessor(self.model.config.vision_config)
        elif self.model_spec.model_type == "qwen2_5_omni":
            self.video_processor = Qwen25OmniVideoProcessor(
                self.model.config.vision_config
            )
        else:
            self.video_processor = None
        if self.model_spec.model_type == "qwen2_5_omni":
            self.audio_processor = Qwen25OmniAudioProcessor(
                self.model.config.audio_config
            )
        elif self.model_spec.model_type in {"gemma4", "gemma4_unified"} and getattr(
            self.model, "audio_tower", None
        ) is not None:
            self.audio_processor = Gemma4AudioProcessor(
                self.model.config.audio_config
            )
        else:
            self.audio_processor = None
        if self.tokenizer.vocab_size > self.model.config.vocab_size:
            raise EngineError("tokenizer can emit ids outside the model vocabulary")
        model_max_positions = getattr(
            self.model.config, "max_position_embeddings", None
        )
        if model_max_positions is None:
            text_config = getattr(self.model.config, "text_config", None)
            model_max_positions = getattr(text_config, "max_position_embeddings", None)
        if model_max_positions != self.model_spec.position.max_positions:
            raise EngineError("ModelSpec and model context limits differ")
        self.startup.advance(
            StartupPhase.ALLOCATE_BIND,
            loaded_weight_tensors=self.weight_report.tensor_count,
            source_weight_bytes=self.weight_report.total_bytes,
            tokenizer_vocab_size=self.tokenizer.vocab_size,
            model_vocab_size=self.model.config.vocab_size,
        )
        resident_model_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                *tuple(self.model.parameters()),
                *tuple(self.model.buffers()),
            )
        )
        if self.omni_output is not None:
            backend = self.omni_output.backend
            for stage_name in ("talker", "token2wav"):
                stage = getattr(backend, stage_name, None)
                if stage is None:
                    continue
                resident_model_bytes += sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (
                        *tuple(stage.parameters()),
                        *tuple(stage.buffers()),
                    )
                    if tensor.device == self.device
                )
        tensorplay.cuda.synchronize(self.device)
        free_after_weights_bytes, _ = tensorplay.cuda.mem_get_info(self.device)
        workspace_bytes = self.model.configure_moe_workspace(
            config.max_num_batched_tokens
        )
        if workspace_bytes != self.plan_draft.estimated_workspace_bytes:
            raise EngineError("allocated MoE workspace differs from the PlanDraft")
        runtime_reserve_bytes = max(
            config.min_runtime_memory_bytes,
            initial_free_bytes
            - int(initial_free_bytes * config.gpu_memory_utilization),
        )
        cache_budget_bytes = (
            free_after_weights_bytes - workspace_bytes - runtime_reserve_bytes
        )
        fixed_state_bytes = estimate_gdn_state_bytes(
            self.model_spec,
            num_slots=(
                config.max_num_sequences * (2 if self._transactional_gdn else 1)
            ),
            dtype=dtype,
        )
        vision_cache_capacity_bytes = (
            config.vision_cache_bytes if self.image_processor is not None else 0
        )
        # Models with heterogeneous per-layer KV geometry (e.g. Gemma-4's
        # 256-wide sliding heads vs 512-wide full heads) normalise the arena
        # to the largest geometry and take per-layer views at runtime.
        cache_head_dim = int(
            getattr(self.model, "cache_head_dim", self.model.config.head_dim)
        )
        cache_kv_heads = int(
            getattr(
                self.model,
                "cache_num_kv_heads",
                self.model.config.num_key_value_heads,
            )
        )
        cache_layer_indices = tuple(
            getattr(self.model, "kv_cache_layer_indices", self._full_attention_layers)
        )
        cache_layer_count = len(cache_layer_indices)
        bytes_per_page = estimate_kv_page_bytes(
            self.model_spec,
            page_size=config.page_size,
            dtype=kv_dtype,
        )
        if hasattr(self.model, "cache_head_dim"):
            element_size = tensorplay.empty((), dtype=kv_dtype).element_size()
            bytes_per_page = (
                2
                * cache_layer_count
                * config.page_size
                * cache_kv_heads
                * cache_head_dim
                * element_size
            )
        if kv_quant is not None:
            # The estimator assumes one element per logical dimension; packed
            # 4-bit modes and inline per-token scales change the real footprint.
            bytes_per_page = (
                bytes_per_page * kv_quant.content_dim(cache_head_dim) // cache_head_dim
            )
        if config.speculative_mode == "mtp":
            mtp_element_size = (
                kv_quant.element_size
                if kv_quant
                else tensorplay.empty((), dtype=dtype).element_size()
            )
            bytes_per_page += (
                2
                * config.page_size
                * cache_kv_heads
                * cache_head_dim
                * mtp_element_size
            )
        pages_by_memory = (
            cache_budget_bytes - fixed_state_bytes - vision_cache_capacity_bytes
        ) // bytes_per_page
        logical_cache_tokens = max_model_len * config.max_num_sequences
        pages_by_workload = (
            logical_cache_tokens + config.page_size - 1
        ) // config.page_size
        if config.max_cache_tokens is None:
            pages_by_user = pages_by_workload
            cache_capacity_source = "profiled"
        else:
            pages_by_user = config.max_cache_tokens // config.page_size
            cache_capacity_source = "explicit_cap"
        num_pages = min(pages_by_memory, pages_by_workload, pages_by_user)
        if num_pages <= 0:
            raise EngineError(
                "weights and runtime reserve leave no page-aligned KV capacity; "
                "lower gpu_memory_utilization constraints or use a larger GPU"
            )
        cache_tokens = num_pages * config.page_size
        if config.max_model_len is None and config.tensor_parallel == 1:
            max_model_len = min(max_model_len, cache_tokens)
            self.max_model_len = max_model_len
        if cache_tokens < max_model_len:
            raise EngineError(
                f"KV budget holds {cache_tokens} tokens but max_model_len is "
                f"{max_model_len}; lower max_model_len or increase GPU memory budget"
            )
        kv_cache = PagedKVCache(
            # Looped and YOCO models expose their physical cache layer count
            # explicitly (Inkling-2 via RuntimeConfig, Nanbeige and Gemma-4 via
            # num_kv_cache_layers).
            num_layers=getattr(
                self.model,
                "num_kv_cache_layers",
                self.model.config.num_hidden_layers,
            ),
            num_pages=num_pages,
            page_size=config.page_size,
            num_kv_heads=cache_kv_heads,
            head_dim=cache_head_dim,
            dtype=dtype,
            device=self.device,
            layer_indices=cache_layer_indices,
            quant=kv_quant,
            compute_dtype=dtype,
        )
        self.mtp_cache = (
            PagedKVCache(
                num_layers=1,
                num_pages=num_pages,
                page_size=config.page_size,
                num_kv_heads=cache_kv_heads,
                head_dim=cache_head_dim,
                dtype=dtype,
                device=self.device,
                layer_indices=(0,),
                quant=kv_quant,
                compute_dtype=dtype,
            )
            if config.speculative_mode == "mtp"
            else None
        )
        if self._hybrid_state:
            gdn_specs = tuple(
                self.model_spec.layers[index].gated_delta_net
                for index in self._gdn_layers
            )
            first_gdn = gdn_specs[0]
            assert first_gdn is not None
            if any(spec != first_gdn for spec in gdn_specs):
                raise EngineError("one Replica requires a uniform GDN state geometry")
            key_width = first_gdn.num_key_heads * first_gdn.key_head_dim
            value_width = first_gdn.num_value_heads * first_gdn.value_head_dim
            gdn_cache = GDNStateCache(
                layer_indices=self._gdn_layers,
                num_slots=(
                    config.max_num_sequences * (2 if self._transactional_gdn else 1)
                ),
                conv_width=2 * key_width + value_width,
                conv_kernel_dim=first_gdn.conv_kernel_dim,
                num_value_heads=first_gdn.num_value_heads,
                key_head_dim=first_gdn.key_head_dim,
                value_head_dim=first_gdn.value_head_dim,
                dtype=dtype,
                device=self.device,
            )
            self.cache = HybridCache(kv=kv_cache, gdn=gdn_cache)
            if gdn_cache.nbytes != fixed_state_bytes:
                raise EngineError(
                    "allocated GDN state differs from the cold-path budget"
                )
        else:
            self.cache = kv_cache
        self.vision_cache = GPUVisionCache(
            capacity_bytes=vision_cache_capacity_bytes,
            # Deepstack VL towers emit the main hidden state followed by one
            # slice per deepstack tap; the cache stores the full packed row.
            width=self.model_spec.hidden_size
            * (1 + int(getattr(self.model, "deepstack_depth", 0))),
            dtype=dtype,
            device=self.device,
        )
        self.vision_pipeline = (
            None
            if self.image_processor is None
            else VisionPipeline(
                processor=self.image_processor,
                video_processor=self.video_processor,
                cache=self.vision_cache,
                model_digest=self.model_spec.digest,
                encode_images=self.model.encode_images,
                device=self.device,
                max_encode_patches=(config.vision_max_encode_patches),
            )
        )
        self.audio_pipeline = (
            None
            if self.audio_processor is None
            else AudioPipeline(
                processor=self.audio_processor,
                cache=self.vision_cache,
                model_digest=self.model_spec.digest,
                encode_audios=self.model.encode_audios,
            )
        )
        committed_cache_bytes = (
            self.cache.nbytes
            + (0 if self.mtp_cache is None else self.mtp_cache.nbytes)
            + vision_cache_capacity_bytes
        )
        offload: TieredKVOffload | None = None
        if self._prefix_cache_enabled and config.kv_host_cache_bytes:
            identity_parts = [
                "megalm-kv-v2" if kv_quant else "megalm-kv-v1",
                self.model_spec.digest,
                config.dtype,
                str(config.page_size),
                str(cache_layer_count),
                str(cache_kv_heads),
                str(cache_head_dim),
            ]
            if kv_quant is not None:
                identity_parts.append(kv_quant.scale_signature())
            identity = "|".join(identity_parts)
            identity_digest = hashlib.blake2b(
                identity.encode(), digest_size=32
            ).digest()
            try:
                host = HostKVTier(
                    identity_digest=identity_digest,
                    rows=cache_layer_count,
                    page_size=config.page_size,
                    num_kv_heads=cache_kv_heads,
                    head_dim=cache_head_dim,
                    dtype=kv_dtype,
                    capacity_bytes=config.kv_host_cache_bytes,
                    copier=TensorCopier(asynchronous=self.device.type == "cuda"),
                    pin_memory=self.device.type == "cuda",
                )
                disk = (
                    DiskKVTier(
                        root=config.kv_disk_cache_dir,
                        identity=identity,
                        identity_digest=identity_digest,
                        rows=cache_layer_count,
                        page_size=config.page_size,
                        num_kv_heads=cache_kv_heads,
                        head_dim=cache_head_dim,
                        dtype=kv_dtype,
                        capacity_bytes=config.kv_disk_cache_bytes,
                        host=host,
                        read_threads=config.kv_disk_read_threads,
                        write_threads=config.kv_disk_write_threads,
                    )
                    if config.kv_disk_cache_bytes
                    else None
                )
                offload = TieredKVOffload(
                    identity=identity,
                    host=host,
                    disk=disk,
                )
            except KVOffloadError as exc:
                raise EngineError(f"KV offload setup failed: {exc}") from exc
        self.offload = offload
        self.cache_manager = KVCacheManager(
            num_pages=num_pages,
            page_size=config.page_size,
            max_sequences=config.max_num_sequences,
            max_model_len=max_model_len,
            device=self.device,
            enable_prefix_cache=self._prefix_cache_enabled,
            gdn_state_slots=(
                (config.max_num_sequences * (2 if self._transactional_gdn else 1))
                if self._hybrid_state
                else 0
            ),
            gdn_states_per_request=(2 if self._transactional_gdn else 1),
            prefix_state=self._prefix_state,
            offload=offload,
            kv_quant=kv_quant,
            offload_key=kv_cache.key if offload is not None else None,
            offload_value=kv_cache.value if offload is not None else None,
        )
        self.scheduler = ContinuousBatchScheduler(
            self.cache_manager,
            eos_token_id=self.tokenizer.eos_token_ids,
            max_num_sequences=config.max_num_sequences,
            max_num_batched_tokens=config.max_num_batched_tokens,
            prefill_chunk_size=config.prefill_chunk_size,
            mixed_prefill_chunk_size=config.mixed_prefill_chunk_size,
            emit_events=self.emit_engine_events,
            align_prefill_for_state=self._prefix_state is not None,
            lora_manager=self.lora_manager if config.enable_lora else None,
        )
        self.sampler = Sampler(
            use_triton=config.use_triton,
            kernel_bindings=self.kernel_bindings.sampling,
        )
        speculative_rows = (
            min(config.num_speculative_tokens, config.max_num_batched_tokens - 1) + 1
            if config.speculative_mode == "mtp"
            else min(
                config.num_speculative_tokens,
                config.max_num_batched_tokens,
            )
        )
        speculative_enabled = config.speculative_mode in {
            "ngram",
            "suffix",
            "dflash",
            "mtp",
        }
        self._speculative_sampled = (
            tensorplay.empty(
                (config.max_num_sequences, speculative_rows),
                dtype=tensorplay.int64,
                device=self.device,
            )
            if speculative_enabled
            else None
        )
        self._speculative_targets = (
            tensorplay.empty(
                (config.max_num_sequences, speculative_rows),
                dtype=tensorplay.int64,
                device=self.device,
            )
            if speculative_enabled
            else None
        )
        self._speculative_sequence_lengths = (
            tensorplay.empty(
                speculative_rows,
                dtype=tensorplay.int32,
                device=self.device,
            )
            if speculative_enabled
            else None
        )
        self._speculative_accepted = (
            tensorplay.empty(
                config.max_num_sequences,
                dtype=tensorplay.int32,
                device=self.device,
            )
            if speculative_enabled
            else None
        )
        mtp_feature_taps = (
            4 if self.model_spec.model_type in {"inkling2", "inkling2_loop"} else 1
        )
        self._mtp_target_hidden = (
            tensorplay.empty(
                (speculative_rows, mtp_feature_taps, self.model_spec.hidden_size),
                dtype=dtype,
                device=self.device,
            )
            if config.speculative_mode == "mtp"
            else None
        )
        self.startup.advance(
            StartupPhase.WARM_CAPTURE,
            resident_model_bytes=resident_model_bytes,
            free_after_weights_bytes=free_after_weights_bytes,
            workspace_bytes=workspace_bytes,
            cache_bytes=committed_cache_bytes,
            cache_tokens=cache_tokens,
            cache_pages=num_pages,
            cache_capacity_source=cache_capacity_source,
            runtime_reserve_bytes=runtime_reserve_bytes,
            prefix_cache=self._prefix_cache_enabled,
            cuda_graph_buckets=graph_sizes,
        )
        self.graph_runner: DecodeGraphRunner | None = None
        self.mtp_graph_runner: MTPUpdateGraphRunner | None = None
        if graph_sizes:
            self.graph_runner = DecodeGraphRunner(
                self.model,
                self.cache,
                batch_sizes=graph_sizes,
                max_blocks_per_sequence=self.cache_manager.max_blocks_per_sequence,
                page_size=config.page_size,
                greedy_argmax=self.kernel_bindings.sampling.greedy_argmax,
                speculative_lengths=(
                    (2,)
                    if config.speculative_mode == "mtp"
                    else (
                        tuple(range(2, speculative_rows + 1))
                        if (
                            config.speculative_mode
                            in {"ngram", "suffix", "dflash"}
                            and (
                                (
                                    not self._hybrid_state
                                    and self.model_spec.model_type
                                    in {"qwen3", "qwen3_moe"}
                                )
                                or (
                                    self._hybrid_state
                                    and self.model_spec.model_type
                                    in {"qwen3_5", "qwen3_5_moe"}
                                )
                            )
                        )
                        else ()
                    )
                ),
                stream_priority=self.execution_stream_priority,
                strict=config.strict_cuda_graph,
                loop_steps=(
                    config.decode_chunk_steps
                    if config.speculative_mode == "off"
                    else 1
                ),
            )
            if config.speculative_mode == "mtp" and self.model_spec.model_type in {
                "inkling2",
                "inkling2_loop",
            }:
                # The DSpark draft takes host-side feature/slot arguments and
                # mutates recurrent state, so it runs eagerly.
                self.mtp_graph_runner = None
            elif config.speculative_mode == "mtp":
                predictor = getattr(self.model, "mtp", None)
                if predictor is None or self.mtp_cache is None:
                    raise EngineError("native MTP graph inputs are unavailable")
                self.mtp_graph_runner = MTPUpdateGraphRunner(
                    predictor,
                    self.model.lm_head(),
                    self.mtp_cache,
                    max_blocks_per_sequence=(
                        self.cache_manager.max_blocks_per_sequence
                    ),
                    page_size=config.page_size,
                    hidden_size=self.model_spec.hidden_size,
                    dtype=dtype,
                    stream_priority=self.execution_stream_priority,
                    strict=config.strict_cuda_graph,
                )
        warmup_started_ns = time.perf_counter_ns()
        prefill_moe_token_counts = self.model.warmup_prefill_kernels(
            max_batched_tokens=config.max_num_batched_tokens,
        )
        (
            full_prefill_sequences,
            full_prefill_tokens,
        ) = self._warmup_full_prefill(
            num_pages=num_pages,
            max_model_len=max_model_len,
        )
        greedy_sampling_batch_size = min(2, config.max_num_sequences)
        warmup_logits = tensorplay.zeros(
            (greedy_sampling_batch_size, self.model.config.vocab_size),
            dtype=dtype,
            device=self.device,
        )
        warmup_request_ids = tuple(
            f"__megalm_warmup_{index}" for index in range(greedy_sampling_batch_size)
        )
        warmup_sampled = self.sampler.sample(
            warmup_logits,
            (SamplingParams(),) * greedy_sampling_batch_size,
            warmup_request_ids,
            ((),) * greedy_sampling_batch_size,
        )
        tensorplay.cuda.synchronize(self.device)
        warmup_finished_ns = time.perf_counter_ns()
        for request_id in warmup_request_ids:
            self.sampler.release(request_id)
        del warmup_logits, warmup_sampled
        # Retain the full-prefill allocator pool. Releasing it here turns the
        # first production wave back into synchronous cudaMalloc traffic.
        tensorplay.cuda.synchronize(self.device)
        self.kernel_warmup_profile = KernelWarmupProfile(
            prefill_moe_token_counts=prefill_moe_token_counts,
            full_prefill_sequences=full_prefill_sequences,
            full_prefill_tokens=full_prefill_tokens,
            greedy_sampling_batch_size=greedy_sampling_batch_size,
            duration_ns=warmup_finished_ns - warmup_started_ns,
        )
        free_after_init_bytes, _ = tensorplay.cuda.mem_get_info(self.device)
        if free_after_init_bytes < config.min_runtime_memory_bytes:
            raise EngineError(
                "CUDA Graph capture consumed the runtime safety headroom; "
                "lower cache capacity, graph buckets, or gpu_memory_utilization"
            )
        self.startup.advance(
            StartupPhase.SEAL,
            graph_bytes=(
                (0 if self.graph_runner is None else self.graph_runner.memory_bytes)
                + (
                    0
                    if self.mtp_graph_runner is None
                    else self.mtp_graph_runner.memory_bytes
                )
            ),
            captured_graph_buckets=(
                () if self.graph_runner is None else self.graph_runner.batch_sizes
            ),
            kernel_warmup_ms=(self.kernel_warmup_profile.duration_ns / 1_000_000),
            full_prefill_tokens=full_prefill_tokens,
            free_after_init_bytes=free_after_init_bytes,
        )
        graph_buckets = (
            () if self.graph_runner is None else self.graph_runner.batch_sizes
        )
        graph_bytes = (
            0 if self.graph_runner is None else self.graph_runner.memory_bytes
        ) + (0 if self.mtp_graph_runner is None else self.mtp_graph_runner.memory_bytes)
        self.memory_profile = DeviceMemoryProfile(
            total_bytes=total_bytes,
            initial_free_bytes=initial_free_bytes,
            free_after_weights_bytes=free_after_weights_bytes,
            resident_model_bytes=resident_model_bytes,
            workspace_bytes=workspace_bytes,
            fixed_state_bytes=fixed_state_bytes,
            vision_cache_capacity_bytes=vision_cache_capacity_bytes,
            runtime_reserve_bytes=runtime_reserve_bytes,
            cache_budget_bytes=max(0, cache_budget_bytes),
            cache_bytes=committed_cache_bytes,
            cache_tokens=cache_tokens,
            cache_capacity_source=cache_capacity_source,
            graph_bytes=graph_bytes,
            free_after_init_bytes=free_after_init_bytes,
        )
        accepted_regions = tuple(
            sorted(
                {
                    item.region
                    for item in self.compile_report.regions
                    if item.accepted
                }
            )
        )
        accepted_buckets = tuple(
            sorted(
                {
                    bucket
                    for item in self.compile_report.regions
                    if item.accepted
                    for bucket in item.buckets
                }
            )
        )
        sealed_compiled = CompiledPlan.disabled()
        if not self.draft_compiled.minimal:
            if self.compile_report.backend != self.draft_compiled.backend:
                raise EngineError("compile provider changed between DRAFT and seal")
            sealed_compiled = CompiledPlan(
                backend=self.draft_compiled.backend,
                regions=accepted_regions,
                buckets=accepted_buckets,
                digest=self.draft_compiled.digest,
                artifact_bytes=self.compile_report.artifact_bytes,
                max_artifact_bytes=self.draft_compiled.max_artifact_bytes,
            )
        self.execution_plan = build_execution_plan(
            model_spec=self.model_spec,
            hardware=self.hardware_spec,
            resident_model_bytes=resident_model_bytes,
            source_dtype=source_dtype,
            dtype=config.dtype,
            workload=workload,
            cache_bytes=committed_cache_bytes,
            graph_bytes=graph_bytes,
            workspace_bytes=workspace_bytes + sealed_compiled.artifact_bytes,
            page_size=config.page_size,
            max_batched_tokens=config.max_num_batched_tokens,
            cuda_graph_buckets=graph_buckets,
            kernel_policy=self.kernel_policy,
            linear_backend=self.kernel_bindings.linear_backend,
            prefill_backend=self.kernel_bindings.prefill_backend,
            kernel_artifacts=self.kernel_bindings.artifacts,
            kernel_binding_digest=self.kernel_bindings.digest,
            enable_prefix_cache=self._prefix_cache_enabled,
            topology=self.topology,
            compiled=sealed_compiled,
        )
        self.plan_draft.verify_sealed(self.execution_plan)
        self.execution_plan.verify_digest()
        if self.tp_group.enabled:
            sealed_digest = self.tp_group.broadcast_object(
                self.execution_plan.plan_digest
            )
            if sealed_digest != self.execution_plan.plan_digest:
                raise EngineError("tensor-parallel ranks sealed different plans")
        self.startup.advance(
            StartupPhase.READY,
            plan_digest=self.execution_plan.plan_digest,
            model_type=self.model_spec.model_type,
            dtype=config.dtype,
            device=str(self.device),
            resident_model_bytes=resident_model_bytes,
            cache_tokens=cache_tokens,
            cache_pages=num_pages,
            prefix_cache=self._prefix_cache_enabled,
            cuda_graph_buckets=graph_buckets,
            kernel_backends=tuple(
                sorted({kernel.backend for kernel in self.execution_plan.kernels})
            ),
            free_after_init_bytes=free_after_init_bytes,
        )
        self.metrics = EngineMetrics(_grammar=self._grammar)
        self._mtp_drafts: dict[str, tuple[int, int]] = {}
        self._mtp_ready: set[str] = set()
        self._pending_step: PendingEngineStep | None = None
        # Decode gets the device's highest priority while media towers stay at
        # the lowest. This preserves token cadence when the two streams overlap.
        self.execution_stream = tensorplay.cuda.Stream(
            device=self.device,
            priority=self.execution_stream_priority,
        )
        self.media_stream = tensorplay.cuda.Stream(
            device=self.device,
            priority=self.media_stream_priority,
        )
        if self.omni_output is not None:
            self.omni_output.start()

    @tensorplay.inference_mode()
    def _warmup_full_prefill(
        self,
        *,
        num_pages: int,
        max_model_len: int,
    ) -> tuple[int, int]:
        """Run one real full-model prefill without touching production state."""

        if max_model_len <= 1:
            return 0, 0
        prompt_length = min(
            max_model_len - 1,
            self.config.prefill_chunk_size,
            self.config.max_num_batched_tokens,
        )
        pages_per_sequence = (
            prompt_length + self.config.page_size - 1
        ) // self.config.page_size
        sequence_count = min(
            self.config.max_num_sequences,
            self.config.max_num_batched_tokens // prompt_length,
            num_pages // pages_per_sequence,
        )
        if sequence_count <= 0:
            return 0, 0
        warmup_cache_manager = KVCacheManager(
            num_pages=num_pages,
            page_size=self.config.page_size,
            max_sequences=self.config.max_num_sequences,
            max_model_len=max_model_len,
            device=self.device,
            enable_prefix_cache=False,
            gdn_state_slots=(
                self.config.max_num_sequences if self._hybrid_state else 0
            ),
        )
        warmup_scheduler = ContinuousBatchScheduler(
            warmup_cache_manager,
            eos_token_id=self.tokenizer.eos_token_ids,
            max_num_sequences=self.config.max_num_sequences,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            prefill_chunk_size=self.config.prefill_chunk_size,
        )
        token_ids = (0,) * prompt_length
        sampling = SamplingParams(ignore_eos=True)
        request_ids = tuple(
            f"__megalm_full_prefill_{index}" for index in range(sequence_count)
        )
        for request_id in request_ids:
            warmup_scheduler.submit(
                InferenceRequest(
                    request_id=request_id,
                    model=self.model.config.model_type,
                    input_token_ids=token_ids,
                    max_new_tokens=1,
                    sampling=sampling,
                )
            )
        scheduled = warmup_scheduler.schedule()
        if scheduled is None:
            raise EngineError("full-model prefill warmup could not be scheduled")
        expected_tokens = sequence_count * prompt_length
        if scheduled.execution.num_tokens != expected_tokens:
            raise EngineError("full-model prefill warmup shape is incomplete")
        logits = self.model(scheduled.execution, self.cache)
        sampled = self.sampler.sample(
            logits,
            scheduled.sampling,
            scheduled.request_ids,
            scheduled.histories,
            history_counts=scheduled.history_counts,
        )
        completion = warmup_scheduler.complete(scheduled, sampled)
        if len(completion.finished_request_ids) != sequence_count:
            raise EngineError("full-model prefill warmup did not terminate")
        for request_id in request_ids:
            self.sampler.release(request_id)
        return sequence_count, expected_tokens

    def descriptor(self) -> dict[str, object]:
        """Immutable model, hardware, memory and kernel facts for frontends."""

        plan = self.execution_plan
        vision_stats = self.vision_cache.stats()
        media_cache = {
            **asdict(vision_stats),
            "hit_rate": vision_stats.hit_rate,
        }
        cache = self.cache_manager.stats().snapshot()
        return {
            "model_path": str(self.config.model_path),
            "tokenizer_path": str(self.config.model_path),
            "model_type": self.model_spec.model_type,
            "input_modalities": (
                ()
                if self.image_processor is None
                else (
                    (
                        ("image", "video", "audio")
                        if self.audio_processor is not None
                        else ("image", "video")
                    )
                    if self.video_processor is not None
                    else (
                        ("image", "audio")
                        if self.audio_processor is not None
                        else ("image",)
                    )
                )
            ),
            "output_modalities": (
                ("text", "audio") if self.omni_output is not None else ("text",)
            ),
            "audio_output": (
                None
                if self.omni_output is None
                else {
                    "format": "wav",
                    "sample_rate": 24_000,
                    "voices": self.omni_output.voices,
                    "default_voice": (
                        "Chelsie"
                        if "Chelsie" in self.omni_output.voices
                        else self.omni_output.voices[0]
                    ),
                }
            ),
            "dtype": self.config.dtype,
            "max_model_len": self.max_model_len,
            "max_num_sequences": self.config.max_num_sequences,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "prefill_chunk_size": self.config.prefill_chunk_size,
            "mixed_prefill_chunk_size": (self.config.mixed_prefill_chunk_size),
            "cpu_intraop_threads": self.cpu_intraop_threads,
            "vision_max_encode_patches": (self.config.vision_max_encode_patches),
            "omni_admission_coalesce_wait_us": (
                self.config.omni_admission_coalesce_wait_us
            ),
            "omni_light_admission_batch_size": (
                self.config.omni_light_admission_batch_size
            ),
            "video_admission_max_defer_ms": (self.config.video_admission_max_defer_ms),
            "video_admission_defer_min_target": (
                self.config.video_admission_defer_min_target
            ),
            "media_admission_streams": 2,
            "async_multimodal_admission": True,
            "execution_stream_priority": self.execution_stream_priority,
            "media_stream_priority": self.media_stream_priority,
            "speculative_mode": self.config.speculative_mode,
            "num_speculative_tokens": (self.config.num_speculative_tokens),
            "prompt_lookup_min_ngram": (self.config.prompt_lookup_min_ngram),
            "prompt_lookup_max_ngram": (self.config.prompt_lookup_max_ngram),
            "prompt_lookup_min_candidate_support": (
                self.config.prompt_lookup_min_candidate_support
            ),
            "prompt_lookup_min_candidate_agreement": (
                self.config.prompt_lookup_min_candidate_agreement
            ),
            "prompt_lookup_probe_interval": (self.config.prompt_lookup_probe_interval),
            "enable_prefix_cache": self._prefix_cache_enabled,
            "enable_cuda_graph": self.graph_runner is not None,
            "cuda_graph_batch_sizes": (
                () if self.graph_runner is None else self.graph_runner.batch_sizes
            ),
            "cuda_graph_speculative_lengths": (
                ()
                if self.graph_runner is None
                else self.graph_runner.speculative_lengths
            ),
            "hardware": {
                "accelerator": plan.hardware.accelerator,
                "device_name": plan.hardware.device_name,
                "compute_capability": plan.hardware.compute_capability,
                "total_memory_bytes": plan.hardware.total_memory_bytes,
            },
            "provider": self.provider_evidence.describe(),
            "lora": {
                **self.lora_manager.stats(),
                "adapters": self.lora_manager.describe(),
            },
            "memory": self.memory_profile.snapshot(),
            "cache": cache,
            "media_cache": media_cache,
            "vision_cache": media_cache,
            "kernel_warmup": self.kernel_warmup_profile.snapshot(),
            "startup": {
                "phase": self.startup.phase.value,
                "plan_schema_version": plan.schema_version,
                "draft_model_digest": self.plan_draft.model_digest,
                "kernel_binding_digest": plan.kernel_binding_digest,
                "compiled": plan.compiled.as_dict(),
                "tensor_parallel": plan.topology.tensor_parallel,
            },
            "scheduler": {
                "policy": "decode_first_mixed_chunked_prefill",
                "max_num_sequences": self.config.max_num_sequences,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
                "prefill_chunk_size": self.config.prefill_chunk_size,
                "mixed_prefill_chunk_size": (self.config.mixed_prefill_chunk_size),
            },
            "kernels": [
                {
                    "operator": kernel.operator,
                    "phases": tuple(phase.value for phase in kernel.phases),
                    "backend": kernel.backend,
                    "layer_indices": kernel.layer_indices,
                    "guard": asdict(kernel.guard),
                    "variant": kernel.variant,
                    "abi_version": kernel.abi_version,
                    "artifact_digest": kernel.artifact_digest,
                    "weight_layout": kernel.weight_layout,
                    "workspace": kernel.workspace,
                    "workspace_bytes": kernel.workspace_bytes,
                }
                for kernel in plan.kernels
            ],
            "plan_digest": plan.plan_digest,
        }

    def load_lora(
        self,
        path: str,
        *,
        name: str | None = None,
        adapter_id: int | None = None,
    ) -> dict[str, Any]:
        """Dynamically register (and activate) one PEFT adapter."""

        try:
            outcome = self.lora_manager.load(path, name=name, adapter_id=adapter_id)
        except LoRAError as exc:
            log_event(
                _LOGGER,
                logging.WARNING,
                "engine.lora_load_failed",
                path=path,
                name=name,
                code=exc.code,
                message=str(exc),
            )
            raise EngineLoRAError(
                str(exc), code=exc.code, retryable=exc.retryable
            ) from exc
        log_event(
            _LOGGER,
            logging.INFO,
            "engine.lora_loaded",
            adapter_id=outcome["id"],
            name=outcome["name"],
            slot=outcome["slot"],
            rank=outcome["rank"],
        )
        return outcome

    def unload_lora(self, reference: str | int) -> dict[str, Any]:
        """Unregister an adapter by id or name; in-flight requests keep their slot."""

        try:
            outcome = self.lora_manager.unload(reference)
        except LoRAError as exc:
            log_event(
                _LOGGER,
                logging.WARNING,
                "engine.lora_unload_failed",
                reference=str(reference),
                code=exc.code,
                message=str(exc),
            )
            raise EngineLoRAError(
                str(exc), code=exc.code, retryable=exc.retryable
            ) from exc
        log_event(
            _LOGGER,
            logging.INFO,
            "engine.lora_unloaded",
            adapter_id=outcome["id"],
            name=outcome["name"],
        )
        return outcome

    def list_loras(self) -> list[dict[str, Any]]:
        return self.lora_manager.describe()

    def add_request(
        self,
        input_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        sampling: SamplingParams | None = None,
        request_id: str | None = None,
        priority: int = 0,
        deadline_ns: int | None = None,
        prefix_fingerprint: str | None = None,
        output_modalities: Sequence[OutputModality] = (OutputModality.TEXT,),
        audio_output: AudioOutputOptions | None = None,
        lora_request: LoRARequest | None = None,
    ) -> str:
        token_ids = tuple(int(token) for token in input_token_ids)
        if any(
            token_id < 0 or token_id >= self.model.config.vocab_size
            for token_id in token_ids
        ):
            raise EngineError("input token id is outside the model vocabulary")
        if lora_request is not None:
            if not self.config.enable_lora:
                raise EngineError("LoRA is disabled on this replica")
            if (
                audio_output is not None
                or self.model_spec.model_type not in GENERIC_LORA_MODEL_TYPES
            ):
                raise EngineError(
                    "LoRA is only supported on text-only generic causal LMs"
                )
        required_context = len(token_ids) + max_new_tokens
        capacity = min(
            self.max_model_len,
            self.cache.geometry.token_capacity,
        )
        if required_context > capacity:
            raise EngineError(
                f"request can require {required_context} context tokens, "
                f"replica capacity is {capacity}"
            )
        resolved_id = request_id or uuid4().hex
        modalities = tuple(OutputModality(item) for item in output_modalities)
        if OutputModality.AUDIO in modalities:
            stage = self.omni_output
            if stage is None:
                raise EngineError(
                    "the loaded Replica does not support Omni audio output"
                )
            if audio_output is None:
                raise EngineError("audio output modality requires audio options")
            if audio_output.voice not in stage.voices:
                raise EngineError(
                    f"unknown Omni voice {audio_output.voice!r}; "
                    f"available={stage.voices}"
                )
        # Constrained decoding needs per-token target logits; speculative
        # verification and xgrammar are independent state machines. The helper
        # degrades such requests to best-effort instead of risking a mismatch.
        # Start the matcher as late as possible and roll it back on any failure
        # below, so a rejected admission cannot leak grammar state.
        requested_structured = (
            sampling is not None and sampling.structured_output is not None
        )
        prior_grammar = self._grammar.get(resolved_id)
        resolved_sampling = resolve_structured_output(
            sampling or SamplingParams(),
            manager=self._grammar,
            request_id=resolved_id,
            speculative_mode=self.config.speculative_mode,
        )
        # Only roll back a matcher this admission actually created. A duplicate
        # id keeps the live request's matcher and degrades the new one instead.
        owns_grammar = self._grammar.get(resolved_id) is not prior_grammar
        try:
            request = InferenceRequest(
                request_id=resolved_id,
                model=self.model.config.model_type,
                input_token_ids=token_ids,
                max_new_tokens=max_new_tokens,
                sampling=resolved_sampling,
                output_modalities=modalities,
                audio_output=audio_output,
                priority=priority,
                deadline_ns=deadline_ns,
                prefix_fingerprint=prefix_fingerprint,
                lora_request=lora_request,
            )
            self.scheduler.submit(request)
        except Exception as exc:
            if owns_grammar:
                self._grammar.release(resolved_id)
            if lora_request is not None:
                self.lora_manager.unpin(
                    lora_request.lora_int_id or lora_request.lora_name
                )
            if isinstance(exc, SchedulerLoRAError):
                log_event(
                    _LOGGER,
                    logging.WARNING,
                    "engine.lora_rejected",
                    request_id=resolved_id,
                    code=exc.code,
                    message=exc.error.message,
                    retryable=exc.retryable,
                )
            raise
        if lora_request is not None:
            self.metrics.lora_requests += 1
            self._request_lora[resolved_id] = (
                lora_request.lora_int_id or lora_request.lora_name
            )
        if requested_structured:
            self.metrics.structured_requests += 1
            if resolved_sampling.structured_output is None:
                self.metrics.structured_degraded += 1
        if audio_output is not None:
            self._omni_captures[resolved_id] = _OmniFeatureCapture(
                prompt_token_ids=token_ids,
                audio_output=audio_output,
                deadline_ns=deadline_ns,
            )
        if self.tp_group.enabled and self.tp_group.rank == 0:
            # Only a committed admission is mirrored: a failed submit must not
            # leave the follower scheduler holding a request rank0 rejected.
            self.tp_group.broadcast_object(
                {
                    "kind": "add",
                    "token_ids": token_ids,
                    "max_new_tokens": max_new_tokens,
                    "sampling": sampling,
                    "request_id": resolved_id,
                    "priority": priority,
                    "deadline_ns": deadline_ns,
                    "prefix_fingerprint": prefix_fingerprint,
                    "output_modalities": tuple(
                        item.value for item in modalities
                    ),
                    "audio_output": audio_output,
                    "lora_request": lora_request,
                }
            )
        return resolved_id

    def tp_apply_command(self, command: dict[str, Any]) -> None:
        """Apply a leader-broadcast control command on a TP follower rank."""

        kind = command.get("kind")
        if kind == "add":
            self.add_request(
                command["token_ids"],
                max_new_tokens=command["max_new_tokens"],
                sampling=command.get("sampling"),
                request_id=command["request_id"],
                priority=command.get("priority", 0),
                deadline_ns=command.get("deadline_ns"),
                prefix_fingerprint=command.get("prefix_fingerprint"),
                output_modalities=tuple(
                    OutputModality(item)
                    for item in command.get(
                        "output_modalities", (OutputModality.TEXT.value,)
                    )
                ),
                audio_output=command.get("audio_output"),
                lora_request=command.get("lora_request"),
            )
            return
        if kind == "cancel":
            self.cancel(command["request_id"])
            return
        raise EngineError(f"unknown TP command kind {kind!r}")

    def add_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int,
        sampling: SamplingParams | None = None,
        request_id: str | None = None,
        priority: int = 0,
        deadline_ns: int | None = None,
        open_thinking: bool = False,
        output_modalities: Sequence[OutputModality] = (OutputModality.TEXT,),
        audio_output: AudioOutputOptions | None = None,
        lora_request: LoRARequest | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        token_ids = self.tokenizer.encode_chat(
            messages,
            open_thinking=open_thinking,
            extras=chat_template_kwargs,
        )
        return self.add_request(
            token_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            request_id=request_id,
            priority=priority,
            deadline_ns=deadline_ns,
            output_modalities=output_modalities,
            audio_output=audio_output,
            lora_request=lora_request,
        )

    @tensorplay.inference_mode()
    def add_multimodal_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        images: Sequence[ImageInput] = (),
        *,
        videos: Sequence[VideoInput] = (),
        audios: Sequence[AudioInput] = (),
        max_new_tokens: int,
        sampling: SamplingParams | None = None,
        request_id: str | None = None,
        priority: int = 0,
        deadline_ns: int | None = None,
        open_thinking: bool = False,
        output_modalities: Sequence[OutputModality] = (OutputModality.TEXT,),
        audio_output: AudioOutputOptions | None = None,
        tools: tuple[dict[str, Any], ...] = (),
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        admission = MultimodalAdmission(
            messages=tuple(messages),
            images=tuple(images),
            videos=tuple(videos),
            audios=tuple(audios),
            max_new_tokens=max_new_tokens,
            sampling=sampling or SamplingParams(),
            request_id=request_id or uuid4().hex,
            priority=priority,
            deadline_ns=deadline_ns,
            open_thinking=open_thinking,
            output_modalities=tuple(output_modalities),
            audio_output=audio_output,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        started_ns = time.perf_counter_ns()
        try:
            return self._admit_multimodal_single(admission)
        finally:
            self._record_media_admission(
                1,
                time.perf_counter_ns() - started_ns,
            )

    @tensorplay.inference_mode()
    def add_multimodal_batch(
        self,
        admissions: Sequence[MultimodalAdmission],
    ) -> tuple[str | BaseException, ...]:
        """Resolve one HTTP wave while preserving the synchronous Engine API."""

        return self.finish_multimodal_batch(self.begin_multimodal_batch(admissions))

    @tensorplay.inference_mode()
    def begin_multimodal_batch(
        self,
        admissions: Sequence[MultimodalAdmission],
    ) -> PendingMultimodalBatch:
        """Enqueue one batched media-tower wave on the media CUDA stream."""

        ordered = tuple(admissions)
        if not ordered:
            return PendingMultimodalBatch(
                ordered=(),
                outputs=[],
                valid=(),
                resolved=None,
                resolution_error=None,
                ready_event=None,
                started_ns=time.perf_counter_ns(),
            )
        request_ids = tuple(item.request_id for item in ordered)
        if any(not request_id for request_id in request_ids) or len(
            set(request_ids)
        ) != len(request_ids):
            raise EngineError(
                "multimodal batch request ids must be unique and non-empty"
            )
        started_ns = time.perf_counter_ns()
        outputs: list[str | BaseException | None] = [None] * len(ordered)
        valid: list[tuple[int, MultimodalAdmission]] = []
        for index, admission in enumerate(ordered):
            try:
                self._validate_multimodal(admission)
            except BaseException as exc:
                outputs[index] = exc
            else:
                valid.append((index, admission))
        resolved = None
        resolution_error = None
        ready_event = None
        if valid:
            valid_admissions = tuple(item for _, item in valid)
            ready_event = tensorplay.cuda.Event()
            submit_started_ns = time.perf_counter_ns()
            with tensorplay.cuda.stream(self.media_stream):
                try:
                    resolved = self._resolve_multimodal_batch(valid_admissions)
                except BaseException as exc:
                    resolution_error = exc
                finally:
                    ready_event.record(self.media_stream)
            self.metrics.media_stream_batches += 1
            self.metrics.media_stream_submit_time_ns += (
                time.perf_counter_ns() - submit_started_ns
            )
        return PendingMultimodalBatch(
            ordered=ordered,
            outputs=outputs,
            valid=tuple(valid),
            resolved=resolved,
            resolution_error=resolution_error,
            ready_event=ready_event,
            started_ns=started_ns,
        )

    @staticmethod
    def multimodal_batch_ready(pending: PendingMultimodalBatch) -> bool:
        """Return without host blocking when the media CUDA event is complete."""

        if pending.consumed:
            raise EngineError("multimodal batch has already been consumed")
        event = pending.ready_event
        return event is None or event.query()

    @tensorplay.inference_mode()
    def finish_multimodal_batch(
        self,
        pending: PendingMultimodalBatch,
    ) -> tuple[str | BaseException, ...]:
        """Wait for media work, then atomically expose admissions to scheduling."""

        if pending.consumed:
            raise EngineError("multimodal batch has already been consumed")
        pending.consumed = True
        wait_started_ns = time.perf_counter_ns()
        if pending.ready_event is not None:
            pending.ready_event.synchronize()
        self.metrics.media_stream_wait_time_ns += (
            time.perf_counter_ns() - wait_started_ns
        )
        outputs = pending.outputs
        if pending.valid:
            if pending.resolution_error is not None:
                for index, admission in pending.valid:
                    try:
                        outputs[index] = self._admit_multimodal_single(admission)
                    except BaseException as exc:
                        outputs[index] = exc
            else:
                if pending.resolved is None:
                    raise EngineError(
                        "multimodal media event completed without results"
                    )
                for index, admission in pending.valid:
                    try:
                        outputs[index] = self._commit_multimodal(
                            admission,
                            *pending.resolved[admission.request_id],
                        )
                    except BaseException as exc:
                        self._media_requests.pop(
                            admission.request_id,
                            None,
                        )
                        self.vision_cache.release_request(admission.request_id)
                        outputs[index] = exc
        if pending.ordered:
            self._record_media_admission(
                len(pending.ordered),
                time.perf_counter_ns() - pending.started_ns,
            )
        if any(output is None for output in outputs):
            raise EngineError("multimodal batch admission left an unresolved result")
        return tuple(output for output in outputs if output is not None)

    @tensorplay.inference_mode()
    def cancel_multimodal_batch(
        self,
        pending: PendingMultimodalBatch,
    ) -> None:
        """Synchronize and release a media wave that will not be committed."""

        if pending.consumed:
            return
        pending.consumed = True
        if pending.ready_event is not None:
            pending.ready_event.synchronize()
        for _, admission in pending.valid:
            self._media_requests.pop(admission.request_id, None)
            self.vision_cache.release_request(admission.request_id)

    def _admit_multimodal_single(
        self,
        admission: MultimodalAdmission,
    ) -> str:
        self._validate_multimodal(admission)
        pipeline = self.vision_pipeline
        audio_pipeline = self.audio_pipeline
        try:
            resolved_images = (
                None
                if not admission.images
                else pipeline.resolve(
                    admission.request_id,
                    admission.images,
                )
            )
            resolved_videos = (
                None
                if not admission.videos
                else pipeline.resolve_videos(
                    admission.request_id,
                    admission.videos,
                )
            )
            resolved_audios = (
                None
                if not admission.audios
                else audio_pipeline.resolve(
                    admission.request_id,
                    admission.audios,
                )
            )
            return self._commit_multimodal(
                admission,
                resolved_images,
                resolved_videos,
                resolved_audios,
            )
        except BaseException:
            self._media_requests.pop(admission.request_id, None)
            self.vision_cache.release_request(admission.request_id)
            raise

    def _resolve_multimodal_batch(
        self,
        admissions: tuple[MultimodalAdmission, ...],
    ) -> dict[
        str,
        tuple[
            ResolvedImages | None,
            ResolvedVideos | None,
            ResolvedAudios | None,
        ],
    ]:
        pipeline = self.vision_pipeline
        audio_pipeline = self.audio_pipeline
        resolved: dict[str, list[object | None]] = {
            admission.request_id: [None, None, None] for admission in admissions
        }
        try:
            image_inputs = tuple(
                (admission.request_id, admission.images)
                for admission in admissions
                if admission.images
            )
            video_inputs = tuple(
                (admission.request_id, admission.videos)
                for admission in admissions
                if admission.videos
            )
            if image_inputs or video_inputs:
                assert pipeline is not None
                resolved_images, resolved_videos = pipeline.resolve_media_batch(
                    image_requests=image_inputs,
                    video_requests=video_inputs,
                )
                for (request_id, _), value in zip(
                    image_inputs,
                    resolved_images,
                    strict=True,
                ):
                    resolved[request_id][0] = value
                for (request_id, _), value in zip(
                    video_inputs,
                    resolved_videos,
                    strict=True,
                ):
                    resolved[request_id][1] = value
            audio_inputs = tuple(
                (admission.request_id, admission.audios)
                for admission in admissions
                if admission.audios
            )
            if audio_inputs:
                assert audio_pipeline is not None
                for (request_id, _), value in zip(
                    audio_inputs,
                    audio_pipeline.resolve_batch(audio_inputs),
                    strict=True,
                ):
                    resolved[request_id][2] = value
        except BaseException:
            for admission in admissions:
                self.vision_cache.release_request(admission.request_id)
            raise
        return {request_id: tuple(values) for request_id, values in resolved.items()}

    def _commit_multimodal(
        self,
        admission: MultimodalAdmission,
        resolved_images: ResolvedImages | None,
        resolved_videos: ResolvedVideos | None,
        resolved_audios: ResolvedAudios | None,
    ) -> str:
        prepared = build_media_admission(
            admission,
            tokenizer=self.tokenizer,
            resolved_images=resolved_images,
            resolved_videos=resolved_videos,
            resolved_audios=resolved_audios,
            model_type=self.model_spec.model_type,
            spatial_merge_size=getattr(
                self.model.config.vision_config, "spatial_merge_size", 1
            ),
            position_ids_per_second=getattr(
                self.model.config,
                "position_id_per_seconds",
                None,
            ),
            device=self.device,
        )
        self._media_requests[admission.request_id] = prepared.state
        return self.add_request(
            prepared.input_ids,
            max_new_tokens=admission.max_new_tokens,
            sampling=admission.sampling,
            request_id=admission.request_id,
            priority=admission.priority,
            deadline_ns=admission.deadline_ns,
            prefix_fingerprint=prepared.prefix_fingerprint,
            output_modalities=admission.output_modalities,
            audio_output=admission.audio_output,
        )

    def _validate_multimodal(
        self,
        admission: MultimodalAdmission,
    ) -> None:
        if (admission.images or admission.videos) and self.vision_pipeline is None:
            raise EngineError("the loaded native model does not support visual inputs")
        if admission.audios and self.audio_pipeline is None:
            raise EngineError("the loaded native model does not support audio inputs")
        if not admission.images and not admission.videos and not admission.audios:
            raise EngineError("multimodal chat requires image, video, or audio input")

    def _record_media_admission(
        self,
        batch_size: int,
        duration_ns: int,
    ) -> None:
        self.metrics.media_admission_batches += 1
        self.metrics.media_admission_requests += batch_size
        self.metrics.media_admission_launch_time_ns += duration_ns
        self.metrics.media_admission_batch_histogram[batch_size] += 1

    def update_weights(self, path: str) -> dict[str, Any]:
        """Cold, admin-driven weight swap. Refuses to run with live requests."""

        from pathlib import Path as _Path

        if not self.scheduler.is_idle:
            raise EngineError("weight update requires an idle scheduler")
        loaded_path = _Path(path)
        if not loaded_path.exists():
            raise EngineError(f"weight path does not exist: {path}")
        report = self.model.load_weights(loaded_path)
        log_event(
            _LOGGER,
            logging.INFO,
            "weights_updated",
            path=path,
            digest=self.plan_draft.model_digest,
        )
        return {
            "ok": True,
            "digest": self.plan_draft.model_digest,
            "source_bytes": getattr(report, "total_bytes", 0),
        }

    def cancel(self, request_id: str) -> EngineEvent:
        if self.tp_group.enabled and self.tp_group.rank == 0:
            self.tp_group.broadcast_object(
                {"kind": "cancel", "request_id": request_id}
            )
        if (
            request_id in self._omni_terminals
            and self.omni_output is not None
            and self.omni_output.cancel(request_id)
        ):
            return EngineEvent(
                request_id=request_id,
                sequence_no=self._omni_terminals[request_id].next_sequence_no,
                kind=EventKind.STATE_CHANGED,
                state=RequestState.OUTPUT_BLOCKED,
                monotonic_ns=time.monotonic_ns(),
                schema_version=2,
            )
        event = self.scheduler.cancel(request_id)
        self._omni_captures.pop(request_id, None)
        self.sampler.release(request_id)
        self._grammar.release(request_id)
        self._release_media_state(request_id)
        self._release_lora(request_id)
        self._mtp_drafts.pop(request_id, None)
        self._mtp_ready.discard(request_id)
        return event

    def _release_lora(self, request_id: str) -> None:
        reference = self._request_lora.pop(request_id, None)
        if reference is not None:
            self.lora_manager.unpin(reference)

    def _release_media_state(self, request_id: str) -> None:
        state = self._media_requests.pop(request_id, None)
        if state is not None:
            self.vision_cache.release_request(request_id)

    @tensorplay.inference_mode()
    def begin_step(self) -> PendingEngineStep:
        """Schedule one batch and enqueue model plus sampler kernels.

        Exactly one step may be in flight. Callers may do bounded host work
        before ``finish_step``; scheduling or launching another batch first
        would violate the scheduler/cache ownership contract.
        """

        if self._pending_step is not None:
            raise EngineError("an Engine step is already in flight")
        step_started_ns = time.perf_counter_ns()
        prior_events = self.scheduler.drain_events()
        scheduled = self.scheduler.schedule()
        needs_score = bool(
            scheduled
            and any(
                item is not None and bool(item.score_labels)
                for item in scheduled.sampling
            )
        )
        if scheduled is None:
            scheduled_ns = time.perf_counter_ns()
            if not self.scheduler.is_idle and self.scheduler.is_waiting_on_offload:
                # Lower-tier promotion is in flight. Yield so the leader
                # loop can poll the transfer without burning a core.
                self.cache_manager.poll_offload(block=True, timeout_s=0.002)
                time.sleep(0.0005)
            elif not self.scheduler.is_idle:
                raise EngineError(
                    "scheduler cannot make progress within the cache budget"
                )
            pending = PendingEngineStep(
                prior_events=prior_events,
                scheduled=None,
                sampled=None,
                sample_ready_event=None,
                step_started_ns=step_started_ns,
                scheduled_ns=scheduled_ns,
                model_launched_ns=scheduled_ns,
                sampler_launched_ns=scheduled_ns,
            )
            self._pending_step = pending
            return pending
        if (
            self.config.speculative_mode in {"ngram", "suffix", "dflash"}
            and not needs_score
            and scheduled.execution.all_decode
            and not any(
                request_id in self._omni_captures or request_id in self._request_lora
                for request_id in scheduled.request_ids
            )
        ):
            self.metrics.speculative_draft_attempts += 1
            prepared = self.scheduler.prepare_ngram_speculative(
                scheduled,
                num_speculative_tokens=(self.config.num_speculative_tokens),
                min_ngram_size=self.config.prompt_lookup_min_ngram,
                max_ngram_size=self.config.prompt_lookup_max_ngram,
                min_candidate_support=(self.config.prompt_lookup_min_candidate_support),
                min_candidate_agreement=(
                    self.config.prompt_lookup_min_candidate_agreement
                ),
                probe_interval=self.config.prompt_lookup_probe_interval,
                draft_source=self._draft_source,
            )
            if prepared is scheduled:
                self.metrics.speculative_no_draft += 1
            else:
                scheduled = prepared
                self.metrics.speculative_draft_tokens += sum(
                    len(draft) for draft in scheduled.draft_token_ids
                )
                if self._transactional_gdn:
                    if not isinstance(self.cache, HybridCache):
                        raise EngineError("transactional GDN cache is unavailable")
                    for request_id in scheduled.request_ids:
                        self.cache.gdn.copy_state(
                            self.cache_manager.gdn_state_indices(request_id),
                            self.cache_manager.gdn_state_indices(
                                request_id,
                                shadow=True,
                            ),
                        )
        elif (
            self.config.speculative_mode == "mtp"
            and not needs_score
            and scheduled.execution.all_decode
            and scheduled.execution.num_sequences == 1
            and scheduled.request_ids[0] not in self._media_requests
            and scheduled.request_ids[0] not in self._omni_captures
            and scheduled.request_ids[0] not in self._request_lora
        ):
            self.metrics.speculative_draft_attempts += 1
            request_id = scheduled.request_ids[0]
            draft_state = self._mtp_drafts.get(request_id)
            expected_position = scheduled.execution.context_lengths[0]
            if draft_state is None or draft_state[0] != expected_position:
                self._mtp_drafts.pop(request_id, None)
                self._mtp_ready.discard(request_id)
                self.metrics.speculative_no_draft += 1
            else:
                prepared = self.scheduler.prepare_mtp_speculative(
                    scheduled,
                    draft_token_id=draft_state[1],
                )
                if prepared is scheduled:
                    self.metrics.speculative_no_draft += 1
                else:
                    scheduled = prepared
                    self.metrics.speculative_draft_tokens += len(draft_state[1])
                    if isinstance(self.cache, HybridCache):
                        active_state = self.cache_manager.gdn_state_indices(request_id)
                        shadow_state = self.cache_manager.gdn_state_indices(
                            request_id,
                            shadow=True,
                        )
                        self.cache.gdn.copy_state(
                            active_state,
                            shadow_state,
                        )
        sampling, request_ids, histories, history_counts = (
            self.scheduler.sampling_metadata(scheduled)
        )
        scheduled_ns = time.perf_counter_ns()
        batch = self._materialize_multimodal_batch(scheduled)
        needs_logprobs = any(item is not None and item.logprobs for item in sampling)
        needs_top_logprobs = any(
            item is not None and item.top_logprobs > 0 for item in sampling
        )
        top_logprobs: tuple[tuple[tuple[int, float], ...] | None, ...] | None = None
        sampled_logprobs = None
        score_label_rows: tuple[int, ...] = ()
        score_logprobs: tensorplay.Tensor | None = None
        grammar_states = tuple(
            self._grammar.get(request_id) if request_id else None
            for request_id in request_ids
        )
        has_grammar = any(state is not None for state in grammar_states)
        # LoRA banks and per-token slots are dynamic state that the captured
        # base-weight graphs do not carry; take the eager path for such batches.
        has_lora = any(request_id in self._request_lora for request_id in request_ids)
        # A greedy CUDA graph captures argmax over raw logits; a constrained
        # batch must take the eager path so the bitmask can be applied first.
        use_greedy_graph = (
            self.graph_runner is not None
            and self.graph_runner.can_replay(batch)
            and self.sampler.can_use_greedy_graph(sampling)
            and not needs_logprobs
            and not has_grammar
            and not needs_score
            and not has_lora
        )
        chunk_steps = 1
        if has_lora:
            self.metrics.lora_steps += 1
            self.metrics.lora_tokens += batch.num_tokens
            if self.graph_runner is not None:
                # CUDA Graph capture froze base weights; LoRA batches must run
                # eagerly even when a matching bucket is available.
                self.metrics.lora_eager_steps += 1
            if not self.config.use_triton or self.device.type != "cuda":
                self.metrics.lora_fallback_steps += 1
        speculative_accepted = None
        conv_snapshots = None
        self._conv_snapshots = None
        if scheduled.is_speculative and self.model_spec.model_type in {
            "inkling2",
            "inkling2_loop",
        }:
            self._conv_snapshots = {
                request_id: self._snapshot_inkling_conv(
                    self.cache_manager.request_slot(request_id)
                )
                for request_id in scheduled.request_ids
            }
        if scheduled.is_speculative:
            sampling_param = sampling[0]
            if sampling_param is None:
                raise EngineError("speculative verification requires greedy sampling")
            # Graph capture bakes greedy sampling in; sampling modes
            # verify through the eager path with the real sampler.
            pure_greedy = all(
                item is not None
                and item.temperature == 0
                and item.repetition_penalty == 1.0
                for item in sampling
            )
            num_chains = batch.num_sequences
            chain_rows = scheduled.scheduled_tokens[0]
            speculative_rows = batch.num_tokens
            if (
                self._speculative_sampled is None
                or self._speculative_accepted is None
                or self._speculative_targets is None
                or self._speculative_sequence_lengths is None
                or num_chains > self._speculative_sampled.shape[0]
                or speculative_rows > self._speculative_sampled.shape[1]
            ):
                raise EngineError("speculative verification workspace is unavailable")
            self._speculative_sequence_lengths[:speculative_rows].copy_(batch.positions)
            self._speculative_sequence_lengths[:speculative_rows].add_(1)
            target_ids = self._speculative_targets[:num_chains, :speculative_rows]
            sampled_matrix = self._speculative_sampled[:num_chains, :speculative_rows]
            speculative_accepted = self._speculative_accepted[:num_chains]
            if num_chains == 1:
                draft_ids = batch.input_ids[1:].view(1, -1)
            else:
                draft_ids = tensorplay.tensor(
                    scheduled.draft_token_ids,
                    dtype=batch.input_ids.dtype,
                    device=batch.input_ids.device,
                )
            if (
                self.graph_runner is not None
                and pure_greedy
                and num_chains == 1
                and self.graph_runner.can_replay_speculative(batch)
            ):
                self.execution_stream.wait_stream(
                    tensorplay.cuda.current_stream(self.device)
                )
                with tensorplay.cuda.stream(self.execution_stream):
                    self.metrics.graph_steps += 1
                    target = self.graph_runner.run_speculative_greedy(
                        batch,
                    )
                    if self.config.speculative_mode == "mtp":
                        mtp_hidden = self._mtp_target_hidden
                        if mtp_hidden is None:
                            raise EngineError(
                                "MTP target hidden workspace is unavailable"
                            )
                        graph_hidden = self.graph_runner.speculative_hidden_states(
                            speculative_rows
                        )
                        if graph_hidden is None:
                            raise EngineError("MTP graph hidden capture is unavailable")
                        mtp_hidden[:speculative_rows].copy_(graph_hidden)
                    model_launched_ns = time.perf_counter_ns()
                    self.kernel_bindings.speculative.greedy_chain_verify(
                        draft_ids,
                        target.view(1, -1),
                        sampled_matrix,
                        speculative_accepted,
                    )
                    sample_ready_event = tensorplay.cuda.Event()
                    sample_ready_event.record(self.execution_stream)
                    sampler_launched_ns = time.perf_counter_ns()
            elif (
                self.graph_runner is not None
                and self.graph_runner.batch_sizes
                and pure_greedy
            ):
                self.execution_stream.wait_stream(
                    tensorplay.cuda.current_stream(self.device)
                )
                with tensorplay.cuda.stream(self.execution_stream):
                    for chain in range(num_chains):
                        for index in range(chain_rows):
                            decode_batch = self._speculative_decode_slice(
                                batch,
                                chain,
                                index,
                            )
                            self.metrics.graph_steps += 1
                            target = self.graph_runner.run_greedy(decode_batch)
                            if self.config.speculative_mode == "mtp":
                                self._capture_mtp_target_hidden(index)
                            target_ids[chain, index : index + 1].copy_(target[:1])
                    model_launched_ns = time.perf_counter_ns()
                    self.kernel_bindings.speculative.greedy_chain_verify(
                        draft_ids,
                        target_ids,
                        sampled_matrix,
                        speculative_accepted,
                    )
                    sample_ready_event = tensorplay.cuda.Event()
                    sample_ready_event.record(self.execution_stream)
                    sampler_launched_ns = time.perf_counter_ns()
            else:
                for chain in range(num_chains):
                    for index in range(chain_rows):
                        decode_batch = self._speculative_decode_slice(
                            batch,
                            chain,
                            index,
                        )
                        logits = self._execute_batch(decode_batch)
                        target = self.sampler.sample(
                            logits,
                            (sampling_param,),
                            (request_ids[chain],),
                            (scheduled.histories[chain],),
                            history_counts=(scheduled.history_counts[chain],),
                            sample_indices=decode_batch.sample_indices,
                        )
                        if self.config.speculative_mode == "mtp":
                            self._capture_mtp_target_hidden(index)
                        target_ids[chain, index : index + 1].copy_(target[:1])
                model_launched_ns = time.perf_counter_ns()
                self.kernel_bindings.speculative.greedy_chain_verify(
                    draft_ids,
                    target_ids,
                    sampled_matrix,
                    speculative_accepted,
                )
                sample_ready_event = None
                sampler_launched_ns = time.perf_counter_ns()
            sampled = sampled_matrix.reshape(-1)
        elif use_greedy_graph:
            assert self.graph_runner is not None
            # Relay inputs were populated on the caller's current stream.
            self.execution_stream.wait_stream(
                tensorplay.cuda.current_stream(self.device)
            )
            with tensorplay.cuda.stream(self.execution_stream):
                target_steps = min(
                    int(self.config.decode_chunk_steps),
                    self.graph_runner.loop_capacity_for(batch),
                )
                if target_steps > 1 and any(
                    request_id in self._omni_captures
                    for request_id in scheduled.request_ids
                ):
                    target_steps = 1
                if target_steps > 1:
                    chunk_steps = self.scheduler.reserve_decode_tokens(
                        scheduled,
                        target_steps,
                    )
                if chunk_steps > 1:
                    self.metrics.graph_steps += chunk_steps
                    sampled = self.graph_runner.run_greedy_loop(
                        batch,
                        chunk_steps,
                    )
                else:
                    chunk_steps = 1
                    self.metrics.graph_steps += 1
                    sampled = self.graph_runner.run_greedy(batch)
                sample_ready_event = tensorplay.cuda.Event()
                sample_ready_event.record(self.execution_stream)
                model_launched_ns = time.perf_counter_ns()
                sampler_launched_ns = model_launched_ns
        else:
            logits = self._execute_batch(batch, has_lora=has_lora)
            if has_grammar:
                self._apply_grammar_mask(logits, grammar_states)
            model_launched_ns = time.perf_counter_ns()
            if needs_score:
                score_label_rows, score_logprobs = self._read_score_logprobs(
                    logits, sampling
                )
            sampled = self._tp_sample(
                logits,
                sampling,
                request_ids,
                histories,
                history_counts=history_counts,
                sample_indices=batch.sample_indices,
                return_logprobs=needs_logprobs,
            )
            if needs_logprobs:
                sampled, sampled_logprobs = sampled
            else:
                sampled_logprobs = None
            if needs_top_logprobs:
                top_k = max(
                    item.top_logprobs
                    for item in sampling
                    if item is not None and item.top_logprobs > 0
                )
                top_logprobs = self.sampler.top_logprobs(
                    logits,
                    sampling,
                    histories,
                    history_counts,
                    top_k,
                )
            sample_ready_event = None
            sampler_launched_ns = time.perf_counter_ns()
        omni_input_embeddings = None
        omni_hidden_states = None
        if any(
            request_id in self._omni_captures for request_id in scheduled.request_ids
        ):
            if scheduled.is_speculative:
                raise EngineError(
                    "Omni audio output cannot use speculative verification"
                )
            if self.graph_runner is not None and batch.all_decode:
                (
                    omni_input_embeddings,
                    omni_hidden_states,
                ) = self.graph_runner.last_features()
            else:
                omni_input_embeddings = self.model.last_input_embeddings
                omni_hidden_states = self.model.last_hidden_states
        pending = PendingEngineStep(
            prior_events=prior_events,
            scheduled=scheduled,
            sampled=sampled,
            sample_ready_event=sample_ready_event,
            step_started_ns=step_started_ns,
            scheduled_ns=scheduled_ns,
            model_launched_ns=model_launched_ns,
            sampler_launched_ns=sampler_launched_ns,
            sampled_logprobs=sampled_logprobs,
            top_logprobs=top_logprobs,
            score_label_rows=score_label_rows,
            score_logprobs=score_logprobs,
            speculative_accepted=speculative_accepted,
            omni_input_embeddings=omni_input_embeddings,
            omni_hidden_states=omni_hidden_states,
            chunk_steps=chunk_steps,
        )
        self._pending_step = pending
        return pending

    @tensorplay.inference_mode()
    def finish_step(
        self,
        pending: PendingEngineStep,
    ) -> tuple[EngineEvent, ...]:
        """Synchronize one launched batch and commit its scheduler effects."""

        events, _ = self._finish_step(pending)
        return events

    @tensorplay.inference_mode()
    def finish_step_compact(
        self,
        pending: PendingEngineStep,
    ) -> Completion:
        """Commit one process-mode step without materializing EngineEvent."""

        if self.emit_engine_events:
            raise EngineError(
                "compact completion requires Engine event materialization disabled"
            )
        _, completion = self._finish_step(pending)
        return completion

    def _finish_step(
        self,
        pending: PendingEngineStep,
    ) -> tuple[tuple[EngineEvent, ...], Completion]:
        """Common step commit for embedded events and compact process output."""

        if pending is not self._pending_step:
            raise EngineError("finish_step does not match the in-flight step")
        finish_started_ns = time.perf_counter_ns()
        scheduled = pending.scheduled
        sampled = pending.sampled
        if scheduled is None:
            self._pending_step = None
            return pending.prior_events, Completion((), (), (), (), ())
        if sampled is None:
            self._pending_step = None
            raise EngineError("launched Engine step has no sampler output")
        if pending.sample_ready_event is not None:
            pending.sample_ready_event.synchronize()
        sampled_host = sampled.tolist()
        self._capture_omni_step(pending)
        sampled_ready_ns = time.perf_counter_ns()
        committed_steps = 1
        if scheduled.is_speculative:
            accepted = pending.speculative_accepted
            if accepted is None:
                self._pending_step = None
                raise EngineError("speculative step has no acceptance result")
            accepted_drafts = [int(value) for value in accepted.tolist()]
            num_chains = scheduled.execution.num_sequences
            chain_rows = scheduled.execution.num_tokens // num_chains
            verified_rows: list[tuple[int, ...]] = []
            for chain in range(num_chains):
                row_tokens = sampled_host[chain * chain_rows : (chain + 1) * chain_rows]
                verified_rows.append(
                    tuple(
                        int(token_id) for token_id in row_tokens if int(token_id) >= 0
                    )
                )
            conv_snapshots = getattr(self, "_conv_snapshots", None)
            if self._transactional_gdn or conv_snapshots is not None:
                for chain in range(num_chains):
                    self._commit_speculative_target_state(
                        scheduled,
                        row=chain,
                        accepted_drafts=accepted_drafts[chain],
                        conv_snapshot=(
                            None
                            if conv_snapshots is None
                            else conv_snapshots[scheduled.request_ids[chain]]
                        ),
                    )
            completion = self.scheduler.complete_speculative_host(
                scheduled,
                tuple(verified_rows),
                accepted_drafts=tuple(accepted_drafts),
            )
            if (
                self.config.speculative_mode == "mtp"
                and not completion.finished_request_ids
            ):
                self._advance_mtp_after_speculative(
                    scheduled,
                    list(verified_rows[0]),
                )
            self.metrics.speculative_steps += 1
            self.metrics.speculative_accepted_draft_tokens += sum(accepted_drafts)
            self.metrics.speculative_emitted_tokens += len(completion.output_token_ids)
        else:
            sampled_logprobs = pending.sampled_logprobs
            if pending.chunk_steps > 1:
                # The device-side self-loop produced the whole chunk before the
                # host saw it.  Commit one row at a time so every sequence keeps
                # its exact per-token lifecycle: a chunk stops at the first
                # token that finishes anything, and later rows are dropped.
                completions = []
                for row in sampled_host:
                    step_completion = self.scheduler.complete_host(
                        scheduled,
                        row,
                    )
                    completions.append(step_completion)
                    if step_completion.finished_request_ids:
                        break
                completion = _merge_completions(completions)
                committed_steps = len(completions)
            else:
                structural_done = self._advance_grammar_for_tokens(
                    scheduled,
                    sampled_host,
                )
                completion = self.scheduler.complete_host(
                    scheduled,
                    sampled_host,
                    (None if sampled_logprobs is None else sampled_logprobs.tolist()),
                    structurally_complete=structural_done,
                    score_logprobs=self._score_logprobs_host(pending),
                    top_logprobs=pending.top_logprobs,
                )
                if (
                    self.config.speculative_mode == "mtp"
                    and not completion.finished_request_ids
                ):
                    self._advance_mtp_after_normal(
                        scheduled,
                        sampled_host,
                    )
        # Constrained requests never take the speculative branch, so their
        # matchers were already advanced above; keep this for any residual
        # grammar state on speculative replicas (defensive no-op today).
        if scheduled.is_speculative:
            self._advance_grammar(completion)
        completed_ns = time.perf_counter_ns()
        completed_request_count = len(completion.finished_request_ids)
        for request_id in completion.finished_request_ids:
            self.sampler.release(request_id)
            self._grammar.release(request_id)
            self._release_media_state(request_id)
            self._release_lora(request_id)
            self._mtp_drafts.pop(request_id, None)
            self._mtp_ready.discard(request_id)
        events = pending.prior_events + self.scheduler.drain_events()
        # Deadline expiry calls the scheduler's own cancel(), which never
        # reaches Engine.cancel(). Release per-request runtime state from the
        # resulting events so expired structured requests cannot leak matchers.
        for event in events:
            if event.kind is EventKind.STATE_CHANGED and event.state in {
                RequestState.CANCELLED,
                RequestState.REJECTED,
            }:
                self.metrics.finish_reasons[event.state.value] += 1
                self.sampler.release(event.request_id)
                self._grammar.release(event.request_id)
                self._release_lora(event.request_id)
                self._mtp_drafts.pop(event.request_id, None)
                self._mtp_ready.discard(event.request_id)
        events, completion = self._defer_omni_output(
            events,
            completion,
        )
        self._sample_gpu_metrics()
        self.metrics.steps += committed_steps
        for reason in completion.finish_reasons:
            if reason is not None:
                self.metrics.finish_reasons[reason.value] += 1
        self.metrics.scheduled_tokens += (
            scheduled.execution.num_tokens * committed_steps
        )
        self.metrics.emitted_tokens += len(completion.output_token_ids)
        batch = scheduled.execution
        self.metrics.sequence_batch_histogram[batch.num_sequences] += committed_steps
        self.metrics.token_batch_histogram[batch.num_tokens] += committed_steps
        if scheduled.is_speculative:
            self.metrics.decode_steps += committed_steps
        elif batch.all_decode:
            self.metrics.decode_steps += committed_steps
            if self.graph_runner is not None:
                bucket_size = self.graph_runner.bucket_size_for(batch)
                if bucket_size is not None:
                    self.metrics.graph_bucket_histogram[bucket_size] += committed_steps
        elif any(phase is BatchPhase.DECODE for phase in batch.phases):
            self.metrics.mixed_steps += committed_steps
        else:
            self.metrics.prefill_steps += committed_steps
        step_finished_ns = time.perf_counter_ns()
        step_time_ns = step_finished_ns - pending.step_started_ns
        self.metrics.step_time_ns += step_time_ns
        self.metrics.scheduler_time_ns += pending.scheduled_ns - pending.step_started_ns
        self.metrics.model_launch_time_ns += (
            pending.model_launched_ns - pending.scheduled_ns
        )
        self.metrics.sampler_launch_time_ns += (
            pending.sampler_launched_ns - pending.model_launched_ns
        )
        self.metrics.overlap_window_time_ns += (
            finish_started_ns - pending.sampler_launched_ns
        )
        self.metrics.completion_time_ns += completed_ns - finish_started_ns
        self.metrics.sample_wait_time_ns += sampled_ready_ns - finish_started_ns
        self.metrics.scheduler_commit_time_ns += completed_ns - sampled_ready_ns
        if scheduled.is_speculative or batch.all_decode:
            self.metrics.decode_time_ns += step_time_ns
        elif any(phase is BatchPhase.DECODE for phase in batch.phases):
            self.metrics.mixed_time_ns += step_time_ns
        else:
            self.metrics.prefill_time_ns += step_time_ns
        if completed_request_count and _LOGGER.isEnabledFor(logging.INFO):
            cache_stats = self.cache_manager.stats()
            prefix_stats = cache_stats.prefix
            scheduler_stats = self.scheduler.metrics_snapshot()
            log_event(
                _LOGGER,
                logging.INFO,
                "request_batch_complete",
                completed_requests=completed_request_count,
                active_requests=scheduler_stats["active_requests"],
                running_requests=scheduler_stats["running_requests"],
                waiting_requests=scheduler_stats["waiting_requests"],
                batch_sequences=batch.num_sequences,
                batch_tokens=batch.num_tokens,
                step_time_ms=step_time_ns / 1_000_000,
                total_steps=self.metrics.steps,
                total_scheduled_tokens=self.metrics.scheduled_tokens,
                total_emitted_tokens=self.metrics.emitted_tokens,
                scheduled_token_throughput=(
                    0.0
                    if self.metrics.step_time_ns == 0
                    else (
                        self.metrics.scheduled_tokens
                        * 1_000_000_000
                        / self.metrics.step_time_ns
                    )
                ),
                emitted_token_throughput=(
                    0.0
                    if self.metrics.step_time_ns == 0
                    else (
                        self.metrics.emitted_tokens
                        * 1_000_000_000
                        / self.metrics.step_time_ns
                    )
                ),
                kv_cache_used_pages=cache_stats.allocated_pages,
                kv_cache_total_pages=cache_stats.total_pages,
                kv_cache_utilization=cache_stats.utilization,
                prefix_cache_enabled=prefix_stats is not None,
                prefix_cache_hits=(0 if prefix_stats is None else prefix_stats.hits),
                prefix_cache_lookups=(
                    0 if prefix_stats is None else prefix_stats.lookups
                ),
                prefix_cache_request_hit_rate=(
                    0.0 if prefix_stats is None else prefix_stats.request_hit_rate
                ),
                prefix_cache_hit_pages=(
                    0 if prefix_stats is None else prefix_stats.hit_pages
                ),
                prefix_cache_lookup_pages=(
                    0 if prefix_stats is None else prefix_stats.lookup_pages
                ),
                prefix_cache_token_hit_rate=(
                    0.0 if prefix_stats is None else prefix_stats.token_hit_rate
                ),
            )
        self._pending_step = None
        return events, completion

    def _capture_omni_step(self, pending: PendingEngineStep) -> None:
        scheduled = pending.scheduled
        inputs = pending.omni_input_embeddings
        hidden = pending.omni_hidden_states
        if scheduled is None or inputs is None or hidden is None:
            return
        batch = scheduled.execution
        if (
            inputs.shape != hidden.shape
            or inputs.ndim != 2
            or inputs.shape[0] < batch.num_tokens
        ):
            raise EngineError("Omni Thinker feature capture is invalid")
        media_ids = tensorplay.tensor(
            (
                self.model.config.image_token_id,
                self.model.config.video_token_id,
                self.model.config.audio_token_id,
            ),
            dtype=batch.input_ids.dtype,
            device=batch.input_ids.device,
        )
        for index, request_id in enumerate(scheduled.request_ids):
            capture = self._omni_captures.get(request_id)
            if capture is None:
                continue
            start = batch.query_start_loc[index]
            end = batch.query_start_loc[index + 1]
            token_ids = batch.input_ids[start:end]
            token_embeddings = inputs[start:end].clone()
            is_media = (token_ids[:, None] == media_ids[None, :]).any(dim=1)
            if bool(is_media.any().item()):
                token_embeddings[is_media] = 0
            capture.features.append(
                (hidden[start:end] + token_embeddings).detach().clone()
            )

    def _defer_omni_output(
        self,
        events: tuple[EngineEvent, ...],
        completion: Completion,
    ) -> tuple[tuple[EngineEvent, ...], Completion]:
        if not completion.finished_request_ids:
            return events, completion
        reasons = {
            request_id: reason
            for request_id, reason in zip(
                completion.output_request_ids,
                completion.finish_reasons,
                strict=True,
            )
            if reason is not None
        }
        deferred: set[str] = set()
        event_terminals = {
            event.request_id: event
            for event in events
            if event.kind is EventKind.TERMINAL
        }
        for request_id in completion.finished_request_ids:
            capture = self._omni_captures.pop(request_id, None)
            reason = reasons.get(request_id)
            if capture is None or reason not in {
                FinishReason.STOP,
                FinishReason.LENGTH,
            }:
                continue
            stage = self.omni_output
            if stage is None:
                continue
            terminal = event_terminals.get(request_id)
            next_sequence_no = (
                terminal.sequence_no if terminal is not None else len(capture.features)
            )
            self._omni_terminals[request_id] = _OmniPendingTerminal(
                finish_reason=reason,
                next_sequence_no=next_sequence_no,
            )
            try:
                bos_id, eos_id, pad_id = stage.text_embedding_ids(
                    capture.audio_output.voice
                )
                special_ids = tensorplay.tensor(
                    (bos_id, eos_id, pad_id),
                    dtype=tensorplay.long,
                    device=self.device,
                )
                special_embeddings = (
                    self.model.input_embeddings(special_ids).detach().clone()
                )
                stage.submit(
                    OmniSpeechTask(
                        request_id=request_id,
                        prompt_token_ids=capture.prompt_token_ids,
                        output_token_ids=self.scheduler.result_tokens(request_id),
                        thinker_features=tensorplay.cat(
                            capture.features,
                            dim=0,
                        ),
                        special_embeddings=special_embeddings,
                        voice=capture.audio_output.voice,
                        finish_reason=reason,
                        deadline_ns=capture.deadline_ns,
                    )
                )
            except BaseException as exc:
                self._omni_terminals.pop(request_id, None)
                self._omni_ready_events.append(
                    EngineEvent(
                        request_id=request_id,
                        sequence_no=next_sequence_no,
                        kind=EventKind.TERMINAL,
                        state=RequestState.FAILED,
                        monotonic_ns=time.monotonic_ns(),
                        finish_reason=FinishReason.ERROR,
                        error=ContractEngineError(
                            code="omni_output_failed",
                            message=str(exc) or type(exc).__name__,
                            retryable=False,
                        ),
                        schema_version=2,
                    )
                )
            deferred.add(request_id)
        if not deferred:
            return events, completion
        filtered_events = tuple(
            event
            for event in events
            if not (event.request_id in deferred and event.kind is EventKind.TERMINAL)
        )
        filtered_reasons = tuple(
            None if request_id in deferred else reason
            for request_id, reason in zip(
                completion.output_request_ids,
                completion.finish_reasons,
                strict=True,
            )
        )
        return filtered_events, replace(
            completion,
            events=tuple(
                event
                for event in completion.events
                if not (
                    event.request_id in deferred and event.kind is EventKind.TERMINAL
                )
            ),
            finish_reasons=filtered_reasons,
        )

    @property
    def has_pending_output(self) -> bool:
        return bool(
            self._omni_terminals
            or self._omni_ready_events
            or (self.omni_output is not None and self.omni_output.pending)
        )

    def drain_output_events(self) -> tuple[EngineEvent, ...]:
        events = self._omni_ready_events
        self._omni_ready_events = []
        stage = self.omni_output
        if stage is None:
            return tuple(events)
        for result in stage.drain():
            terminal = self._omni_terminals.pop(
                result.request_id,
                None,
            )
            if terminal is None:
                continue
            sequence_no = terminal.next_sequence_no
            for chunk in result.chunks:
                events.append(
                    EngineEvent(
                        request_id=result.request_id,
                        sequence_no=sequence_no,
                        kind=EventKind.AUDIO,
                        state=RequestState.OUTPUT_BLOCKED,
                        monotonic_ns=time.monotonic_ns(),
                        audio=chunk,
                        schema_version=2,
                    )
                )
                sequence_no += 1
            state = (
                RequestState.FAILED
                if result.error is not None
                else (
                    RequestState.CANCELLED
                    if result.finish_reason is FinishReason.CANCELLED
                    else RequestState.FINISHED
                )
            )
            events.append(
                EngineEvent(
                    request_id=result.request_id,
                    sequence_no=sequence_no,
                    kind=EventKind.TERMINAL,
                    state=state,
                    monotonic_ns=time.monotonic_ns(),
                    finish_reason=result.finish_reason,
                    error=result.error,
                    schema_version=2,
                )
            )
        return tuple(events)

    def close(self) -> None:
        if self.offload is not None:
            self.offload.close()
        if self.omni_output is not None:
            self.omni_output.close()
        self.tp_group.close()

    @tensorplay.inference_mode()
    def step(self) -> tuple[EngineEvent, ...]:
        """Run one step synchronously for the embedded Python SDK."""

        return self.finish_step(self.begin_step())

    def run_until_complete(
        self,
        request_ids: Sequence[str],
    ) -> tuple[GenerationResult, ...]:
        ids = tuple(request_ids)
        while not all(self.scheduler.is_finished(request_id) for request_id in ids):
            self.step()
        results = tuple(
            GenerationResult(
                request_id=request_id,
                token_ids=self.scheduler.result_tokens(request_id),
                text=self.tokenizer.decode(
                    self.scheduler.result_tokens(request_id),
                    skip_special_tokens=False,
                ),
                finish_reason=self.scheduler.result_finish_reason(request_id),
            )
            for request_id in ids
        )
        for request_id in ids:
            self.scheduler.forget(request_id)
        self.cache_manager.assert_integrity()
        self.vision_cache.assert_integrity()
        return results

    def generate_chat(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        sampling: SamplingParams | None = None,
        lora_request: LoRARequest | None = None,
    ) -> tuple[GenerationResult, ...]:
        request_ids = tuple(
            self.add_chat(
                [{"role": "user", "content": prompt}],
                max_new_tokens=max_new_tokens,
                sampling=sampling,
                lora_request=lora_request,
            )
            for prompt in prompts
        )
        return self.run_until_complete(request_ids)

    def _capture_mtp_target_hidden(self, index: int) -> None:
        target_hidden = self._mtp_target_hidden
        if target_hidden is None or index < 0 or index >= target_hidden.shape[0]:
            raise EngineError("MTP target hidden capture is out of range")
        target_hidden[index].copy_(self.model.last_hidden_states[0])

    @tensorplay.inference_mode()
    def _capture_prefix_state(
        self,
        request_id: str,
        token_count: int,
    ) -> dict[str, object] | None:
        """Copy one request's convolution rings to the host.

        The cache manager calls this right after a step that published exactly
        ``token_count`` tokens, so the rings hold the history a resumed request
        at that boundary must start from.  Speculative chains roll the rings
        back after publishing, so they are excluded until the commit path is
        checkpoint aware.
        """

        if self.config.speculative_mode != "off":
            return None
        arenas = getattr(self.model, "_conv", None)
        if not arenas:
            return None
        slot = self.cache_manager.request_slot(request_id)
        keys = tuple(arenas)
        rows = [arenas[key][slot] for key in keys]
        # One concatenated device->host copy keeps the capture at a single
        # stream sync instead of one per layer visit.
        flat = tensorplay.cat(
            [row.reshape(-1) for row in rows],
            dim=0,
        ).cpu()
        return {
            "keys": keys,
            "sizes": tuple(int(row.numel()) for row in rows),
            "rings": flat,
        }

    @tensorplay.inference_mode()
    def _restore_prefix_state(
        self,
        request_id: str,
        token_count: int,
        payload: object,
        slot: int,
    ) -> None:
        """Write checkpointed rings back into a freshly admitted slot."""

        arenas = getattr(self.model, "_conv", None)
        if not arenas or not isinstance(payload, dict):
            raise EngineError("prefix state payload does not match the model")
        keys = payload.get("keys")
        sizes = payload.get("sizes")
        rings = payload.get("rings")
        if not isinstance(keys, tuple) or not isinstance(sizes, tuple):
            raise EngineError("prefix state payload is malformed")
        missing = set(arenas) - set(keys)
        if missing:
            raise EngineError(
                f"prefix state is missing {len(missing)} convolution ring(s)"
            )
        flat = rings.to(device=self.device)
        offset = 0
        for key, size in zip(keys, sizes):
            arena = arenas.get(key)
            if arena is None:
                raise EngineError(f"prefix state references missing arena {key!r}")
            chunk = flat[offset : offset + size].view(
                arena.shape[1],
                arena.shape[2],
            )
            arena[slot].copy_(chunk)
            offset += size

    def _snapshot_inkling_conv(
        self, slot: int
    ) -> dict[tuple[int, str], tensorplay.Tensor]:
        model = self.model
        arenas = getattr(model, "_conv", {})
        return {key: arena[slot].clone() for key, arena in arenas.items()}

    def _restore_inkling_conv(
        self,
        slot: int,
        snapshot: dict[tuple[int, str], tensorplay.Tensor],
    ) -> None:
        arenas = getattr(self.model, "_conv", {})
        for key, value in snapshot.items():
            if key in arenas:
                arenas[key][slot].copy_(value)

    def _commit_speculative_target_state(
        self,
        scheduled: ScheduledBatch,
        *,
        row: int = 0,
        accepted_drafts: int,
        conv_snapshot: dict[tuple[int, str], tensorplay.Tensor] | None = None,
    ) -> None:
        """Commit a fully accepted GDN shadow or replay the exact prefix."""

        if not isinstance(self.cache, HybridCache):
            request_id = scheduled.request_ids[row]
            slot = self.cache_manager.request_slot(request_id)
            if conv_snapshot is None:
                raise EngineError("inkling2 MTP lost its convolution snapshot")
            self._restore_inkling_conv(slot, conv_snapshot)
            batch = self._materialize_multimodal_batch(scheduled)
            for index in range(accepted_drafts + 1):
                self._execute_batch(self._speculative_decode_slice(batch, row, index))
                if self.config.speculative_mode == "mtp":
                    self._capture_mtp_target_hidden(index)
            return
        request_id = scheduled.request_ids[row]
        draft_count = len(scheduled.draft_token_ids[row])
        if accepted_drafts == draft_count:
            self.cache_manager.swap_gdn_state(request_id)
            return
        if accepted_drafts < 0 or accepted_drafts >= draft_count:
            raise EngineError("MTP target acceptance count is invalid")

        active_state = tensorplay.tensor(
            (self.cache_manager.gdn_state_indices(request_id),),
            dtype=tensorplay.int32,
            device=self.device,
        )
        batch = self._materialize_multimodal_batch(scheduled)
        for index in range(accepted_drafts + 1):
            replay = replace(
                self._speculative_decode_slice(batch, row, index),
                gdn_state_indices=active_state,
            )
            self._execute_batch(replay)
            if self.config.speculative_mode == "mtp":
                self._capture_mtp_target_hidden(index)

    def _advance_mtp_after_normal(
        self,
        scheduled: ScheduledBatch,
        sampled_host: list[int],
    ) -> None:
        if (
            scheduled.execution.num_sequences != 1
            or scheduled.request_ids[0] in self._media_requests
            or not scheduled.should_sample[0]
            or len(sampled_host) != 1
            or sampled_host[0] < 0
        ):
            return
        sampling = scheduled.sampling[0]
        if (
            sampling is None
            or sampling.temperature != 0
            or sampling.repetition_penalty != 1.0
        ):
            return
        request_id = scheduled.request_ids[0]
        batch = scheduled.execution
        sampled_token = int(sampled_host[0])
        target_buffer = self.model.last_hidden_states
        if batch.all_decode:
            if request_id not in self._mtp_ready:
                return
            start_position = batch.context_lengths[0]
            mtp_tokens = (sampled_token,)
            target_hidden = target_buffer[:1]
        else:
            if batch.context_lengths[0] != 0:
                self._mtp_drafts.pop(request_id, None)
                self._mtp_ready.discard(request_id)
                return
            prompt_tail = tuple(
                int(token_id) for token_id in batch.input_ids[1:].tolist()
            )
            mtp_tokens = prompt_tail + (sampled_token,)
            start_position = 0
            target_hidden = target_buffer[: batch.num_tokens]
        self._run_mtp_update(
            request_id=request_id,
            target_batch=batch,
            input_token_ids=mtp_tokens,
            target_hidden_states=target_hidden,
            start_position=start_position,
        )

    def _advance_mtp_after_speculative(
        self,
        scheduled: ScheduledBatch,
        verified_tokens: list[int],
    ) -> None:
        target_hidden = self._mtp_target_hidden
        if target_hidden is None or not verified_tokens:
            raise EngineError("MTP speculative hidden state is unavailable")
        self._run_mtp_update(
            request_id=scheduled.request_ids[0],
            target_batch=scheduled.execution,
            input_token_ids=tuple(verified_tokens),
            target_hidden_states=target_hidden[: len(verified_tokens)],
            start_position=scheduled.execution.context_lengths[0],
        )

    def _run_mtp_update(
        self,
        *,
        request_id: str,
        target_batch: ExecutionBatch,
        input_token_ids: tuple[int, ...],
        target_hidden_states: tensorplay.Tensor,
        start_position: int,
    ) -> None:
        predictor = getattr(self.model, "mtp", None)
        if predictor is None or self.mtp_cache is None:
            raise EngineError("native MTP predictor is unavailable")
        token_count = len(input_token_ids)
        if self.model_spec.model_type in {"inkling2", "inkling2_loop"}:
            draft_count = int(os.environ.get("MEGALM_DRAFT_TOKENS", "7"))
            anchor_position = start_position + token_count - 1
            drafts = predictor.draft_block(
                input_token_ids[-1],
                target_hidden_states[-1],
                anchor_position,
                self.cache_manager.request_slot(request_id),
                count=draft_count,
            )
            self._mtp_ready.add(request_id)
            self._mtp_drafts[request_id] = (
                start_position + token_count,
                drafts,
            )
            return
        token_count = len(input_token_ids)
        if (
            token_count <= 0
            or target_hidden_states.shape[0] != token_count
            or token_count > target_batch.num_tokens
        ):
            raise EngineError("MTP update token and hidden rows do not align")
        positions = tensorplay.arange(
            start_position,
            start_position + token_count,
            dtype=tensorplay.int64,
            device=self.device,
        )
        phase = BatchPhase.DECODE if token_count == 1 else BatchPhase.PREFILL
        mtp_batch = ExecutionBatch(
            input_ids=tensorplay.tensor(
                input_token_ids,
                dtype=tensorplay.int64,
                device=self.device,
            ),
            positions=positions,
            slot_mapping=target_batch.slot_mapping[:token_count],
            block_tables=target_batch.block_tables[:1],
            sequence_lengths=tensorplay.tensor(
                (start_position + token_count,),
                dtype=tensorplay.int32,
                device=self.device,
            ),
            token_active=tensorplay.ones(
                token_count,
                dtype=tensorplay.bool,
                device=self.device,
            ),
            query_start_loc=(0, token_count),
            query_lengths=(token_count,),
            context_lengths=(start_position,),
            phases=(phase,),
            sample_indices=tensorplay.tensor(
                (token_count - 1,),
                dtype=tensorplay.int64,
                device=self.device,
            ),
            page_size=target_batch.page_size,
        )
        if self.mtp_graph_runner is not None and token_count in {1, 2}:
            sampled = self.mtp_graph_runner.run(
                mtp_batch,
                target_hidden_states,
            )
            draft_token = int(sampled[-1].item())
        else:
            if self.model_spec.model_type in {"inkling2", "inkling2_loop"}:
                features = target_hidden_states
                slot = self.cache_manager.request_slot(request_id)
                hidden_states = predictor(
                    mtp_batch,
                    target_hidden_states,
                    self.mtp_cache,
                    features=features,
                    slot=slot,
                )
            else:
                hidden_states = predictor(
                    mtp_batch,
                    target_hidden_states,
                    self.mtp_cache,
                )
            logits = predictor.compute_logits(
                hidden_states[-1:],
                self.model.lm_head(),
            )
            draft_token = int(tensorplay.argmax(logits, dim=-1).item())
        self._mtp_ready.add(request_id)
        self._mtp_drafts[request_id] = (
            start_position + token_count,
            draft_token,
        )

    def _sample_gpu_metrics(self) -> None:
        """Sample GPU utilization/memory at most once per second.

        Best-effort by design: NVML-backed calls may be unavailable on CPU
        builds or in containers without device access. A failure only leaves
        the gauges at zero and never affects inference.
        """

        now = time.monotonic_ns()
        if now - getattr(self, "_last_gpu_sample_ns", 0) < 1_000_000_000:
            return
        self._last_gpu_sample_ns = now
        cuda = getattr(tensorplay, "cuda", None)
        if cuda is None or not str(self.device).startswith("cuda"):
            return
        try:
            if hasattr(cuda, "utilization"):
                self.metrics.gpu_utilization = int(cuda.utilization(self.device))
            if hasattr(cuda, "memory_usage"):
                self.metrics.gpu_memory_utilization = int(
                    cuda.memory_usage(self.device)
                )
            if hasattr(cuda, "mem_get_info"):
                free, total = cuda.mem_get_info(self.device)
                self.metrics.gpu_memory_free_bytes = int(free)
                self.metrics.gpu_memory_total_bytes = int(total)
            if hasattr(cuda, "memory_allocated"):
                self.metrics.gpu_memory_allocated_bytes = int(
                    cuda.memory_allocated(self.device)
                )
        except Exception:  # pragma: no cover - device-dependent
            pass

    def _advance_grammar(self, completion: Completion) -> None:
        """Accept emitted tokens into each request's grammar matcher."""

        for request_id, token_id in zip(
            completion.output_request_ids,
            completion.output_token_ids,
        ):
            grammar = self._grammar.get(request_id)
            if grammar is None:
                continue
            if not grammar.accept_token(token_id):
                self.metrics.structured_rejected += 1
                log_event(
                    _LOGGER,
                    logging.WARNING,
                    "constrained_token_rejected",
                    request_id=request_id,
                    token_id=token_id,
                )
                self._grammar.release(request_id)

    def _advance_grammar_for_tokens(
        self,
        scheduled: Any,
        sampled_tokens: Sequence[int],
    ) -> frozenset[str]:
        """Advance matchers on sampled tokens before the scheduler commits.

        Doing this before ``complete_host`` lets us stop as soon as the grammar
        reaches a terminal state, instead of burning tokens until EOS or the
        length limit. Returns the request ids that are structurally complete.
        """

        structurally_complete: set[str] = set()
        for index, request_id in enumerate(scheduled.request_ids):
            if not scheduled.should_sample[index]:
                continue
            grammar = self._grammar.get(request_id)
            if grammar is None:
                continue
            token_id = int(sampled_tokens[index])
            if not grammar.accept_token(token_id):
                self.metrics.structured_rejected += 1
                log_event(
                    _LOGGER,
                    logging.WARNING,
                    "constrained_token_rejected",
                    request_id=request_id,
                    token_id=token_id,
                )
                self._grammar.release(request_id)
                continue
            if grammar.is_terminated():
                structurally_complete.add(request_id)
        return frozenset(structurally_complete)

    def _read_score_logprobs(
        self,
        logits: tensorplay.Tensor,
        sampling: tuple[Any, ...],
    ) -> tuple[tuple[int, ...], tensorplay.Tensor]:
        """Read raw next-token log-probs at the decision labels.

        This is the read side of Jev-style decisions: no token is generated,
        and no grammar/penalty mask is applied, so the values are a pure
        function of the model logits. The gateway converts them to label
        probabilities with ``softmax(logprobs / temperature)`` and computes
        ``label_mass`` from the unscaled values, as vLLM/SGLang do.
        """

        rows = tuple(
            index
            for index, item in enumerate(sampling)
            if item is not None and item.score_labels
        )
        if not rows:
            return (), tensorplay.zeros((0, 0), device=logits.device)
        width = max(len(sampling[index].score_labels) for index in rows)
        if any(
            label >= logits.shape[1]
            for index in rows
            for label in sampling[index].score_labels
        ):
            raise EngineError("score label id is outside the model vocabulary")
        label_ids = tensorplay.zeros(
            (logits.shape[0], width),
            dtype=tensorplay.long,
            device=logits.device,
        )
        for index in rows:
            labels = sampling[index].score_labels
            label_ids[index, : len(labels)] = tensorplay.tensor(
                list(labels),
                dtype=tensorplay.long,
                device=logits.device,
            )
        log_probs = tensorplay.log_softmax(logits.float(), dim=-1)
        return rows, tensorplay.gather(log_probs, 1, label_ids)

    @staticmethod
    def _score_logprobs_host(
        pending: PendingEngineStep,
    ) -> tuple[tuple[float, ...] | None, ...] | None:
        """Materialize per-row score log-probs after the step has synced."""

        if pending.score_logprobs is None:
            return None
        values = pending.score_logprobs.tolist()
        result: list[tuple[float, ...] | None] = [None] * len(
            pending.scheduled.request_ids
        )
        for row, source in enumerate(pending.score_label_rows):
            result[source] = tuple(float(value) for value in values[row])
        return tuple(result)

    def _apply_grammar_mask(
        self,
        logits: tensorplay.Tensor,
        grammar_states: tuple[Any, ...],
    ) -> None:
        """Fill each constrained row's xgrammar bitmask and apply it to logits.

        Rows whose bitmask already allows every token are skipped. ``need_apply``
        from xgrammar lets us avoid both the host-to-device copy and the kernel
        launch for the common unconstrained-prefix case.
        """

        import numpy as np

        active = [
            (row, grammar)
            for row, grammar in enumerate(grammar_states)
            if grammar is not None
        ]
        if not active:
            return
        bitmask = self._grammar.bitmask_buffer(len(active))
        selected: list[tuple[int, int]] = []
        for slot, (row, grammar) in enumerate(active):
            if grammar.fill_bitmask(bitmask, slot):
                selected.append((row, slot))
            elif grammar.failed and not grammar.error_counted:
                grammar.error_counted = True
                self.metrics.structured_grammar_errors += 1
        if not selected:
            return
        packed = np.asarray(bitmask)[[slot for _, slot in selected]]
        mask_device = tensorplay.tensor(
            np.ascontiguousarray(packed),
            dtype=tensorplay.int32,
            device=logits.device,
        )
        rows_device = tensorplay.tensor(
            [row for row, _ in selected],
            dtype=tensorplay.int32,
            device=logits.device,
        )
        apply_grammar_bitmask(
            logits,
            mask_device,
            rows_device,
            vocab_size=logits.shape[1],
            grammar_vocab=self._grammar.vocab_size,
        )
        self.metrics.structured_mask_steps += 1
        self.metrics.structured_masked_rows += len(selected)

    def _tp_sample(self, *args: Any, **kwargs: Any):
        """Leader samples and broadcasts tokens; followers only receive them.

        Host RNG state is not shared across ranks, so sampling independently
        would break lockstep for stochastic requests. Greedy graph paths do not
        reach this method: their argmax is captured in the graph and stays
        deterministic after the logits all-reduce.
        """

        if not self.tp_group.enabled:
            return self.sampler.sample(*args, **kwargs)
        if self.tp_group.rank == 0:
            sampled = self.sampler.sample(*args, **kwargs)
            payload = sampled.tolist() if hasattr(sampled, "tolist") else list(sampled)
            self.tp_group.broadcast_object(payload)
            return sampled
        payload = self.tp_group.broadcast_object(None)
        return tensorplay.tensor(payload, dtype=tensorplay.int64, device=self.device)

    def _execute_batch(
        self,
        batch: ExecutionBatch,
        *,
        has_lora: bool = False,
    ) -> tensorplay.Tensor:
        if (
            self.graph_runner is not None
            and not has_lora
            and self.graph_runner.can_replay(batch)
        ):
            self.metrics.graph_steps += 1
            return self.graph_runner.run(batch)
        self.metrics.eager_steps += 1
        if self.graph_runner is not None and batch.all_decode:
            self.graph_runner.metrics.eager_fallbacks += 1
        return self.model(batch, self.cache)

    def _speculative_decode_slice(
        self,
        batch: ExecutionBatch,
        chain: int,
        index: int,
    ) -> ExecutionBatch:
        """View one reserved draft position through the exact decode ABI."""

        sequence_lengths = self._speculative_sequence_lengths
        token_index = batch.query_start_loc[chain] + index
        if (
            not batch.is_speculative
            or sequence_lengths is None
            or chain < 0
            or chain >= batch.num_sequences
            or index < 0
            or token_index >= batch.num_tokens
        ):
            raise EngineError("speculative decode slice is invalid")
        mrope_positions = (
            None
            if batch.mrope_positions is None
            else batch.mrope_positions[:, token_index : token_index + 1]
        )
        return ExecutionBatch(
            input_ids=batch.input_ids[token_index : token_index + 1],
            positions=batch.positions[token_index : token_index + 1],
            slot_mapping=batch.slot_mapping[token_index : token_index + 1],
            block_tables=batch.block_tables[chain : chain + 1],
            sequence_lengths=sequence_lengths[token_index : token_index + 1],
            token_active=batch.token_active[token_index : token_index + 1],
            query_start_loc=(0, 1),
            query_lengths=(1,),
            context_lengths=(int(batch.context_lengths[chain]) + index,),
            phases=(BatchPhase.DECODE,),
            sample_indices=batch.sample_indices[:1],
            page_size=batch.page_size,
            gdn_state_indices=(
                None
                if batch.gdn_state_indices is None
                else batch.gdn_state_indices[chain : chain + 1]
            ),
            state_slots=(
                None
                if batch.state_slots is None
                else batch.state_slots[chain : chain + 1]
            ),
            mrope_positions=mrope_positions,
            validate_contract=False,
        )

    def _materialize_multimodal_batch(
        self,
        scheduled: ScheduledBatch,
    ) -> ExecutionBatch:
        batch = scheduled.execution
        # Pure decode of a non-MRoPE multimodal model (e.g. OXR) needs neither
        # cached visual embeddings nor 3-D media positions: generated tokens
        # are ordinary token ids and use the model's 1-D RoPE cache. Skipping
        # the materializer here avoids three host syncs per decode step.
        if batch.all_decode and not getattr(self.model, "uses_mrope", False):
            return batch
        states = tuple(
            self._media_requests.get(request_id) for request_id in scheduled.request_ids
        )
        if not any(state is not None for state in states):
            if (
                bool(getattr(self.model, "uses_mrope", False))
                and batch.mrope_positions is None
            ):
                return replace(
                    batch,
                    mrope_positions=(batch.positions.unsqueeze(0).expand(3, -1)),
                )
            return batch
        mrope_positions = batch.positions.unsqueeze(0).expand(3, -1).clone()
        embedding_rows: list[int] = []
        feature_rows: list[tensorplay.Tensor] = []
        media_row_lookup = {
            request_id: {
                position: source
                for position, source in zip(
                    state.media_positions,
                    state.media_sources,
                )
            }
            for request_id, state in zip(scheduled.request_ids, states)
            if state is not None
        }
        for sequence_index, (request_id, state) in enumerate(
            zip(scheduled.request_ids, states)
        ):
            if state is None:
                continue
            start = batch.query_start_loc[sequence_index]
            end = batch.query_start_loc[sequence_index + 1]
            absolute_positions = batch.positions[start:end]
            prompt_length = len(state.token_type_ids)
            prompt_mask = absolute_positions < prompt_length
            if bool(prompt_mask.any().item()):
                prompt_positions = absolute_positions[prompt_mask].long()
                target_rows = tensorplay.arange(
                    start,
                    end,
                    device=self.device,
                )[prompt_mask]
                mrope_positions[:, target_rows] = state.mrope_positions.index_select(
                    1, prompt_positions
                )
            generated_mask = ~prompt_mask
            if bool(generated_mask.any().item()):
                target_rows = tensorplay.arange(
                    start,
                    end,
                    device=self.device,
                )[generated_mask]
                generated = absolute_positions[generated_mask] + state.rope_delta
                mrope_positions[:, target_rows] = generated.unsqueeze(0)
            lookup = media_row_lookup[request_id]
            for packed_row, absolute in zip(
                range(start, end),
                absolute_positions.tolist(),
            ):
                source = lookup.get(int(absolute))
                if source is None:
                    continue
                media_index, feature_index = source
                embedding_rows.append(packed_row)
                feature_rows.append(state.media_embeddings[media_index][feature_index])
        deepstack_features: tuple[tensorplay.Tensor, ...] = ()
        if feature_rows:
            depth = getattr(self.model, "deepstack_depth", 0)
            hidden_size = getattr(self.model, "hidden_size", 0)
            if depth:
                if hidden_size <= 0:
                    raise RuntimeError("deepstack model must declare its hidden size")
                expected_width = hidden_size * (1 + depth)
                if any(row.shape[-1] != expected_width for row in feature_rows):
                    raise RuntimeError(
                        "vision features do not match deepstack geometry"
                    )
                deepstack_features = tuple(
                    tensorplay.stack(
                        [
                            row[
                                ...,
                                (index + 1) * hidden_size : (index + 2) * hidden_size,
                            ]
                            for row in feature_rows
                        ]
                    )
                    for index in range(depth)
                )
                feature_rows = [row[..., :hidden_size] for row in feature_rows]
        input_embeddings = None
        deepstack_embeddings: tuple[tensorplay.Tensor, ...] = ()
        if embedding_rows:
            input_embeddings = self.model.input_embeddings(batch.input_ids)
            row_indices = tensorplay.tensor(
                embedding_rows,
                dtype=tensorplay.long,
                device=self.device,
            )
            input_embeddings.index_copy_(
                0,
                row_indices,
                tensorplay.stack(feature_rows).to(input_embeddings.dtype),
            )
            copied: list[tensorplay.Tensor] = []
            for chunk in deepstack_features:
                embeddings = tensorplay.zeros_like(input_embeddings)
                embeddings.index_copy_(
                    0,
                    row_indices,
                    chunk.to(input_embeddings.dtype),
                )
                copied.append(embeddings)
            deepstack_embeddings = tuple(copied)
        return replace(
            batch,
            mrope_positions=mrope_positions,
            input_embeddings=input_embeddings,
            deepstack_embeddings=deepstack_embeddings,
        )
