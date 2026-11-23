"""Inkling-2 native execution: depth residuals, hybrid attention, loop and MoE.

This is the first-party executor behind ``Inkling2ForCausalLM``.  It keeps the
routed experts in their checkpoint precision (NVFP4 / MXFP8) and reproduces the
checkpoint's arithmetic:

* depth-wise attention residuals: every sublayer reads a softmax mixture over
  the completed residual blocks plus the running partial, scored by learned
  pseudo-queries that are RMS-normalised per source;
* hybrid attention: local layers share one input, global layers take separate
  Q/K/V streams, both with learned relative position bias, L2-clipped heads,
  a per-head gate and a zero-value attention sink;
* a looped core (layers 6..17 x3): each pass after the first rewinds to the
  core entry and gates the previous pass's blocks into every core input through
  a small message controller;
* 128-expert top-8 routing with sigmoid scores, a normalised route scale and
  capped SwiGLU experts.

The loop materialises the 48 layer visits the model card describes: prelude
0..5 once, core 6..17 three times, coda 18..23 once.  Each visit owns its KV
cache layer and its short-convolution state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import os
import time

import tensorplay
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.kernels.inkling2_depth_residual import depth_residual_mix
from megalm.kernels.inkling2_head_prep import prepare_inkling2_heads
from megalm.kernels.inkling2_paged_decode import (
    MAX_PAGED_DECODE_TOKENS,
    inkling2_paged_decode,
)
from megalm.kernels.inkling2_sconv import short_convolution_decode
from megalm.kernels.moe_grouped_int4 import (
    Int4MoEWorkspace,
    pack_int4_expert_rowmajor,
    routed_moe_grouped_int4,
    stack_int4_experts,
)
from megalm.kernels.ops import KernelPolicy
from megalm.kernels.qwen35 import active_kv_write
from megalm.kernels.quant_fp import (
    int4_weight_mm,
    pack_int4_weight,
    unpack_mxfp8,
    unpack_nvfp4,
)
from megalm.kernels.silu_mul import capped_swiglu_cuda
from megalm.loading import WeightLoadReport
from megalm.models.inkling2.loader import (
    Inkling2Config,
    Inkling2Layer,
    load_inkling2_config,
)
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache

Tensor = tensorplay.Tensor

# The order of the four loop-message inputs of a core layer.
LOOP_INPUTS = ("attn", "k", "v", "mlp")


class Inkling2Error(RuntimeError):
    """Raised when a checkpoint fact or tensor violates the implemented contract."""


# ---------------------------------------------------------------------------
# Elementwise helpers
# ---------------------------------------------------------------------------


def _mean_last(x: Tensor) -> Tensor:
    return tensorplay.sum(x, dim=-1, keepdim=True) / float(x.shape[-1])


def rms_norm(
    x: Tensor,
    weight: Tensor | None = None,
    eps: float = 1e-6,
    *,
    output_dtype: tensorplay.dtype | None = None,
) -> Tensor:
    """Unweighted or learned RMSNorm over the last axis, computed in FP32."""

    values = x.float()
    inverse = tensorplay.rsqrt(_mean_last(values * values) + eps)
    normed = values * inverse
    if weight is not None:
        normed = normed * weight.float()
    return normed.to(output_dtype if output_dtype is not None else x.dtype)


def l2_clip(x: Tensor, threshold: float, eps: float = 1e-6) -> Tensor:
    """Scale each row down to ``threshold`` when its L2 norm exceeds it."""

    values = x.float()
    norm = tensorplay.sqrt(tensorplay.sum(values * values, dim=-1, keepdim=True))
    coefficient = tensorplay.minimum(
        threshold / (norm + eps),
        tensorplay.full_like(norm, 1.0),
    )
    return (values * coefficient).to(x.dtype)


def log_scale_tau(positions: Tensor, floor: int, alpha: float) -> Tensor:
    """``1 + alpha * log(max((position + 1) / floor, 1))`` per position."""

    ratio = (positions.float() + 1.0) / float(floor)
    ratio = tensorplay.maximum(ratio, tensorplay.full_like(ratio, 1.0))
    return 1.0 + alpha * tensorplay.log(ratio)


def softcap(x: Tensor, cap: float) -> Tensor:
    """``cap * tanh(x / cap)``; a non-positive cap disables the cap."""

    if cap <= 0:
        return x
    limit = tensorplay.full_like(x, cap)
    return tensorplay.tanh(x / limit) * limit


def capped_swiglu(gate: Tensor, up: Tensor, gate_cap: float, up_cap: float) -> Tensor:
    """Capped SwiGLU: the sigmoid sees the raw gate, the caps bound both arms."""

    gate_f = gate.float()
    up_f = up.float()
    return (
        softcap(gate_f, gate_cap)
        * tensorplay.sigmoid(gate_f)
        * softcap(up_f, up_cap)
    )


def residual_sum(terms: tuple[Tensor, ...]) -> Tensor:
    """Sum same-shaped sources in FP32, then round once to the first dtype."""

    if not terms:
        raise ValueError("residual sum needs at least one input")
    total = terms[0].float()
    for term in terms[1:]:
        total = total + term.float()
    return total.to(terms[0].dtype)


# ---------------------------------------------------------------------------
# Depth attention residual state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoopEntry:
    num_blocks: int


class AttentionResidualState:
    """Completed depth blocks plus the running partial of one forward call.

    ``blocks`` is a ``[max_blocks, tokens, hidden]`` arena.  ``num_blocks``
    completed blocks occupy ``blocks[:num_blocks]``; ``blocks[num_blocks]`` is
    the temporary slot a block boundary rounds the current partial into.
    """

    eps = 1e-6

    def __init__(
        self,
        config: Inkling2Config,
        embedding: Tensor,
        output_query: Tensor,
    ) -> None:
        self.config = config
        self.blocks = tensorplay.zeros(
            (
                config.num_hidden_layers // config.n_layers_per_attn_res_block + 2,
                embedding.shape[0],
                config.hidden_size,
            ),
            dtype=embedding.dtype,
            device=embedding.device,
        )
        self.num_blocks = 0
        self.block_stored = False
        self.terms: tuple[Tensor, ...] = (embedding,)
        self.output_queries = output_query.reshape(1, -1)

    # -- block bookkeeping -------------------------------------------------

    def is_block_boundary(self, layer: int) -> bool:
        return layer % self.config.n_layers_per_attn_res_block == 0

    @property
    def partial(self) -> Tensor:
        if len(self.terms) > 1:
            self.terms = (residual_sum(self.terms),)
        return self.terms[0]

    # -- mixing ------------------------------------------------------------

    def _candidate(self, terms: tuple[Tensor, ...], candidate_out: Tensor | None) -> Tensor:
        total = terms[0].float()
        for term in terms[1:]:
            total = total + term.float()
        if candidate_out is None:
            return total
        candidate_out.copy_(total.to(candidate_out.dtype))
        return candidate_out.float()

    def _mix(
        self,
        queries: Tensor,
        num_blocks: int,
        terms: tuple[Tensor, ...],
        candidate_out: Tensor | None,
        norms: tuple[Tensor, ...] | None,
    ) -> tuple[Tensor, ...]:
        """Mix ``queries`` over the completed blocks plus the current candidate."""

        blocks = self.blocks[:num_blocks]
        mixed = depth_residual_mix(
            blocks,
            queries,
            terms,
            candidate_out,
            norms,
            normalize_values=self.config.attn_res_normalize_values,
            eps=self.eps,
        )
        if mixed is not None:
            return tuple(mixed[index] for index in range(mixed.shape[0]))

        candidate = self._candidate(terms, candidate_out)
        sources = tensorplay.cat(
            (blocks.float(), candidate.unsqueeze(0)),
            dim=0,
        )
        normalized = rms_norm(
            sources, None, self.eps, output_dtype=tensorplay.float32
        )
        # [queries, sources, tokens]
        logits = tensorplay.einsum("qh,sth->qst", queries.float(), normalized)
        probabilities = tensorplay.softmax(logits, dim=1)
        values = normalized if self.config.attn_res_normalize_values else sources
        mixed = tensorplay.einsum("qst,sth->qth", probabilities, values)
        if norms is None:
            return tuple(mixed[index] for index in range(mixed.shape[0]))
        return tuple(
            rms_norm(
                mixed[index],
                norms[index],
                self.eps,
                output_dtype=norms[index].dtype,
            )
            for index in range(mixed.shape[0])
        )

    def _candidate_out(self, boundary: bool) -> Tensor | None:
        if boundary:
            return None if self.block_stored else self.blocks[self.num_blocks]
        if len(self.terms) > 1:
            return tensorplay.empty_like(self.terms[0])
        return None

    def attention_input(
        self,
        layer: int,
        queries: Tensor,
        norms: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """The normalised attention inputs of ``layer``."""

        boundary = self.is_block_boundary(layer)
        if self.num_blocks == 0:
            partial = self.partial
            if boundary and not self.block_stored:
                self.blocks[0].copy_(partial)
                self.block_stored = True
            return tuple(
                rms_norm(partial, norm, self.eps, output_dtype=norm.dtype)
                for norm in norms
            )
        candidate_out = self._candidate_out(boundary)
        mixed = self._mix(queries, self.num_blocks, self.terms, candidate_out, norms)
        if candidate_out is not None:
            self.terms = (candidate_out,)
        self.block_stored = boundary
        return mixed

    def attention_mixtures(self, layer: int, queries: Tensor) -> tuple[Tensor, ...]:
        """The FP32 attention mixtures before the input norms."""

        boundary = self.is_block_boundary(layer)
        if self.num_blocks == 0:
            return (self.partial.float(),) * queries.shape[0]
        candidate_out = self._candidate_out(boundary)
        mixed = self._mix(queries, self.num_blocks, self.terms, candidate_out, None)
        if candidate_out is not None:
            self.terms = (candidate_out,)
        self.block_stored = boundary
        return mixed

    def mlp_input(
        self,
        layer: int,
        attn_output: Tensor,
        query: Tensor,
        norm: Tensor,
    ) -> Tensor:
        """The normalised MoE input: the partial plus ``attn_output``."""

        boundary = self.is_block_boundary(layer)
        if boundary and not self.block_stored:
            self.blocks[self.num_blocks].copy_(self.partial)
            self.block_stored = True
        terms = (attn_output,) if boundary else (self.partial, attn_output)
        return self._mix(
            query.reshape(1, -1),
            self.num_blocks + int(boundary),
            terms,
            None,
            (norm,),
        )[0]

    def mlp_mixture(self, layer: int, attn_output: Tensor, query: Tensor) -> Tensor:
        """The FP32 MoE mixture before the MoE norm."""

        boundary = self.is_block_boundary(layer)
        if boundary and not self.block_stored:
            self.blocks[self.num_blocks].copy_(self.partial)
            self.block_stored = True
        terms = (attn_output,) if boundary else (self.partial, attn_output)
        return self._mix(
            query.reshape(1, -1),
            self.num_blocks + int(boundary),
            terms,
            None,
            None,
        )[0]

    def finalize_layer(
        self,
        layer: int,
        attn_output: Tensor,
        mlp_output: Tensor,
    ) -> None:
        if self.is_block_boundary(layer):
            if not self.block_stored:
                self.blocks[self.num_blocks].copy_(self.partial)
            self.num_blocks += 1
            self.block_stored = False
            self.terms = (attn_output, mlp_output)
        else:
            self.terms = (self.partial, attn_output, mlp_output)

    # -- loop replay -------------------------------------------------------

    def loop_entry(self, layer: int) -> LoopEntry:
        if not self.is_block_boundary(layer) or self.block_stored:
            raise Inkling2Error("a looped core must start at an unstored block boundary")
        return LoopEntry(self.num_blocks)

    def loop_bank(self, entry: LoopEntry) -> tuple[Tensor, Tensor]:
        stored = self.blocks[entry.num_blocks + 1 : self.num_blocks].clone()
        return stored, self.partial

    def rewind_loop(self, entry: LoopEntry) -> None:
        self.num_blocks = entry.num_blocks
        self.terms = (self.blocks[entry.num_blocks],)
        self.block_stored = True

    def compute_output(self, norm: Tensor) -> Tensor:
        return self._mix(
            self.output_queries,
            self.num_blocks,
            self.terms,
            tensorplay.empty_like(self.terms[0]) if len(self.terms) > 1 else None,
            (norm,),
        )[0]


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LayerWeights:
    """Every checkpoint tensor of one decoder layer, already shaped."""

    index: int
    is_local: bool
    attn_norm: Tensor
    q_norm: Tensor | None
    k_norm: Tensor | None
    v_norm: Tensor | None
    mlp_norm: Tensor
    wq: Tensor
    wk: Tensor
    wv: Tensor
    wr: Tensor
    wo: Tensor
    gate: Tensor
    sink: Tensor
    q_clip: Tensor
    rel_clip: Tensor
    rel_proj: Tensor
    k_sconv: Tensor
    v_sconv: Tensor
    k_sconv_proj: Tensor | None
    v_sconv_proj: Tensor | None
    attn_sconv: Tensor
    mlp_sconv: Tensor
    gate_weight: Tensor
    gate_bias: Tensor
    w13: Tensor
    w13_scale: Tensor
    w13_amax: Tensor | None
    w13_denominator: Tensor | None
    w2: Tensor
    w2_scale: Tensor
    w2_amax: Tensor | None
    w2_denominator: Tensor | None
    controllers: dict[str, tuple[Tensor, Tensor, Tensor]] | None
    stacked_loop_controllers: tuple[Tensor, Tensor, Tensor] | None = None
    # int4 weight-only experts (tensorplay native): one packed B-fragment
    # tensor plus ``[K/groups, N, 2]`` scale/zero metadata per expert.
    w13_packed: list[Tensor] | None = None
    w13_packed_scale: list[Tensor] | None = None
    w2_packed: list[Tensor] | None = None
    w2_packed_scale: list[Tensor] | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Inkling2RuntimeConfig:
    """The subset of the model contract the engine reads off ``model.config``."""

    def __init__(self, config: Inkling2Config) -> None:
        self.model_type = config.model_type
        self.architecture = "Inkling2LoopForConditionalGeneration"
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        # One KV cache layer per layer visit: the loop replays its core.
        self.num_hidden_layers = config.total_layer_visits
        self.num_key_value_heads = config.layers[0].num_kv_heads
        self.head_dim = config.layers[0].head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.num_experts = config.layers[0].num_experts
        self.num_experts_per_tok = config.layers[0].experts_per_token
        self.rms_norm_eps = config.rms_norm_eps


_CAPTURE_ACTIVE = False


def set_moe_capture_active(active: bool) -> None:
    """Toggle the graph-capture-safe MoE path (used by the graph runner)."""

    global _CAPTURE_ACTIVE
    _CAPTURE_ACTIVE = bool(active)


class Inkling2ForCausalLM(tensorplay.nn.Module):
    """Native Inkling-2 executor with quantized-resident experts."""

    # Decode uses GPU gather/scatter on per-request convolution state, so the
    # hot path is CUDA-graph capturable; prefill stays eager.
    graph_capable = True

    def set_capture_active(self, active: bool) -> None:
        set_moe_capture_active(active)

    def __init__(
        self,
        config: Inkling2Config,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
        enable_mtp: bool = False,
    ) -> None:
        super().__init__()
        self.inkling = config
        self.config = Inkling2RuntimeConfig(config)
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.kernel_policy = kernel_policy
        self.kernel_bindings = kernel_bindings
        self.mtp: object | None = None
        self.enable_mtp = bool(enable_mtp)
        self._mtp_features: Tensor | None = None
        self._stacked_int4: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        self._int4_workspace: Int4MoEWorkspace | None = None
        self.hidden = config.hidden_size
        self.eps = config.rms_norm_eps
        self.visits = config.layer_visits()
        self.core_start = config.loop_core_start
        self.core_end = config.loop_core_end
        self.num_passes = config.loop_num_passes
        # Diagnostic switch: replay the core fewer/more times without touching
        # the checkpoint contract.
        override = int(os.environ.get("MEGALM_INK_PASSES", "0"))
        if override > 0 and override != self.num_passes:
            self.num_passes = override
            visits = [(layer, 0) for layer in range(self.core_start)]
            for pass_index in range(override):
                visits.extend(
                    (layer, pass_index)
                    for layer in range(self.core_start, self.core_end)
                )
            visits.extend(
                (layer, 0) for layer in range(self.core_end, config.num_hidden_layers)
            )
            self.visits = tuple(visits)
        self.weights: dict[str, Tensor] = {}
        self.layers: list[LayerWeights] = []
        self._conv: dict[tuple[int, str], Tensor] = {}
        self.num_state_slots = int(os.environ.get("MEGALM_STATE_SLOTS", "32"))
        self._expert_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        self._expert_cache_order: list[tuple[int, int]] = []
        self.expert_cache_size = int(os.environ.get("MEGALM_EXPERT_CACHE", "0"))
        # Expert-projection layout switches, kept as attributes so a numeric
        # comparison against the reference can flip them without a rebuild.
        self.w13_transposed = os.environ.get("MEGALM_W13_TRANSPOSE", "1") != "0"
        self.w13_interleaved = os.environ.get("MEGALM_W13_INTERLEAVE", "1") != "0"
        # Attention logits are ``q . k / scale``; the checkpoint family uses
        # ``sqrt(head_dim)`` in the reference implementation.
        self.attn_scale = os.environ.get("MEGALM_ATTN_SCALE", "sqrt")
        self.w_denominator = float(os.environ.get("MEGALM_W_DENOM", "1.0"))
        self.debug = os.environ.get("MEGALM_DEBUG_INK", "0") != "0"
        self.profile = os.environ.get("MEGALM_PROFILE_INK", "0") != "0"
        self._prof: dict[str, float] = {}
        self._prof_calls: dict[str, int] = {}
        # Routed experts live as int4 weight-only GEMMs: tensorplay's native
        # pack/unpack runs the per-group dequant inside the kernel, which is
        # ~10x cheaper than decoding the NVFP4 slabs in Python per use.
        self.int4_experts = os.environ.get("MEGALM_INT4_EXPERTS", "1") != "0"
        self.int4_group = int(os.environ.get("MEGALM_INT4_GROUP", "128"))
        self.int4_bytes = 0
        self.attn_queries: dict[int, Tensor] = {}
        self.mlp_queries: list[Tensor] = []
        self.output_query: Tensor | None = None
        self.loop_score_weights: Tensor | None = None

    # -- module plumbing ---------------------------------------------------

    def parameters(self):  # noqa: D102 - the arena is owned through buffers()
        return ()

    def buffers(self):  # noqa: D102 - every checkpoint tensor is a buffer
        return tuple(self.weights.values())

    def named_parameters(self):  # noqa: D102
        return iter(())

    def named_buffers(self):  # noqa: D102
        return iter(self.weights.items())

    def eval(self):  # noqa: D102
        return self

    def train(self, mode: bool = True):  # noqa: D102
        return self

    # -- allocation --------------------------------------------------------

    def _alloc(
        self,
        name: str,
        shape: tuple[int, ...],
        tensor_dtype: tensorplay.dtype,
    ) -> Tensor:
        buffer = tensorplay.zeros(shape, dtype=tensor_dtype, device=self.device)
        self.weights[name] = buffer
        return buffer

    def allocate_weights(self) -> None:
        """Allocate every resident tensor once, in checkpoint precision."""

        config = self.inkling
        hidden = self.hidden
        self._alloc(
            "embed.weight", (config.padded_vocab_size, hidden), tensorplay.bfloat16
        )
        self._alloc("embed_norm.weight", (hidden,), tensorplay.bfloat16)
        self._alloc("norm.weight", (hidden,), tensorplay.bfloat16)
        self._alloc(
            "unembed.weight", (config.padded_vocab_size, hidden), tensorplay.bfloat16
        )
        self._alloc("attn_res.output_query", (hidden,), tensorplay.float32)
        for index in range(1, config.num_hidden_layers):
            layer = config.layers[index]
            count = 1 if layer.is_local else 3
            self._alloc(
                f"attn_res.attn_queries.{index - 1}",
                (count, hidden),
                tensorplay.float32,
            )
        for index in range(config.num_hidden_layers):
            self._alloc(
                f"attn_res.mlp_queries.{index}", (hidden,), tensorplay.float32
            )
        if config.loop_num_passes > 1:
            rows = 2 * (config.loop_core_end - config.loop_core_start)
            self._alloc(
                "middle_cycle_cross_pass_queries",
                (rows, hidden),
                tensorplay.float32,
            )
            self._alloc(
                "middle_cycle_cross_pass_kv_queries",
                (rows, hidden),
                tensorplay.float32,
            )
        for layer in config.layers:
            self._allocate_layer(layer)

    def _allocate_layer(self, layer: Inkling2Layer) -> None:
        prefix = f"layers.{layer.index}"
        hidden = self.hidden
        kv_width = layer.num_kv_heads * layer.head_dim
        q_width = layer.num_heads * layer.head_dim
        rel_width = layer.num_heads * layer.d_rel
        kernel = layer.sconv_kernel_size
        if layer.is_local:
            self._alloc(f"{prefix}.attn_norm.weight", (hidden,), tensorplay.bfloat16)
        else:
            for stream in ("q", "k", "v"):
                self._alloc(
                    f"{prefix}.{stream}_input_norm.weight",
                    (hidden,),
                    tensorplay.bfloat16,
                )
        self._alloc(f"{prefix}.mlp_norm.weight", (hidden,), tensorplay.bfloat16)
        self._alloc(
            f"{prefix}.attn.wq_du.weight", (q_width, hidden), tensorplay.bfloat16
        )
        self._alloc(
            f"{prefix}.attn.wk_dv.weight", (kv_width, hidden), tensorplay.bfloat16
        )
        self._alloc(
            f"{prefix}.attn.wv_dv.weight", (kv_width, hidden), tensorplay.bfloat16
        )
        self._alloc(
            f"{prefix}.attn.wr_du.weight", (rel_width, hidden), tensorplay.bfloat16
        )
        self._alloc(
            f"{prefix}.attn.gate_attn.weight", (q_width, hidden), tensorplay.bfloat16
        )
        self._alloc(
            f"{prefix}.attn.attn_sink_proj.weight",
            (layer.num_heads, hidden),
            tensorplay.bfloat16,
        )
        self._alloc(
            f"{prefix}.attn.wo_ud.weight",
            (hidden, q_width),
            tensorplay.bfloat16,
        )
        self._alloc(
            f"{prefix}.attn.q_l2_clip.scale",
            (layer.num_heads, layer.head_dim),
            tensorplay.bfloat16,
        )
        self._alloc(
            f"{prefix}.attn.rel_l2_clip.scale",
            (layer.num_heads, layer.d_rel),
            tensorplay.bfloat16,
        )
        self._alloc(
            f"{prefix}.attn.rel_logits_proj.proj",
            (layer.d_rel, layer.rel_extent),
            tensorplay.bfloat16,
        )
        self._alloc(
            f"{prefix}.attn_sconv.weight",
            (hidden, 1, kernel),
            tensorplay.bfloat16,
        )
        self._alloc(
            f"{prefix}.mlp_sconv.weight",
            (hidden, 1, kernel),
            tensorplay.bfloat16,
        )
        if layer.is_local:
            self._alloc(
                f"{prefix}.attn.k_sconv.weight",
                (kv_width, 1, kernel),
                tensorplay.bfloat16,
            )
            self._alloc(
                f"{prefix}.attn.v_sconv.weight",
                (kv_width, 1, kernel),
                tensorplay.bfloat16,
            )
        else:
            for stream in ("k", "v"):
                self._alloc(
                    f"{prefix}.attn.{stream}_sconv.static_w",
                    (kernel, kv_width),
                    tensorplay.bfloat16,
                )
                self._alloc(
                    f"{prefix}.attn.{stream}_sconv.weight_proj",
                    (layer.num_kv_heads * kernel, hidden),
                    tensorplay.bfloat16,
                )
        self._alloc(
            f"{prefix}.mlp.gate.weight",
            (layer.num_experts, hidden),
            tensorplay.bfloat16,
        )
        self._alloc(
            f"{prefix}.mlp.gate.bias", (layer.num_experts,), tensorplay.float32
        )
        intermediate = layer.moe_intermediate_size
        experts = layer.num_experts
        if layer.expert_storage == "nvfp4":
            self._alloc(
                f"{prefix}.mlp.experts.w13_weight",
                (experts, hidden, intermediate),
                tensorplay.uint8,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w13_weight_scale",
                (experts, hidden, intermediate * 2 // 16),
                tensorplay.float8_e4m3fn,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w13_weight_amax",
                (experts,),
                tensorplay.float32,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w13_weight_outer_scale_denominator",
                (1,),
                tensorplay.float32,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w2_weight",
                (experts, hidden, intermediate // 2),
                tensorplay.uint8,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w2_weight_scale",
                (experts, hidden, intermediate // 16),
                tensorplay.float8_e4m3fn,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w2_weight_amax",
                (experts,),
                tensorplay.float32,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w2_weight_outer_scale_denominator",
                (1,),
                tensorplay.float32,
            )
        else:
            self._alloc(
                f"{prefix}.mlp.experts.w13_weight",
                (experts, hidden, intermediate * 2),
                tensorplay.float8_e4m3fn,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w13_weight_scale",
                (experts, hidden, intermediate * 2 // 32),
                tensorplay.uint8,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w2_weight",
                (experts, hidden, intermediate),
                tensorplay.float8_e4m3fn,
            )
            self._alloc(
                f"{prefix}.mlp.experts.w2_weight_scale",
                (experts, hidden, intermediate // 32),
                tensorplay.uint8,
            )
        if self.inkling.is_looped(layer.index):
            for name in LOOP_INPUTS:
                self._alloc(
                    f"{prefix}.controllers.{name}.down_weight",
                    (self.inkling.loop_gate_rank, 2 * hidden),
                    tensorplay.bfloat16,
                )
                self._alloc(
                    f"{prefix}.controllers.{name}.forget_weight",
                    (hidden, self.inkling.loop_gate_rank),
                    tensorplay.float32,
                )
                self._alloc(
                    f"{prefix}.controllers.{name}.gain_weight",
                    (hidden, self.inkling.loop_gate_rank),
                    tensorplay.float32,
                )

    # -- loading -----------------------------------------------------------

    def load_weights(self, model_path: str | Path) -> WeightLoadReport:
        from megalm.loading import SafetensorsSource

        source = SafetensorsSource(model_path)
        expected = set(self.weights)
        seen: set[str] = set()
        total = 0
        for info, tensor in source.iter_tensors():
            name = info.name
            if name.startswith("model.visual."):
                continue
            key = name.replace("model.llm.", "", 1)
            if key not in self.weights:
                raise Inkling2Error(f"unexpected Inkling-2 checkpoint tensor {name}")
            self.weights[key].copy_(tensor)
            seen.add(key)
            total += info.nbytes
        missing = expected - seen
        if missing:
            raise Inkling2Error(
                f"Inkling-2 checkpoint is missing {len(missing)} tensors, "
                f"first: {sorted(missing)[:3]}"
            )
        self._derive_queries()
        self._bind_layers()
        if self.int4_experts:
            self._convert_experts_to_int4()
        return WeightLoadReport(
            tensor_count=len(seen),
            total_bytes=total,
            source_names=tuple(sorted(seen)),
            target_names=tuple(sorted(seen)),
        )

    def _derive_queries(self) -> None:
        """Scale the depth pseudo-queries and build the loop message queries."""

        hidden = self.hidden
        scale = float(hidden) ** -0.5
        config = self.inkling
        self.attn_queries = {}
        for layer in range(1, config.num_hidden_layers):
            count = 1 if config.layers[layer].is_local else 3
            tensor = self.weights[f"attn_res.attn_queries.{layer - 1}"]
            self.attn_queries[layer] = tensor.reshape(count, hidden) * scale
        self.mlp_queries = [
            self.weights[f"attn_res.mlp_queries.{layer}"] * scale
            for layer in range(config.num_hidden_layers)
        ]
        self.output_query = self.weights["attn_res.output_query"] * scale
        if config.loop_num_passes > 1:
            queries = self.weights["middle_cycle_cross_pass_queries"].reshape(
                -1, 2, hidden
            )
            kv_queries = self.weights[
                "middle_cycle_cross_pass_kv_queries"
            ].reshape(-1, 2, hidden)
            rows = {
                "attn": queries[:, 0],
                "k": kv_queries[:, 0],
                "v": kv_queries[:, 1],
                "mlp": queries[:, 1],
            }
            self.loop_score_weights = tensorplay.stack(
                [rows[name] * scale for name in LOOP_INPUTS], dim=1
            )

    def _bind_layers(self) -> None:
        self.layers = [
            self._bind_layer(self.inkling.layers[index])
            for index in range(self.inkling.num_hidden_layers)
        ]

    def _bind_layer(self, layer: Inkling2Layer) -> LayerWeights:
        prefix = f"layers.{layer.index}"
        get = self.weights.__getitem__
        kv_width = layer.num_kv_heads * layer.head_dim
        kernel = layer.sconv_kernel_size
        local = layer.is_local
        if local:
            k_sconv = get(f"{prefix}.attn.k_sconv.weight").reshape(kv_width, kernel)
            v_sconv = get(f"{prefix}.attn.v_sconv.weight").reshape(kv_width, kernel)
            k_proj = None
            v_proj = None
        else:
            k_sconv = get(f"{prefix}.attn.k_sconv.static_w").transpose(0, 1)
            v_sconv = get(f"{prefix}.attn.v_sconv.static_w").transpose(0, 1)
            k_proj = get(f"{prefix}.attn.k_sconv.weight_proj")
            v_proj = get(f"{prefix}.attn.v_sconv.weight_proj")
        controllers = None
        if self.inkling.is_looped(layer.index):
            controllers = {
                name: (
                    get(f"{prefix}.controllers.{name}.down_weight"),
                    get(f"{prefix}.controllers.{name}.forget_weight"),
                    get(f"{prefix}.controllers.{name}.gain_weight"),
                )
                for name in LOOP_INPUTS
            }
        stacked_controllers = None
        if controllers is not None:
            stacked_controllers = tuple(
                tensorplay.stack(
                    [controllers[name][slot] for name in LOOP_INPUTS[:3]],
                    dim=0,
                )
                for slot in range(3)
            )
        nvfp4 = layer.expert_storage == "nvfp4"
        return LayerWeights(
            index=layer.index,
            is_local=local,
            attn_norm=get(f"{prefix}.attn_norm.weight")
            if local
            else get(f"{prefix}.q_input_norm.weight"),
            q_norm=None if local else get(f"{prefix}.q_input_norm.weight"),
            k_norm=None if local else get(f"{prefix}.k_input_norm.weight"),
            v_norm=None if local else get(f"{prefix}.v_input_norm.weight"),
            mlp_norm=get(f"{prefix}.mlp_norm.weight"),
            wq=get(f"{prefix}.attn.wq_du.weight"),
            wk=get(f"{prefix}.attn.wk_dv.weight"),
            wv=get(f"{prefix}.attn.wv_dv.weight"),
            wr=get(f"{prefix}.attn.wr_du.weight"),
            wo=get(f"{prefix}.attn.wo_ud.weight"),
            gate=get(f"{prefix}.attn.gate_attn.weight"),
            sink=get(f"{prefix}.attn.attn_sink_proj.weight"),
            q_clip=get(f"{prefix}.attn.q_l2_clip.scale"),
            rel_clip=get(f"{prefix}.attn.rel_l2_clip.scale"),
            rel_proj=get(f"{prefix}.attn.rel_logits_proj.proj"),
            k_sconv=k_sconv,
            v_sconv=v_sconv,
            k_sconv_proj=k_proj,
            v_sconv_proj=v_proj,
            attn_sconv=get(f"{prefix}.attn_sconv.weight").reshape(
                self.hidden, kernel
            ),
            mlp_sconv=get(f"{prefix}.mlp_sconv.weight").reshape(
                self.hidden, kernel
            ),
            gate_weight=get(f"{prefix}.mlp.gate.weight"),
            gate_bias=get(f"{prefix}.mlp.gate.bias"),
            w13=get(f"{prefix}.mlp.experts.w13_weight"),
            w13_scale=get(f"{prefix}.mlp.experts.w13_weight_scale"),
            w13_amax=get(f"{prefix}.mlp.experts.w13_weight_amax") if nvfp4 else None,
            w13_denominator=get(
                f"{prefix}.mlp.experts.w13_weight_outer_scale_denominator"
            )
            if nvfp4
            else None,
            w2=get(f"{prefix}.mlp.experts.w2_weight"),
            w2_scale=get(f"{prefix}.mlp.experts.w2_weight_scale"),
            w2_amax=get(f"{prefix}.mlp.experts.w2_weight_amax") if nvfp4 else None,
            w2_denominator=get(
                f"{prefix}.mlp.experts.w2_weight_outer_scale_denominator"
            )
            if nvfp4
            else None,
            controllers=controllers,
            stacked_loop_controllers=stacked_controllers,
        )

    # -- engine hooks ------------------------------------------------------

    def configure_moe_workspace(self, max_batched_tokens: int) -> int:
        """Allocate the reusable sorted-route/GEMM arena for int4 experts."""

        if max_batched_tokens <= 0:
            raise ValueError("MoE workspace token budget must be positive")
        if not self.int4_experts or self.device.type != "cuda" or not self.layers:
            self._int4_workspace = None
            return 0
        first = self.layers[0]
        packed13, _, _, _ = self._stacked_int4_experts(first)
        self._int4_workspace = Int4MoEWorkspace.allocate(
            max_tokens=max_batched_tokens,
            num_experts=first.gate_weight.shape[0],
            hidden_size=self.hidden,
            out_width13=packed13.shape[1],
            top_k=self.inkling.layers[0].experts_per_token,
            dtype=self.dtype,
            device=self.device,
        )
        return self._int4_workspace.nbytes

    def warmup_prefill_kernels(self, *, max_batched_tokens: int) -> tuple[int, ...]:
        del max_batched_tokens
        return ()

    def configure_mtp_hidden_buffer(self, max_tokens: int) -> int:
        if self.mtp is None:
            self._mtp_features = None
            return 0
        if max_tokens <= 0:
            raise Inkling2Error("MTP feature buffer capacity must be positive")
        self._mtp_features = tensorplay.zeros(
            (max_tokens, 4, self.hidden),
            dtype=self.dtype,
            device=self.device,
        )
        return self._mtp_features.numel() * self._mtp_features.element_size()

    @property
    def last_hidden_states(self) -> Tensor | None:
        return self._mtp_features

    def lm_head(self) -> Tensor:
        return self.weights["unembed.weight"]

    # -- convolution -------------------------------------------------------

    def _conv_state(
        self,
        visit: int,
        kind: str,
        channels: int,
    ) -> Tensor:
        """The ``[slots, channels, k-1]`` arena backing one short conv."""

        key = (visit, kind)
        arena = self._conv.get(key)
        if arena is None:
            arena = tensorplay.zeros(
                (
                    self.num_state_slots,
                    channels,
                    self.inkling.layers[0].sconv_kernel_size - 1,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            self._conv[key] = arena
        return arena

    def _sconv(
        self,
        x: Tensor,
        taps: Tensor,
        *,
        batch: ExecutionBatch,
        visit: int,
        kind: str,
        channels: int,
        dynamic: Tensor | None = None,
    ) -> Tensor:
        """Causal depthwise short convolution with a residual connection.

        Each sequence owns its own ``kernel - 1`` token history in a stable
        per-request slot.  Decode uses gather/scatter on the GPU so the whole
        step is CUDA-graph capturable; prefill uses direct arena views.
        """

        starts = batch.query_start_loc
        arena = self._conv_state(visit, kind, channels)
        if (
            batch.all_decode
            and batch.state_slots is not None
            and batch.query_lengths == (1,) * batch.num_sequences
            and x.device.type == "cuda"
            and x.is_contiguous()
            and taps.is_contiguous()
            and (
                dynamic is None
                or (dynamic.is_contiguous() and dynamic.dtype == taps.dtype)
            )
        ):
            return short_convolution_decode(
                x,
                taps,
                arena,
                batch.state_slots,
                batch.positions,
                batch.token_active,
                dynamic,
            )
        outputs: list[Tensor] = []
        for index in range(batch.num_sequences):
            start = starts[index]
            end = starts[index + 1]
            if start == end:
                continue
            if batch.all_decode:
                if batch.state_slots is None:
                    raise Inkling2Error("graph decode batch has no state slots")
                slot_index = batch.state_slots[index : index + 1]
                state = arena.index_select(0, slot_index)
            else:
                slot = (
                    0
                    if batch.state_slots is None
                    else int(batch.state_slots[index].item())
                )
                if slot >= self.num_state_slots:
                    raise Inkling2Error(
                        f"state slot {slot} exceeds the {self.num_state_slots} "
                        "allocated convolution state slots"
                    )
                state = arena[slot : slot + 1]
            if batch.context_lengths[index] == 0:
                state.zero_()
            outputs.append(
                self._sconv_segment(
                    x[start:end],
                    taps,
                    state=state,
                    dynamic=None if dynamic is None else dynamic[start:end],
                )
            )
            if batch.all_decode:
                arena.index_copy_(0, slot_index, state)
        if not outputs:
            raise Inkling2Error("short convolution received an empty batch")
        return (
            outputs[0]
            if len(outputs) == 1
            else tensorplay.cat(outputs, dim=0)
        )

    def _sconv_segment(
        self,
        x: Tensor,
        taps: Tensor,
        *,
        state: Tensor,
        dynamic: Tensor | None = None,
    ) -> Tensor:
        tokens = x.shape[0]
        kernel = taps.shape[-1]
        values = x.float()
        history = state[0].float().transpose(0, 1)
        padded = tensorplay.cat((history, values), dim=0)
        output = tensorplay.zeros_like(values)
        for tap in range(kernel):
            # ``padded[tap]`` is the sample ``kernel - 1 - tap`` steps back.
            # Static taps are stored in the checkpoint's ``conv1d`` order, so
            # that sample pairs with tap ``tap``; the dynamic taps are stored
            # lag-major (tap 0 is the current token), so they pair with the
            # backward distance directly.
            sample = padded[tap : tap + tokens]
            if dynamic is None:
                coefficient = taps[:, tap].float()
            else:
                lag = kernel - 1 - tap
                coefficient = (taps[:, lag] + dynamic[:, :, lag]).float()
            output = output + sample * coefficient
        scale = (kernel * (2 if dynamic is not None else 1) + 1) ** -0.5
        output = (output + values) * scale
        if kernel > 1:
            state[0].copy_(padded[tokens:].transpose(0, 1))
        return output.to(x.dtype)

    # -- attention ---------------------------------------------------------

    def _attention(
        self,
        layer: LayerWeights,
        streams: tuple[Tensor, ...],
        *,
        visit: int,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        tau: Tensor | None,
    ) -> Tensor:
        spec = self.inkling.layers[layer.index]
        tokens = streams[0].shape[0]
        q_source = streams[0]
        if len(streams) == 1:
            k_source = q_source
            v_source = q_source
        else:
            _, k_source, v_source = streams
        q = tensorplay.matmul(q_source, layer.wq.transpose(0, 1))
        k = tensorplay.matmul(k_source, layer.wk.transpose(0, 1))
        v = tensorplay.matmul(v_source, layer.wv.transpose(0, 1))
        r = tensorplay.matmul(q_source, layer.wr.transpose(0, 1))
        gate = tensorplay.matmul(q_source, layer.gate.transpose(0, 1))
        sink = tensorplay.matmul(q_source, layer.sink.transpose(0, 1))

        kernel = spec.sconv_kernel_size
        kv_width = spec.num_kv_heads * spec.head_dim
        dynamic_k = None
        dynamic_v = None
        if layer.k_sconv_proj is not None:
            dynamic_k = tensorplay.matmul(
                k_source, layer.k_sconv_proj.transpose(0, 1)
            ).reshape(tokens, spec.num_kv_heads, kernel)
            dynamic_v = tensorplay.matmul(
                v_source, layer.v_sconv_proj.transpose(0, 1)
            ).reshape(tokens, spec.num_kv_heads, kernel)
            dynamic_k = dynamic_k.repeat_interleave(spec.head_dim, dim=1)
            dynamic_v = dynamic_v.repeat_interleave(spec.head_dim, dim=1)
        k = self._sconv(
            k,
            layer.k_sconv,
            batch=batch,
            visit=visit,
            kind="k",
            channels=kv_width,
            dynamic=dynamic_k,
        )
        v = self._sconv(
            v,
            layer.v_sconv,
            batch=batch,
            visit=visit,
            kind="v",
            channels=kv_width,
            dynamic=dynamic_v,
        )

        q = q.reshape(tokens, spec.num_heads, spec.head_dim)
        k = k.reshape(tokens, spec.num_kv_heads, spec.head_dim)
        v = v.reshape(tokens, spec.num_kv_heads, spec.head_dim)
        r = r.reshape(tokens, spec.num_heads, spec.d_rel)
        gate = gate.reshape(tokens, spec.num_heads, spec.head_dim)
        sink = sink.reshape(tokens, spec.num_heads)

        attention_tau = None if spec.is_local else tau
        if (
            batch.all_decode
            and q.device.type == "cuda"
            and max(spec.head_dim, spec.d_rel) <= 256
            and q.dtype == layer.q_clip.dtype
            and r.dtype == layer.rel_clip.dtype
            and (attention_tau is None or attention_tau.is_contiguous())
            and all(
                tensor.is_contiguous()
                for tensor in (q, k, v, r, layer.q_clip, layer.rel_clip)
            )
        ):
            q, k, v, r = prepare_inkling2_heads(
                q,
                k,
                v,
                r,
                layer.q_clip,
                layer.rel_clip,
                attention_tau,
                qk_threshold=spec.qk_l2_clip_threshold,
                rel_threshold=spec.rel_l2_clip_threshold,
                eps=self.eps,
            )
        else:
            q = q * layer.q_clip.reshape(1, spec.num_heads, spec.head_dim)
            q = l2_clip(
                q.reshape(tokens * spec.num_heads, spec.head_dim),
                spec.qk_l2_clip_threshold,
            ).reshape(tokens, spec.num_heads, spec.head_dim)
            k = l2_clip(
                k.reshape(tokens * spec.num_kv_heads, spec.head_dim),
                spec.qk_l2_clip_threshold,
            ).reshape(tokens, spec.num_kv_heads, spec.head_dim)
            v = rms_norm(v, None, self.eps, output_dtype=v.dtype)
            r = r * layer.rel_clip.reshape(1, spec.num_heads, spec.d_rel)
            r = l2_clip(
                r.reshape(tokens * spec.num_heads, spec.d_rel),
                spec.rel_l2_clip_threshold,
            ).reshape(tokens, spec.num_heads, spec.d_rel)
            if attention_tau is not None:
                q = (q.float() * attention_tau.reshape(tokens, 1, 1)).to(q.dtype)
                r = (r.float() * attention_tau.reshape(tokens, 1, 1)).to(r.dtype)

        key_cache, value_cache = cache.layer(visit)
        page_size = key_cache.shape[1]
        flat_keys = key_cache.reshape(-1, spec.num_kv_heads, spec.head_dim)
        flat_values = value_cache.reshape(-1, spec.num_kv_heads, spec.head_dim)
        # One fused, device-only scatter replaces the where/zeros/index_copy_
        # chain. Inactive graph rows are skipped; their page tables are clamped
        # to pages whose length mask is empty, so no stale slot is read.
        active_kv_write(
            k,
            v,
            flat_keys,
            flat_values,
            batch.slot_mapping.long(),
            batch.token_active,
        )

        # ``rel_logits_proj.proj`` is [d_rel, rel_extent]; the projection is
        # ``states @ proj`` with no transpose.
        rel_logits = tensorplay.matmul(
            r.reshape(tokens * spec.num_heads, spec.d_rel),
            layer.rel_proj,
        ).reshape(tokens, spec.num_heads, spec.rel_extent)

        scale = (
            spec.head_dim**-0.5 if self.attn_scale == "sqrt" else 1.0 / spec.head_dim
        )
        window = int(self.inkling.sliding_window_size) if spec.is_local else 0
        cache_span = batch.block_tables.shape[1] * page_size
        if window:
            cache_span = min(cache_span, window)
        if (
            batch.all_decode
            and batch.query_lengths == (1,) * batch.num_sequences
            and not batch.is_speculative
            and 0 < cache_span <= MAX_PAGED_DECODE_TOKENS
            and q.device.type == "cuda"
            and all(
                tensor.is_contiguous()
                for tensor in (
                    q,
                    key_cache,
                    value_cache,
                    rel_logits,
                    sink,
                    batch.block_tables,
                    batch.sequence_lengths,
                    batch.positions,
                    batch.token_active,
                )
            )
        ):
            output = inkling2_paged_decode(
                q,
                key_cache,
                value_cache,
                rel_logits,
                sink,
                batch.block_tables,
                batch.sequence_lengths,
                batch.positions,
                batch.token_active,
                window=window,
                scale=scale,
            )
        else:
            output = tensorplay.empty_like(q)
            page_offsets = tensorplay.arange(page_size, device=self.device)
            for sequence_index in range(batch.num_sequences):
                start = batch.query_start_loc[sequence_index]
                end = batch.query_start_loc[sequence_index + 1]
                query_length = end - start
                # Sequence geometry comes from device tensors: decode graphs
                # replay with a static Python tuple but fresh device metadata.
                sequence_length = batch.sequence_lengths[sequence_index]
                pages = batch.block_tables[sequence_index]
                first_key = 0
                if not batch.all_decode:
                    # Eager prefill knows the positions covered by the query.
                    # Gather only the union of pages visible to those tokens.
                    context = batch.context_lengths[sequence_index]
                    last_key = context + query_length
                    first_visible = max(0, context - window + 1) if window else 0
                    first_page = first_visible // page_size
                    last_page = (last_key + page_size - 1) // page_size
                    pages = pages[first_page:last_page]
                    first_key = first_page * page_size
                pages = pages.long().clamp_min(0)
                slots = (
                    pages.reshape(-1, 1) * page_size + page_offsets.reshape(1, -1)
                ).reshape(-1)
                keys = flat_keys.index_select(0, slots)
                values = flat_values.index_select(0, slots)
                key_positions = (
                    tensorplay.arange(slots.numel(), device=self.device) + first_key
                )
                query_positions = batch.positions[start:end]
                distance = query_positions.reshape(-1, 1) - key_positions.reshape(1, -1)
                key_mask = (distance >= 0) & (key_positions < sequence_length)
                if window:
                    key_mask = key_mask & (distance < window)
                valid = (distance >= 0) & (distance < spec.rel_extent)
                zero = tensorplay.zeros_like(distance)
                gather_index = tensorplay.minimum(
                    tensorplay.maximum(distance, zero),
                    tensorplay.full_like(distance, spec.rel_extent - 1),
                )
                # One batched gather replaces one launch per query head.
                bias = tensorplay.gather(
                    rel_logits[start:end],
                    2,
                    gather_index.unsqueeze(1).expand(-1, spec.num_heads, -1),
                )
                bias = tensorplay.where(
                    valid.reshape(query_length, 1, -1),
                    bias,
                    tensorplay.zeros_like(bias),
                )
                repeat = spec.num_heads // spec.num_kv_heads
                key_states = keys.reshape(
                    keys.shape[0], spec.num_kv_heads, spec.head_dim
                ).repeat_interleave(repeat, dim=1)
                value_states = values.reshape(
                    values.shape[0], spec.num_kv_heads, spec.head_dim
                ).repeat_interleave(repeat, dim=1)
                query_states = q[start:end].transpose(0, 1).unsqueeze(0)
                logits = tensorplay.matmul(
                    query_states,
                    key_states.transpose(0, 1).unsqueeze(0).transpose(2, 3),
                ) * scale
                logits = logits + bias.transpose(0, 1).unsqueeze(0)
                logits = tensorplay.where(
                    key_mask.reshape(1, 1, query_length, -1),
                    logits,
                    tensorplay.full_like(logits, float("-inf")),
                )
                sink_column = (
                    sink[start:end]
                    .reshape(1, query_length, spec.num_heads, 1)
                    .transpose(1, 2)
                )
                # HF computes the sink-extended softmax in fp32 and casts the
                # probabilities back to the query dtype before the value matmul.
                combined = tensorplay.cat(
                    (logits.float(), sink_column.float()), dim=-1
                )
                probabilities = tensorplay.softmax(combined, dim=-1)[..., :-1].to(
                    logits.dtype
                )
                attended = tensorplay.matmul(
                    probabilities,
                    value_states.transpose(0, 1).unsqueeze(0),
                )
                output[start:end] = attended.squeeze(0).transpose(0, 1)
        gated = output * tensorplay.sigmoid(gate)
        return tensorplay.matmul(
            gated.reshape(tokens, spec.num_heads * spec.head_dim),
            layer.wo.transpose(0, 1),
        )

    # -- profiling ---------------------------------------------------------

    def _prof_start(self) -> float:
        if not self.profile:
            return 0.0
        self._prof_sync()
        return time.perf_counter()

    def _tick(self, name: str, start: float) -> float:
        if not self.profile:
            return 0.0
        self._prof_sync()
        now = time.perf_counter()
        self._prof[name] = self._prof.get(name, 0.0) + (now - start)
        self._prof_calls[name] = self._prof_calls.get(name, 0) + 1
        return now

    def _prof_sync(self) -> None:
        try:
            tensorplay.cuda.synchronize(self.device)
        except Exception:  # pragma: no cover - profiling is best effort
            pass

    def _prof_report(self, visits: int) -> None:
        if not self.profile or not self._prof:
            return
        total = sum(self._prof.values())
        parts = " ".join(
            f"{name}={self._prof[name] * 1e3:.1f}ms/{self._prof_calls[name]}"
            for name in sorted(self._prof, key=self._prof.__getitem__, reverse=True)
        )
        print(f"[ink-prof] visits={visits} total={total * 1e3:.1f}ms {parts}")
        self._prof.clear()
        self._prof_calls.clear()

    # -- MoE ---------------------------------------------------------------

    def _dequant_slab(
        self,
        packed: Tensor,
        scale: Tensor,
        amax: Tensor | None,
        denominator: Tensor | None,
        quant: str,
    ) -> Tensor:
        """Decode an ``[E, ...]`` quantised slab into the model dtype."""

        if quant == "nvfp4":
            block = unpack_nvfp4(packed, scale.float())
            # ModelOpt NVFP4 stores one E4M3 block scale per 16 values plus a
            # per-expert outer scale ``amax / denominator`` (the checkpoint's
            # denominator is ``448 * 6 = 2688``; the block scales reach 448).
            # Reconstructing with ``amax / 1536`` instead leaves every expert
            # weight 1.75x too large.
            assert amax is not None and denominator is not None
            factor = (amax.float() / (denominator.float() * self.w_denominator)).reshape(
                -1, 1, 1
            )
            block = block * factor
        else:
            block = unpack_mxfp8(packed, scale)
        return block.to(self.dtype)

    def _decode_experts(
        self,
        layer: LayerWeights,
        experts: tuple[int, ...],
        quant: str,
    ) -> dict[int, tuple[Tensor, Tensor]]:
        """Decode every routed expert of one layer in a single batched pass.

        The elementwise decode is launch-bound, so decoding the eight routed
        experts together costs about as much as decoding one, while the LRU
        cache still serves experts that repeat across the loop passes.
        """

        resolved: dict[int, tuple[Tensor, Tensor]] = {}
        missing: list[int] = []
        for expert in experts:
            cached = self._expert_cache.get((layer.index, expert))
            if cached is None:
                missing.append(expert)
            else:
                resolved[expert] = cached
        if missing:
            assert layer.w13 is not None and layer.w13_scale is not None
            assert layer.w2 is not None and layer.w2_scale is not None
            # Chunk the decode: an eight-expert batch needs ~0.7 GiB of
            # transient FP32, which does not fit beside the resident weights
            # on a 24 GiB card.
            for start in range(0, len(missing), 2):
                chunk = missing[start : start + 2]
                index = tensorplay.tensor(
                    chunk, dtype=tensorplay.int64, device=self.device
                )
                w13 = self._dequant_slab(
                    layer.w13.index_select(0, index),
                    layer.w13_scale.index_select(0, index),
                    None if layer.w13_amax is None else layer.w13_amax.index_select(0, index),
                    layer.w13_denominator,
                    quant,
                )
                w2 = self._dequant_slab(
                    layer.w2.index_select(0, index),
                    layer.w2_scale.index_select(0, index),
                    None if layer.w2_amax is None else layer.w2_amax.index_select(0, index),
                    layer.w2_denominator,
                    quant,
                )
                for offset, expert in enumerate(chunk):
                    pair = (w13[offset], w2[offset])
                    resolved[expert] = pair
                    if self.expert_cache_size > 0:
                        key = (layer.index, expert)
                        self._expert_cache[key] = pair
                        self._expert_cache_order.append(key)
                        while len(self._expert_cache_order) > self.expert_cache_size:
                            oldest = self._expert_cache_order.pop(0)
                            self._expert_cache.pop(oldest, None)
        return resolved

    def _convert_experts_to_int4(self) -> None:
        """Re-quantise every routed expert to int4 weight-only, in place.

        tensorplay's ``_weight_int4pack_mm`` dequantises the packed B-fragment
        inside the GEMM kernel, so the model keeps ~13 GiB of experts resident
        instead of decoding a 19 GiB checkpoint slab on every use.  Each
        layer's checkpoint-precision slab is released as soon as its int4
        replacement exists, which keeps the conversion inside the 24 GiB
        budget.
        """

        group = self.int4_group
        total = 0
        for layer in self.layers:
            spec = self.inkling.layers[layer.index]
            quant = spec.expert_storage
            if layer.w13 is None or layer.w2 is None:
                continue
            count = spec.num_experts
            packed13: list[Tensor] = []
            scale13: list[Tensor] = []
            packed2: list[Tensor] = []
            scale2: list[Tensor] = []
            # CUDA deployments keep each expert GEMM in one fused grouped
            # kernel, so the layer owns pre-stacked ``[E, N, K/2]`` codes.
            # Filling the final stacked tensor while the slab chunks stream
            # through avoids a second full pass (and its transient copy) at
            # first use, which fragmented the allocator badly enough to OOM.
            stacked13: Tensor | None = None
            stacked13_scale: Tensor | None = None
            stacked2: Tensor | None = None
            stacked2_scale: Tensor | None = None
            for start in range(0, count, 4):
                stop = min(start + 4, count)
                index = tensorplay.tensor(
                    list(range(start, stop)),
                    dtype=tensorplay.int64,
                    device=self.device,
                )
                block13 = self._dequant_slab(
                    layer.w13.index_select(0, index),
                    layer.w13_scale.index_select(0, index),
                    None
                    if layer.w13_amax is None
                    else layer.w13_amax.index_select(0, index),
                    layer.w13_denominator,
                    quant,
                )
                block2 = self._dequant_slab(
                    layer.w2.index_select(0, index),
                    layer.w2_scale.index_select(0, index),
                    None
                    if layer.w2_amax is None
                    else layer.w2_amax.index_select(0, index),
                    layer.w2_denominator,
                    quant,
                )
                if self.device.type == "cuda" and stacked13 is None:
                    out13, k13 = block13.shape[1], block13.shape[2]
                    out2, k2 = block2.shape[1], block2.shape[2]
                    stacked13 = tensorplay.empty(
                        (count, out13, k13 // 2),
                        dtype=tensorplay.uint8,
                        device=self.device,
                    )
                    stacked13_scale = tensorplay.empty(
                        (count, k13 // group, out13, 2),
                        dtype=tensorplay.bfloat16,
                        device=self.device,
                    )
                    stacked2 = tensorplay.empty(
                        (count, out2, k2 // 2),
                        dtype=tensorplay.uint8,
                        device=self.device,
                    )
                    stacked2_scale = tensorplay.empty(
                        (count, k2 // group, out2, 2),
                        dtype=tensorplay.bfloat16,
                        device=self.device,
                    )
                for offset in range(stop - start):
                    # Checkpoint slabs are ``[out, in]`` (``F.linear`` order),
                    # which is exactly the ``[N, K]`` layout
                    # ``_pack_int4_weight`` expects for ``x @ weight.T``.
                    if self.device.type == "cuda":
                        # Row-major custom layout consumed by the grouped
                        # int4 MoE kernel (capture-safe); the native
                        # tinygemm packing is only needed for the CPU op.
                        packed, scales = pack_int4_expert_rowmajor(
                            block13[offset], group
                        )
                        stacked13[start + offset].copy_(packed)
                        stacked13_scale[start + offset].copy_(
                            scales.to(tensorplay.bfloat16)
                        )
                        packed, scales = pack_int4_expert_rowmajor(
                            block2[offset], group
                        )
                        stacked2[start + offset].copy_(packed)
                        stacked2_scale[start + offset].copy_(
                            scales.to(tensorplay.bfloat16)
                        )
                    else:
                        packed, scales = pack_int4_weight(
                            block13[offset], group
                        )
                        packed13.append(packed)
                        scale13.append(scales.to(tensorplay.bfloat16))
                        del packed, scales
                        packed, scales = pack_int4_weight(
                            block2[offset], group
                        )
                        packed2.append(packed)
                        scale2.append(scales.to(tensorplay.bfloat16))
                    del packed, scales
                del block13, block2
            if self.device.type == "cuda":
                layer.w13_packed = stacked13
                layer.w13_packed_scale = stacked13_scale
                layer.w2_packed = stacked2
                layer.w2_packed_scale = stacked2_scale
                total += stacked13.numel()
                total += stacked13_scale.numel() * 2
                total += stacked2.numel()
                total += stacked2_scale.numel() * 2
            else:
                layer.w13_packed = packed13
                layer.w13_packed_scale = scale13
                layer.w2_packed = packed2
                layer.w2_packed_scale = scale2
                total += sum(t.numel() * 4 for t in packed13)
                total += sum(t.numel() * 2 for t in scale13)
                total += sum(t.numel() * 4 for t in packed2)
                total += sum(t.numel() * 2 for t in scale2)
            # Drop the checkpoint slabs: the int4 copy fully replaces them.
            prefix = f"layers.{layer.index}"
            layer.w13 = None
            layer.w13_scale = None
            layer.w13_amax = None
            layer.w13_denominator = None
            layer.w2 = None
            layer.w2_scale = None
            layer.w2_amax = None
            layer.w2_denominator = None
            for suffix in (
                "mlp.experts.w13_weight",
                "mlp.experts.w13_weight_scale",
                "mlp.experts.w13_weight_amax",
                "mlp.experts.w13_weight_outer_scale_denominator",
                "mlp.experts.w2_weight",
                "mlp.experts.w2_weight_scale",
                "mlp.experts.w2_weight_amax",
                "mlp.experts.w2_weight_outer_scale_denominator",
            ):
                self.weights.pop(f"{prefix}.{suffix}", None)
            if os.environ.get("MEGALM_DEBUG_MEM", "0") != "0":
                free, total = tensorplay.cuda.mem_get_info()
                print(
                    f"[ink-mem] int4 layer={layer.index} "
                    f"free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB",
                    flush=True,
                )
        self.int4_bytes = total
        self._drop_dead_weights()

    def _drop_dead_weights(self) -> None:
        """Release checkpoint buffers the int4 experts made unreachable."""

        live = {
            id(tensor) for tensor in self.weights.values()
        }
        for layer in self.layers:
            for tensor in (
                layer.w13,
                layer.w13_scale,
                layer.w2,
                layer.w2_scale,
            ):
                if tensor is not None:
                    live.add(id(tensor))
        flush = getattr(tensorplay.cuda, "empty_cache", None)
        if flush is not None:
            try:
                flush()
            except Exception:  # pragma: no cover - best effort
                pass
        if os.environ.get("MEGALM_DEBUG_MEM", "0") != "0":
            free, total = tensorplay.cuda.mem_get_info()
            print(
                f"[ink-mem] after drop free={free / 2**30:.2f}GiB "
                f"total={total / 2**30:.2f}GiB",
                flush=True,
            )

    def _moe(self, layer: LayerWeights, x: Tensor) -> Tensor:
        spec = self.inkling.layers[layer.index]
        tokens = x.shape[0]
        top = spec.experts_per_token
        tick = self._prof_start()
        logits = tensorplay.matmul(
            x.float(), layer.gate_weight.float().transpose(0, 1)
        )
        scores = tensorplay.sigmoid(logits)
        biased = scores + layer.gate_bias.reshape(1, -1)
        indices = tensorplay.topk(biased, top, dim=-1).indices.long()
        selected = tensorplay.gather(scores, 1, indices)
        # ``softmax(log sigmoid(logit)) == sigmoid(logit) / sum(sigmoid)``, the
        # normalisation both served references (vLLM, SGLang) apply.
        weights = (
            selected
            / (tensorplay.sum(selected, dim=-1, keepdim=True) + 1e-20)
            * spec.route_scale
        )
        # Group the (token, slot) routes by expert so each expert runs one
        # GEMM over all of its rows instead of one launch per row.  The sort
        # and the per-expert slices stay on the device: a host round trip per
        # expert costs more than the int4 GEMMs themselves on decode.
        flat_idx = indices.reshape(-1).long()
        flat_gate = weights.reshape(-1)
        if x.device.type == "cuda" and self.int4_experts:
            # Resident int4 experts use the grouped kernel: one launch per
            # GEMM stage with static shapes.  With MEGALM_INT4_EXPERTS=0 the
            # checkpoint-precision slabs stay resident and the segmented
            # decode path below runs on CUDA too.
            return self._moe_dense(
                layer,
                spec,
                x,
                indices.long(),
                weights,
            )
        order = tensorplay.argsort(flat_idx)
        sorted_experts = flat_idx.index_select(0, order).tolist()
        sorted_rows = (order // top).long()
        sorted_gate = flat_gate.index_select(0, order)
        tick = self._tick("moe.route", tick)
        starts: list[int] = [0]
        for position in range(1, len(sorted_experts)):
            if sorted_experts[position] != sorted_experts[position - 1]:
                starts.append(position)
        starts.append(len(sorted_experts))
        output = tensorplay.zeros_like(x.float())
        int4 = layer.w13_packed is not None
        decoded: dict[int, tuple[Tensor, Tensor]] | None = None
        if not int4:
            decoded = self._decode_experts(
                layer,
                tuple(int(sorted_experts[start]) for start in starts[:-1]),
                spec.expert_storage,
            )
        for segment in range(len(starts) - 1):
            begin, end = starts[segment], starts[segment + 1]
            expert = int(sorted_experts[begin])
            row_index = sorted_rows[begin:end]
            gamma = sorted_gate[begin:end].reshape(-1, 1)
            # Both GEMMs stay in the activation dtype (tensor cores accumulate
            # FP32); only the routing maths is upcast.
            rows_in = x.index_select(0, row_index)
            inner = self._prof_start()
            if int4:
                assert layer.w13_packed is not None
                assert layer.w13_packed_scale is not None
                gate_up = int4_weight_mm(
                    rows_in,
                    layer.w13_packed[expert],
                    layer.w13_packed_scale[expert],
                    self.int4_group,
                )
                w2_dtype = layer.w2_packed_scale[expert].dtype
            else:
                assert decoded is not None
                w13, w2 = decoded[expert]
                if self.w13_transposed:
                    gate_up = tensorplay.matmul(rows_in, w13.transpose(0, 1))
                else:
                    gate_up = tensorplay.matmul(rows_in, w13)
                w2_dtype = w2.dtype
            if self.w13_interleaved:
                gate = gate_up[:, 0::2]
                up = gate_up[:, 1::2]
            else:
                half = gate_up.shape[-1] // 2
                gate = gate_up[:, :half]
                up = gate_up[:, half:]
            hidden_act = capped_swiglu(
                gate,
                up,
                spec.mlp_gate_softcap,
                spec.mlp_up_softcap,
            ).to(w2_dtype)
            if int4:
                assert layer.w2_packed is not None
                assert layer.w2_packed_scale is not None
                contribution = (
                    int4_weight_mm(
                        hidden_act,
                        layer.w2_packed[expert],
                        layer.w2_packed_scale[expert],
                        self.int4_group,
                    ).float()
                    * gamma
                )
            else:
                assert decoded is not None
                contribution = (
                    tensorplay.matmul(
                        hidden_act, decoded[expert][1].transpose(0, 1)
                    ).float()
                    * gamma
                )
            self._tick("moe.exp", inner)
            output = tensorplay.index_add(output, 0, row_index, contribution)
        self._tick("moe.tail", tick)
        return output.to(x.dtype)

    def _moe_dense(
        self,
        layer: LayerWeights,
        spec,
        x: Tensor,
        flat_idx: Tensor,
        flat_gate: Tensor,
    ) -> Tensor:
        """Capture-safe MoE: one grouped int4 launch per GEMM stage.

        All routing stays on device (argsort/bincount are pure device ops),
        and the packed expert weights are dequantised inline inside the
        Triton kernel, so the captured op sequence has static shapes and a
        bounded node count.
        """

        token_count, top = flat_idx.shape
        packed13, scale13, packed2, scale2 = self._stacked_int4_experts(
            layer
        )
        return routed_moe_grouped_int4(
            x,
            packed13,
            scale13,
            packed2,
            scale2,
            flat_idx,
            flat_gate,
            group_size=self.int4_group,
            num_experts=layer.gate_weight.shape[0],
            gate_softcap=spec.mlp_gate_softcap,
            up_softcap=spec.mlp_up_softcap,
            interleaved=self.w13_interleaved,
            cap_fn=capped_swiglu_cuda,
            workspace=self._int4_workspace,
        )

    def _stacked_int4_experts(
        self, layer: LayerWeights
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return a layer's int4 experts as stacked ``[E, ...]`` tensors.

        CUDA loads already produce the stacked layout during conversion;
        a CPU load keeps per-expert lists for the native op and is stacked
        here on first use.  Slicing the stacked result keeps the eager
        segmented path working unchanged.
        """

        cached = self._stacked_int4.get(layer.index)
        if cached is not None:
            return cached
        if isinstance(layer.w13_packed, list):
            packed13, scale13 = stack_int4_experts(
                list(layer.w13_packed), list(layer.w13_packed_scale)
            )
            packed2, scale2 = stack_int4_experts(
                list(layer.w2_packed), list(layer.w2_packed_scale)
            )
            layer.w13_packed = packed13
            layer.w13_packed_scale = scale13
            layer.w2_packed = packed2
            layer.w2_packed_scale = scale2
        cached = (
            layer.w13_packed,
            layer.w13_packed_scale,
            layer.w2_packed,
            layer.w2_packed_scale,
        )
        self._stacked_int4[layer.index] = cached
        return cached

    # -- message controller ------------------------------------------------

    def _gate_message(
        self,
        controller: tuple[Tensor, Tensor, Tensor],
        destination: Tensor,
        message: Tensor,
    ) -> Tensor:
        down, forget, gain = controller
        stacked = tensorplay.cat(
            (
                rms_norm(
                    destination,
                    None,
                    self.eps,
                    output_dtype=tensorplay.bfloat16,
                ),
                rms_norm(message, None, self.eps, output_dtype=tensorplay.bfloat16),
            ),
            dim=-1,
        )
        hidden = tensorplay.matmul(stacked, down.transpose(0, 1)).float()
        forget_gate = tensorplay.tanh(
            tensorplay.matmul(hidden, forget.transpose(0, 1))
        )
        gain_gate = tensorplay.tanh(tensorplay.matmul(hidden, gain.transpose(0, 1)))
        return (1.0 - forget_gate) * destination.float() + gain_gate * message.float()

    def _gate_messages(
        self,
        controllers: tuple[Tensor, Tensor, Tensor],
        destinations: tuple[Tensor, ...],
        messages: tuple[Tensor, ...],
        norms: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Batched equivalent of :meth:`_gate_message` for the three attention streams."""

        stacked_destination = tensorplay.stack(destinations, dim=0)
        stacked_message = tensorplay.stack(messages, dim=0)
        # Controller weights are stacked once when the layer is bound; doing
        # it here would rebuild three tensors on every loop visit.
        down, forget, gain = controllers
        combined = tensorplay.cat(
            (
                rms_norm(
                    stacked_destination,
                    None,
                    self.eps,
                    output_dtype=tensorplay.bfloat16,
                ),
                rms_norm(
                    stacked_message,
                    None,
                    self.eps,
                    output_dtype=tensorplay.bfloat16,
                ),
            ),
            dim=-1,
        )
        hidden = tensorplay.matmul(combined, down.transpose(1, 2)).float()
        forget_gate = tensorplay.tanh(
            tensorplay.matmul(hidden, forget.transpose(1, 2))
        )
        gain_gate = tensorplay.tanh(
            tensorplay.matmul(hidden, gain.transpose(1, 2))
        )
        gated = (
            (1.0 - forget_gate) * stacked_destination.float()
            + gain_gate * stacked_message.float()
        )
        normalized = rms_norm(
            gated,
            tensorplay.stack(norms, dim=0).unsqueeze(1),
            self.eps,
            output_dtype=norms[0].dtype,
        )
        return tuple(normalized[index] for index in range(normalized.shape[0]))

    def _messages(
        self,
        stored: Tensor,
        partial: Tensor,
        core_index: int,
        first: int,
        count: int,
    ) -> tuple[Tensor, ...]:
        queries = self.loop_score_weights[core_index, first : first + count]
        if stored is not None:
            mixed = depth_residual_mix(
                stored,
                queries,
                (partial,),
                None,
                None,
                normalize_values=True,
                eps=self.eps,
            )
            if mixed is not None:
                return tuple(mixed[index] for index in range(mixed.shape[0]))
        if stored is None or stored.shape[0] == 0:
            sources = partial.float().unsqueeze(0)
        else:
            sources = tensorplay.cat(
                (stored.float(), partial.float().unsqueeze(0)), dim=0
            )
        normalized = rms_norm(
            sources, None, self.eps, output_dtype=tensorplay.float32
        )
        logits = tensorplay.einsum("qh,sth->qst", queries, normalized)
        probabilities = tensorplay.softmax(logits, dim=1)
        mixed = tensorplay.einsum("qst,sth->qth", probabilities, normalized)
        return tuple(mixed[index] for index in range(mixed.shape[0]))

    # -- layer -------------------------------------------------------------

    def _layer_forward(
        self,
        layer: LayerWeights,
        *,
        visit: int,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        tau: Tensor | None,
        state: AttentionResidualState,
        messages: tuple[Tensor, ...] | None,
        core_index: int,
    ) -> None:
        tick = self._prof_start()
        queries = self.attn_queries.get(layer.index)
        if layer.is_local:
            norms = (layer.attn_norm,)
        else:
            norms = (layer.q_norm, layer.k_norm, layer.v_norm)
        if messages is None:
            streams = state.attention_input(layer.index, queries, norms)
        else:
            # A local layer has one mixture but still three inputs (attn, k,
            # v), so the mixture is replicated and each input reads its own
            # message.
            mixtures = state.attention_mixtures(layer.index, queries)
            if layer.is_local:
                mixtures = tuple(mixtures) * 3
                norms = tuple(norms) * 3
            streams = self._gate_messages(
                layer.stacked_loop_controllers,
                mixtures,
                messages[:3],
                norms,
            )
        tick = self._tick("attn_in", tick)
        attention = self._attention(
            layer,
            streams,
            visit=visit,
            batch=batch,
            cache=cache,
            tau=tau,
        )
        tick = self._tick("attn", tick)
        attention = self._sconv(
            attention,
            layer.attn_sconv,
            batch=batch,
            visit=visit,
            kind="attn",
            channels=self.hidden,
        )
        tick = self._tick("attn_sconv", tick)
        if messages is None:
            mlp_input = state.mlp_input(
                layer.index,
                attention,
                self.mlp_queries[layer.index],
                layer.mlp_norm,
            )
        else:
            mixture = state.mlp_mixture(
                layer.index, attention, self.mlp_queries[layer.index]
            )
            mixture = self._gate_message(
                layer.controllers["mlp"], mixture, messages[3]
            )
            mlp_input = rms_norm(
                mixture,
                layer.mlp_norm,
                self.eps,
                output_dtype=layer.mlp_norm.dtype,
            )
        tick = self._tick("mlp_in", tick)
        mlp = self._moe(layer, mlp_input)
        tick = self._tick("moe", tick)
        mlp = self._sconv(
            mlp,
            layer.mlp_sconv,
            batch=batch,
            visit=visit,
            kind="mlp",
            channels=self.hidden,
        )
        tick = self._tick("mlp_sconv", tick)
        if self.debug:
            print(
                f"[ink] visit={visit:2d} layer={layer.index:2d} pass={batch} "
                f"attn_rms={float(attention.float().square().mean().sqrt().item()):.4f} "
                f"mlp_rms={float(mlp.float().square().mean().sqrt().item()):.4f}"
            )
        self._tick("finalize", tick)
        state.finalize_layer(layer.index, attention, mlp)
        del core_index

    # -- entry point -------------------------------------------------------

    def forward(self, batch: ExecutionBatch, cache: PagedKVCache) -> Tensor:
        if not self.weights:
            raise Inkling2Error("Inkling-2 weights are not loaded")
        tick = self._prof_start()
        hidden = F.embedding(batch.input_ids, self.weights["embed.weight"])
        hidden = rms_norm(
            hidden,
            self.weights["embed_norm.weight"],
            self.eps,
            output_dtype=self.dtype,
        )
        tau = log_scale_tau(
            batch.positions,
            self.inkling.log_scaling_n_floor,
            self.inkling.log_scaling_alpha,
        )
        assert self.output_query is not None
        state = AttentionResidualState(self.inkling, hidden, self.output_query)
        tick = self._tick("embed", tick)
        entry: LoopEntry | None = None
        bank: tuple[Tensor, Tensor] | None = None
        core_index = 0
        for visit, (layer_index, pass_index) in enumerate(self.visits):
            layer = self.layers[layer_index]
            if self._mtp_features is not None and visit in (5, 23, 41):
                tap = (5, 23, 41).index(visit)
                self._mtp_features[: hidden.shape[0], tap, :].copy_(state.partial)
            if layer_index == self.core_start:
                if pass_index == 0:
                    entry = state.loop_entry(layer_index)
                else:
                    # Every core layer reads the previous pass's block bank
                    # through its own four pseudo-queries, so the bank is
                    # captured once per pass and mixed per layer below.
                    assert entry is not None
                    bank = state.loop_bank(entry)
                    state.rewind_loop(entry)
                core_index = 0
            elif layer_index == self.core_end:
                bank = None
            messages = None
            glue = self._prof_start()
            if bank is not None:
                stored, partial = bank
                # Every core layer owns one row of message pseudo-queries
                # (``core_index`` is its offset inside the core), so resolve
                # it from the layer itself instead of carrying a counter.
                core_index = layer_index - self.core_start
                messages = self._messages(
                    stored,
                    partial,
                    core_index,
                    0,
                    len(LOOP_INPUTS),
                )
            self._layer_forward(
                layer,
                visit=visit,
                batch=batch,
                cache=cache,
                tau=tau,
                state=state,
                messages=messages,
                core_index=core_index,
            )
            self._tick("loop_glue", glue)
        if self.debug:
            print(
                f"[ink] final state rms={float(state.partial().float().square().mean().sqrt().item()):.4f}"
            )
        tick = self._prof_start()
        hidden = state.compute_output(self.weights["norm.weight"])
        if self._mtp_features is not None:
            self._mtp_features[: hidden.shape[0], 3, :].copy_(hidden)
        tick = self._tick("output", tick)
        logits = F.linear(hidden, self.weights["unembed.weight"])
        tick = self._tick("lm_head", tick)
        if self.debug:
            values = logits.reshape(-1).float()
            print(
                f"[ink] logits rms={float(values.square().mean().sqrt().item()):.4f} "
                f"absmax={float(values.abs().max().item()):.4f} "
                f"last5={[round(v, 3) for v in values.reshape(-1, logits.shape[-1])[-1].tolist()[:5]]}"
            )
        tick = self._tick("softcap", tick)
        if self.inkling.final_logit_softcap:
            logits = softcap(logits, self.inkling.final_logit_softcap)
        # sglang masks the padded tail of the vocabulary so sampling cannot
        # emit an id outside the unpadded vocabulary.
        unpadded_vocab_size = getattr(
            self.inkling, "unpadded_vocab_size", self.inkling.vocab_size
        )
        if unpadded_vocab_size < logits.shape[-1]:
            logits[..., unpadded_vocab_size:] = float("-inf")
        self._prof_report(len(self.visits))
        if batch.all_decode:
            return logits
        sample_rows = batch.sample_indices.clamp_min(0).long()
        sampled = logits.index_select(0, sample_rows)
        valid = batch.sample_indices >= 0
        return tensorplay.where(
            valid.reshape(-1, 1), sampled, tensorplay.zeros_like(sampled)
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: tensorplay.device | str = "cuda",
        dtype: tensorplay.dtype = tensorplay.bfloat16,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
        enable_mtp: bool = False,
    ) -> tuple[Inkling2ForCausalLM, WeightLoadReport]:
        config = load_inkling2_config(model_path)
        model = cls(
            config,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
            enable_mtp=enable_mtp,
        )
        model.allocate_weights()
        report = model.load_weights(model_path)
        if enable_mtp:
            from megalm.models.inkling2.mtp import Inkling2DSparkPredictor

            model.mtp = Inkling2DSparkPredictor(
                model_path,
                embed_weight=model.weights["embed.weight"],
                embed_norm_weight=model.weights["embed_norm.weight"],
                lm_head_weight=model.weights["unembed.weight"],
                device=model.device,
                dtype=model.dtype,
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
            )
        model.eval()
        return model, report


__all__ = [
    "AttentionResidualState",
    "Inkling2Error",
    "Inkling2ForCausalLM",
    "LayerWeights",
    "LoopEntry",
    "capped_swiglu",
    "l2_clip",
    "log_scale_tau",
    "residual_sum",
    "rms_norm",
    "softcap",
]
