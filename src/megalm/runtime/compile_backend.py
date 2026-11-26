"""Opt-in Stax compile provider for static text-model regions.

Stax is a lowering provider, not a replacement for decode graph replay.  A
region enters the engine only after these cold-path checks pass:

1. ``tensorplay.compile(backend="stax", strict_native=True)`` returns a native
   artifact instead of silently falling back to the graph executor.
2. The artifact matches the eager kernel output within the dtype tolerance.
3. The artifact returns the expected shape for the exact bucket (symbolic
   Stax shapes are not used; see ``docs/design/compile-backend.md``).
4. The artifact can be captured and replayed inside a CUDA Graph.
5. The artifact beats ``min_speedup`` on that bucket (median of paired runs).

Any failure rejects the region and leaves the eager path untouched, so an
enabled provider can cost startup time but cannot change serving semantics.
This follows the engine's admission rules: no runtime compilation or host
synchronization on the forward path.  A compiled call allocates its output like
the eager path does; static-address CUDA Graph capture stays safe because the
dispatch never selects a compiled artifact for decode.

Regions:

* ``dense_mlp`` — SwiGLU dense MLPs across model families (``down_proj``
  module, ``down_proj_weight`` tensor or ``down_weight`` tensor layouts).
* ``attn_proj`` — the ``qkv_proj`` and ``o_proj`` linear maps of attention
  modules.  The attention kernel itself stays a hand-written kernel.
* ``vision_mlp`` — SwiGLU MLPs inside multimodal vision towers, opted in
  separately because they run in the encode phase rather than text prefill.

Decode stays on the hand-written kernels: the provider compiles one concrete
artifact per exact token bucket (see ``docs/design/compile-backend.md``) and
dispatches only those sizes.  Equal-shaped modules share one artifact and one
admission decision, because weights are passed as runtime inputs.  A startup
deadline and a device-memory budget bound the cold path, and a compiled call
that fails retires its bucket and falls back to eager instead of failing
serving.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import logging
from pathlib import Path
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.observability import log_event

_LOGGER = logging.getLogger("megalm.runtime.compile")


def compiled_identity_digest(
    *,
    backend: str,
    regions: tuple[str, ...],
    buckets: tuple[int, ...],
    dtype: str,
    device_name: str,
    model_digest: str,
    tolerance: float | None,
    min_speedup: float,
) -> str:
    """Bind the compile provider's implementation and policy to one digest.

    Computed during DISCOVER so the draft and the seal derive the same value
    without compiling anything. Reading the provider source makes a lowering
    change a detectable artifact drift.
    """

    digest = hashlib.sha256()
    source = Path(__file__).read_bytes()
    digest.update(hashlib.sha256(source).digest())
    parts = (
        backend,
        ",".join(sorted(regions)),
        ",".join(str(bucket) for bucket in sorted(buckets)),
        dtype,
        device_name,
        model_digest,
        "" if tolerance is None else repr(tolerance),
        repr(min_speedup),
    )
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


COMPILE_BACKENDS = ("none", "stax")
COMPILE_REGIONS = ("dense_mlp", "attn_proj", "vision_mlp")

_NON_TEXT_TOWERS = ("vision", "audio", "encoder", "tower", "merger")

CompileFn = Callable[[Callable[..., Any]], Callable[..., Any]]


class CompileBackendError(RuntimeError):
    """Raised when a compile policy is malformed."""


@dataclass(slots=True)
class _Decision:
    """Admission outcome for one shape signature, reused by every module."""

    buckets: dict[int, Any]
    output_tail: tuple[int, ...] | None
    error: float | None
    speedup: float | None
    notes: str


@dataclass(frozen=True, slots=True)
class CompilePolicy:
    """Cold-path policy for the Stax provider."""

    backend: str = "none"
    regions: tuple[str, ...] = COMPILE_REGIONS
    tolerance: float | None = None
    min_speedup: float = 1.05
    min_tokens: int = 1024
    candidate_tokens: tuple[int, ...] = (1024,)
    verify_capture: bool = True
    startup_budget_seconds: float = 0.0
    max_artifact_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))
        if self.backend not in COMPILE_BACKENDS:
            raise CompileBackendError(
                f"compile backend must be one of {COMPILE_BACKENDS}"
            )
        unknown = set(self.regions) - set(COMPILE_REGIONS)
        if unknown:
            raise CompileBackendError(f"unknown compile regions: {sorted(unknown)}")
        if self.backend != "none" and not self.regions:
            raise CompileBackendError("a compile backend requires regions")
        if self.tolerance is not None and self.tolerance <= 0.0:
            raise CompileBackendError("compile tolerance must be positive")
        if self.min_speedup < 0.0:
            raise CompileBackendError("compile min_speedup cannot be negative")
        if self.min_tokens <= 0:
            raise CompileBackendError("compile min_tokens must be positive")
        if any(tokens <= 0 for tokens in self.candidate_tokens):
            raise CompileBackendError("compile token buckets must be positive")
        if self.startup_budget_seconds < 0.0:
            raise CompileBackendError(
                "compile startup_budget_seconds cannot be negative"
            )
        if self.max_artifact_bytes < 0:
            raise CompileBackendError("compile max_artifact_bytes cannot be negative")
        object.__setattr__(
            self,
            "candidate_tokens",
            tuple(sorted(set(self.candidate_tokens))),
        )
        if not self.candidate_tokens:
            raise CompileBackendError("compile candidate_tokens cannot be empty")
        if any(tokens <= 0 for tokens in self.candidate_tokens):
            raise CompileBackendError("compile candidate_tokens must be positive")

    @property
    def enabled(self) -> bool:
        return self.backend != "none"


@dataclass(slots=True)
class RegionEvidence:
    """Admission evidence for one candidate region."""

    region: str
    accepted: bool
    reason: str = ""
    compile_seconds: float = 0.0
    max_relative_error: float | None = None
    speedup: float | None = None
    dispatch_tokens: int | None = None
    artifact_bytes: int | None = None
    buckets: tuple[int, ...] = ()
    runtime_calls: int = 0
    runtime_fallbacks: int = 0


@dataclass(slots=True)
class CompileReport:
    """Outcome of the cold-path compile pass."""

    backend: str = "none"
    attempted: int = 0
    accepted: int = 0
    compile_seconds: float = 0.0
    artifact_bytes: int = 0
    memory_bytes: int = 0
    regions: tuple[RegionEvidence, ...] = ()

    @classmethod
    def disabled(cls) -> CompileReport:
        return cls()

    @classmethod
    def failed(cls, backend: str, reason: str) -> CompileReport:
        return cls(
            backend=backend,
            attempted=1,
            accepted=0,
            regions=(RegionEvidence(region="startup", accepted=False, reason=reason),),
        )

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "compile_seconds": round(self.compile_seconds, 3),
            "artifact_bytes": self.artifact_bytes,
            "memory_bytes": self.memory_bytes,
            "regions": [
                {
                    "region": item.region,
                    "accepted": item.accepted,
                    "reason": item.reason,
                    "compile_seconds": round(item.compile_seconds, 3),
                    "max_relative_error": item.max_relative_error,
                    "speedup": item.speedup,
                    "dispatch_tokens": item.dispatch_tokens,
                    "artifact_bytes": item.artifact_bytes,
                    "buckets": list(item.buckets),
                    "runtime_calls": item.runtime_calls,
                    "runtime_fallbacks": item.runtime_fallbacks,
                }
                for item in self.regions
            ],
        }


@dataclass(frozen=True, slots=True)
class _LinearBinding:
    weight: Any
    bias: Any | None = None


@dataclass(frozen=True, slots=True)
class _MLPBinding:
    gate_up: _LinearBinding
    down: _LinearBinding
    intermediate: int


def _linear_binding(module: Any) -> _LinearBinding | None:
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    return _LinearBinding(weight=weight, bias=getattr(module, "bias", None))


def _dense_mlp_binding(module: Any) -> _MLPBinding | None:
    if not hasattr(module, "_silu_and_mul"):
        return None
    if hasattr(module, "experts"):
        return None
    if getattr(module, "lora", None) is not None:
        return None
    if getattr(module, "gate_up_int4", None) is not None:
        return None
    if getattr(module, "down_proj_int4", None) is not None:
        return None
    # A model that pins a specific linear kernel (for example the
    # batch-invariant path) keeps it; Stax must not silently swap it.
    linear = getattr(module, "_linear", None)
    if linear is not None and linear is not F.linear:
        return None

    if hasattr(module, "gate_up_proj"):
        gate_up = _linear_binding(module.gate_up_proj)
    elif hasattr(module, "gate_up_weight"):
        gate_up = _LinearBinding(
            weight=module.gate_up_weight,
            bias=getattr(module, "gate_up_bias", None),
        )
    else:
        return None

    if hasattr(module, "down_proj"):
        down = _linear_binding(module.down_proj)
    elif hasattr(module, "down_proj_weight"):
        down = _LinearBinding(
            weight=module.down_proj_weight,
            bias=getattr(module, "down_proj_bias", None),
        )
    elif hasattr(module, "down_weight"):
        down = _LinearBinding(
            weight=module.down_weight,
            bias=getattr(module, "down_bias", None),
        )
    else:
        return None
    if gate_up is None or down is None:
        return None
    return _MLPBinding(
        gate_up=gate_up,
        down=down,
        intermediate=int(gate_up.weight.shape[0]) // 2,
    )


def _named_modules(model: Any) -> Iterable[tuple[str, Any]]:
    named = getattr(model, "named_modules", None)
    if callable(named):
        yield from named()
        return
    stack: list[tuple[str, Any]] = [("", model)]
    while stack:
        prefix, module = stack.pop()
        yield prefix, module
        children = getattr(module, "_modules", None)
        if isinstance(children, dict):
            for name, child in children.items():
                if child is not None:
                    stack.append((f"{prefix}.{name}" if prefix else name, child))


def _runtime_alias(value: Any) -> Any:
    """Return a storage-sharing alias that Stax treats as a runtime input.

    Stax folds a tensor that does not require grad into the artifact as a
    constant.  For model weights that means the compiled region keeps its own
    copy on device and pins the math to the values present at startup.  A
    ``requires_grad`` alias shares storage, so the artifact reads the live
    weights instead: no per-region weight copy, weight updates stay visible,
    and equal-shaped regions share code-cache entries across layers.
    """

    if value is None or getattr(value, "requires_grad", False):
        return value
    return nn.Parameter(value.detach(), requires_grad=True)


def _functional_mlp(
    binding: _MLPBinding,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Express the SwiGLU MLP with lowerable tensorplay ops.

    The eager module uses the fused ``silu_and_mul`` Triton kernel.  A compiled
    region must expose the same math as ordinary tensorplay ops so Stax can
    fuse it; the parity gate below is what keeps the replacement honest.
    Weights are explicit runtime inputs (see ``_runtime_alias``).
    """

    gate_up_weight = _runtime_alias(binding.gate_up.weight)
    gate_up_bias = _runtime_alias(binding.gate_up.bias)
    down_weight = _runtime_alias(binding.down.weight)
    down_bias = _runtime_alias(binding.down.bias)
    intermediate = binding.intermediate

    def fn(
        x: Any,
        gate_up_weight: Any,
        gate_up_bias: Any,
        down_weight: Any,
        down_bias: Any,
    ) -> Any:
        gate_up_value = F.linear(x, gate_up_weight, gate_up_bias)
        gate = gate_up_value[:, :intermediate]
        up = gate_up_value[:, intermediate:]
        return F.linear(
            (gate * gate.sigmoid()) * up,
            down_weight,
            down_bias,
        )

    return fn, (gate_up_weight, gate_up_bias, down_weight, down_bias)


def _functional_linear(
    binding: _LinearBinding,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    weight = _runtime_alias(binding.weight)
    bias = _runtime_alias(binding.bias)

    def fn(x: Any, weight: Any, bias: Any) -> Any:
        return F.linear(x, weight, bias)

    return fn, (weight, bias)


def _forward_parameters(module: Any) -> Any:
    try:
        return inspect.signature(module.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return {}


def _call_eager(module: Any, x: Any) -> Any:
    parameters = _forward_parameters(module)
    kwargs: dict[str, Any] = {}
    if "decode" in parameters:
        kwargs["decode"] = False
    if "lora_slots" in parameters:
        kwargs["lora_slots"] = None
    return module(x, **kwargs)


def _eager_callable(module: Any) -> Callable[[Any], Any]:
    """Bind the eager module once so timing does not charge it introspection."""

    parameters = _forward_parameters(module)
    kwargs: dict[str, Any] = {}
    if "decode" in parameters:
        kwargs["decode"] = False
    if "lora_slots" in parameters:
        kwargs["lora_slots"] = None
    return lambda x: module(x, **kwargs)


def _default_compile(fn: Callable[..., Any]) -> Callable[..., Any]:
    # Concrete shapes only.  A symbolic (dynamic) Stax artifact currently
    # returns the largest shape it has seen (and truncates larger inputs), so
    # it cannot be dispatched from a variable-size prefill path.  The provider
    # compiles one concrete artifact per token bucket instead.
    return tensorplay.compile(
        fn,
        backend="stax",
        strict_native=True,
    )


def _sync() -> None:
    if not tensorplay.cuda.is_available():
        return
    synchronize = getattr(tensorplay.cuda, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _time_call(fn: Callable[[Any], Any], x: Any, iterations: int = 20) -> float:
    # Serving always runs inference mode.  Timing under autograd would charge
    # both paths for graph construction and can retain large intermediates
    # (the weights are ``requires_grad`` aliases), so measure the mode that is
    # actually deployed.
    with tensorplay.inference_mode():
        for _ in range(3):
            fn(x)
        _sync()
        started = time.perf_counter()
        for _ in range(iterations):
            fn(x)
        _sync()
    return (time.perf_counter() - started) / iterations


def _speedup_ratio(
    eager: Callable[[Any], Any],
    compiled: Callable[[Any], Any],
    value: Any,
    iterations: int = 30,
) -> float:
    """Median of seven paired timings.

    Single-sample ratios swing by several percent on a shared GPU.  The median
    admits a region whose typical performance meets the threshold and rejects
    one whose win is only noise.
    """

    ratios = []
    for _ in range(7):
        eager_seconds = _time_call(eager, value, iterations=iterations)
        compiled_seconds = _time_call(compiled, value, iterations=iterations)
        ratios.append(eager_seconds / max(compiled_seconds, 1e-9))
    ratios.sort()
    return ratios[len(ratios) // 2]


def _relative_error(actual: Any, expected: Any) -> float:
    diff = float((actual.float() - expected.float()).abs().max())
    scale = max(float(expected.float().abs().max()), 1.0)
    return diff / scale


def _capture_safe(
    compiled: Callable[..., Any],
    probe: Any,
    params: tuple[Any, ...],
) -> bool:
    if not tensorplay.cuda.is_available():
        return True
    pool = tensorplay.cuda.graph_pool_handle()
    stream = tensorplay.cuda.Stream()
    current = tensorplay.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with tensorplay.cuda.stream(stream), tensorplay.inference_mode():
            for _ in range(2):
                compiled(probe, *params)
        stream.synchronize()
        current.wait_stream(stream)
        graph = tensorplay.cuda.CUDAGraph()
        with (
            tensorplay.cuda.graph(graph, pool=pool, stream=stream),
            tensorplay.inference_mode(),
        ):
            compiled(probe, *params)
        with tensorplay.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        graph.reset()
        return True
    except Exception:  # noqa: BLE001 - a failed probe rejects the region
        return False
    finally:
        current.wait_stream(stream)


def _resolve_tolerance(policy: CompilePolicy, dtype: str) -> float:
    if policy.tolerance is not None:
        return policy.tolerance
    return 2e-2 if dtype == "bfloat16" else 5e-3


def _reject(region: str, reason: str, elapsed: float = 0.0) -> RegionEvidence:
    return RegionEvidence(
        region=region,
        accepted=False,
        reason=reason,
        compile_seconds=elapsed,
    )


class StaxCompileProvider:
    """Compile admitted text-model regions and report the evidence."""

    def __init__(
        self,
        policy: CompilePolicy,
        *,
        compile_fn: CompileFn | None = None,
    ) -> None:
        self.policy = policy
        self._compile_fn = compile_fn or _default_compile
        self._deadline: float | None = None
        self._artifacts: dict[Any, tuple[Callable[..., Any], int]] = {}
        self._decisions: dict[Any, _Decision] = {}
        self._artifact_bytes_total = 0
        self._artifact_budget_exceeded = False
        self._baseline_alloc: int | None = None

    def _over_deadline(self) -> bool:
        deadline = self._deadline
        return deadline is not None and time.perf_counter() > deadline

    def _over_memory_budget(self) -> bool:
        budget = self.policy.max_artifact_bytes
        if budget <= 0 or self._baseline_alloc is None:
            return self._artifact_budget_exceeded
        if self._artifact_budget_exceeded:
            return True
        if not tensorplay.cuda.is_available():
            return False
        growth = tensorplay.cuda.memory_allocated() - self._baseline_alloc
        if growth > budget:
            self._artifact_budget_exceeded = True
            log_event(
                _LOGGER,
                logging.WARNING,
                "engine.compile.memory_budget",
                used_bytes=growth,
                budget_bytes=budget,
            )
            return True
        return False

    def apply(
        self,
        model: Any,
        *,
        dtype: str,
        device: Any,
        enable_lora: bool = False,
    ) -> CompileReport:
        if not self.policy.enabled:
            return CompileReport.disabled()
        if enable_lora:
            return CompileReport(
                backend=self.policy.backend,
                attempted=len(self.policy.regions),
                regions=tuple(
                    _reject(
                        region,
                        "LoRA adapters keep the projection path mutable",
                    )
                    for region in self.policy.regions
                ),
            )
        if getattr(device, "type", None) != "cuda":
            return CompileReport(
                backend=self.policy.backend,
                attempted=len(self.policy.regions),
                regions=tuple(
                    _reject(region, "Stax regions require a CUDA device")
                    for region in self.policy.regions
                ),
            )

        started = time.perf_counter()
        budget = self.policy.startup_budget_seconds
        self._deadline = started + budget if budget > 0.0 else None
        self._baseline_alloc = (
            tensorplay.cuda.memory_allocated()
            if tensorplay.cuda.is_available()
            else None
        )
        evidence: list[RegionEvidence] = []
        if "dense_mlp" in self.policy.regions:
            evidence.extend(self._apply_dense_mlps(model, dtype))
        if "attn_proj" in self.policy.regions:
            evidence.extend(self._apply_attention_projections(model, dtype))
        if "vision_mlp" in self.policy.regions:
            evidence.extend(self._apply_vision_mlps(model, dtype))
        total = time.perf_counter() - started
        return CompileReport(
            backend=self.policy.backend,
            attempted=len(evidence),
            accepted=sum(1 for item in evidence if item.accepted),
            compile_seconds=total,
            artifact_bytes=self._artifact_bytes_total,
            memory_bytes=(
                max(
                    0,
                    tensorplay.cuda.memory_allocated() - self._baseline_alloc,
                )
                if self._baseline_alloc is not None and tensorplay.cuda.is_available()
                else 0
            ),
            regions=tuple(evidence),
        )

    def _apply_dense_mlps(self, model: Any, dtype: str) -> list[RegionEvidence]:
        evidence: list[RegionEvidence] = []
        for name, module in _named_modules(model):
            binding = _dense_mlp_binding(module)
            if binding is None:
                continue
            if any(part in name for part in _NON_TEXT_TOWERS):
                continue
            evidence.append(
                self._admit(
                    name or "<root>",
                    module,
                    lambda binding=binding: _functional_mlp(binding),
                    int(binding.gate_up.weight.shape[1]),
                    dtype,
                    (
                        "mlp",
                        int(binding.gate_up.weight.shape[1]),
                        binding.intermediate,
                        binding.gate_up.bias is not None,
                        binding.down.bias is not None,
                    ),
                )
            )
        return evidence

    def _apply_attention_projections(
        self,
        model: Any,
        dtype: str,
    ) -> list[RegionEvidence]:
        evidence: list[RegionEvidence] = []
        for name, module in _named_modules(model):
            if not (hasattr(module, "qkv_proj") and hasattr(module, "o_proj")):
                continue
            if any(part in name for part in _NON_TEXT_TOWERS):
                continue
            for attribute in ("qkv_proj", "o_proj"):
                projection = getattr(module, attribute)
                binding = _linear_binding(projection)
                if binding is None:
                    evidence.append(
                        _reject(
                            f"{name}.{attribute}",
                            "projection uses a specialized decode kernel",
                        )
                    )
                    continue
                evidence.append(
                    self._admit(
                        f"{name}.{attribute}",
                        projection,
                        lambda binding=binding: _functional_linear(binding),
                        int(binding.weight.shape[1]),
                        dtype,
                        (
                            "linear",
                            int(binding.weight.shape[1]),
                            int(binding.weight.shape[0]),
                            binding.bias is not None,
                        ),
                    )
                )
        return evidence

    def _apply_vision_mlps(self, model: Any, dtype: str) -> list[RegionEvidence]:
        evidence: list[RegionEvidence] = []
        for name, module in _named_modules(model):
            if not any(part in name for part in _NON_TEXT_TOWERS):
                continue
            binding = _dense_mlp_binding(module)
            if binding is None:
                continue
            evidence.append(
                self._admit(
                    name,
                    module,
                    lambda binding=binding: _functional_mlp(binding),
                    int(binding.gate_up.weight.shape[1]),
                    dtype,
                    (
                        "mlp",
                        int(binding.gate_up.weight.shape[1]),
                        binding.intermediate,
                        binding.gate_up.bias is not None,
                        binding.down.bias is not None,
                    ),
                )
            )
        return evidence

    def _artifact(
        self,
        key: Any,
        fn: Callable[..., Any],
    ) -> tuple[Callable[..., Any], int]:
        """Compile one artifact per shape signature, shared across modules.

        Weights arrive as runtime inputs, so the same artifact serves every
        layer with the same width and bucket.  A model typically needs one
        artifact per bucket rather than one per layer, which is what keeps the
        (large) per-artifact workspace bounded.
        """

        if self._artifact_budget_exceeded:
            raise RuntimeError("artifact memory budget already exceeded")
        cached = self._artifacts.get(key)
        if cached is not None:
            return cached
        # The artifact's memory appears on its first call (lazy lowering), so
        # the caller records the size after that call; until then it is 0.
        compiled = self._compile_fn(fn)
        self._artifacts[key] = (compiled, 0)
        return compiled, 0

    def _record_artifact_bytes(self, key: Any, size: int) -> None:
        entry = self._artifacts.get(key)
        if entry is None:
            return
        previous = entry[1]
        self._artifacts[key] = (entry[0], size)
        self._artifact_bytes_total += size - previous
        budget = self.policy.max_artifact_bytes
        if budget > 0 and self._artifact_bytes_total > budget:
            self._release_artifact(key)
            self._artifact_budget_exceeded = True
            raise RuntimeError(
                f"artifact needs {size} bytes, over the {budget}-byte compile budget"
            )

    def _release_artifact(self, key: Any) -> None:
        entry = self._artifacts.pop(key, None)
        if entry is not None:
            self._artifact_bytes_total = max(
                0,
                self._artifact_bytes_total - entry[1],
            )
            # A rejected bucket leaves the artifact's runtime pool behind; a
            # cold-path sweep returns what the allocator can give back before
            # the next region is considered.
            gc.collect()
            if tensorplay.cuda.is_available():
                tensorplay.cuda.empty_cache()

    def _admit(
        self,
        region: str,
        module: Any,
        builder: Callable[[], tuple[Callable[..., Any], tuple[Any, ...]]],
        width: int,
        dtype: str,
        signature: tuple[Any, ...] = (),
    ) -> RegionEvidence:
        started = time.perf_counter()
        if getattr(module, "_stax_compiled", False):
            return _reject(region, "already compiled by an earlier pass")
        if self._over_deadline():
            return _reject(
                region,
                "compile startup budget exhausted before this region",
            )
        if self._over_memory_budget():
            return _reject(
                region,
                "compile device memory budget already exceeded",
            )
        decision_key = (*signature, dtype)
        decision = self._decisions.get(decision_key)
        if decision is not None:
            if not decision.buckets:
                return _reject(
                    region,
                    decision.notes or "no token bucket was admitted",
                    time.perf_counter() - started,
                )
            try:
                fn, params = builder()
                tokens = min(decision.buckets)
                value = tensorplay.randn(
                    (tokens, width),
                    dtype=self._module_dtype(module),
                    device=self._module_device(module),
                )
                with tensorplay.inference_mode():
                    reference = _call_eager(module, value)
                    actual = decision.buckets[tokens](value, *params)
                if tuple(int(dim) for dim in actual.shape) != tuple(
                    int(dim) for dim in reference.shape
                ) or _relative_error(actual, reference) > _resolve_tolerance(
                    self.policy, dtype
                ):
                    return _reject(
                        region,
                        "shared artifact parity check failed on this module",
                        time.perf_counter() - started,
                    )
            except Exception as exc:  # noqa: BLE001 - keep eager on failure
                return _reject(
                    region,
                    f"{type(exc).__name__}: {exc}",
                    time.perf_counter() - started,
                )
            evidence = RegionEvidence(
                region=region,
                accepted=True,
                reason=decision.notes,
                compile_seconds=time.perf_counter() - started,
                max_relative_error=decision.error,
                speedup=decision.speedup,
                dispatch_tokens=min(decision.buckets),
                artifact_bytes=0,
                buckets=tuple(sorted(decision.buckets)),
            )
            self._install(
                module,
                dict(decision.buckets),
                params,
                decision.output_tail,
                evidence,
            )
            return evidence

        try:
            fn, params = builder()
            cuda = tensorplay.cuda.is_available()
            compiled_by_tokens: dict[int, Callable[..., Any]] = {}
            notes: list[str] = []
            errors: list[float] = []
            speedups: list[float] = []
            accepted_bytes = 0
            output_tail: tuple[int, ...] | None = None
            tolerance = _resolve_tolerance(self.policy, dtype)

            for tokens in self.policy.candidate_tokens:
                if self._over_deadline():
                    notes.append(f"{tokens}: startup budget exhausted")
                    break
                if self._over_memory_budget():
                    notes.append(f"{tokens}: device memory budget exhausted")
                    break
                if tokens < self.policy.min_tokens:
                    notes.append(
                        f"{tokens}: below the {self.policy.min_tokens}-token "
                        "compile floor"
                    )
                    continue
                value = tensorplay.randn(
                    (tokens, width),
                    dtype=self._module_dtype(module),
                    device=self._module_device(module),
                )
                with tensorplay.inference_mode():
                    reference = _call_eager(module, value)
                if output_tail is None:
                    output_tail = tuple(int(dim) for dim in reference.shape[1:])
                artifact_key = (*decision_key, tokens)
                is_new = artifact_key not in self._artifacts
                if is_new and cuda:
                    _sync()
                    allocated_before = tensorplay.cuda.memory_allocated()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with tensorplay.inference_mode():
                            compiled, _ = self._artifact(artifact_key, fn)
                            actual = compiled(value, *params)
                except Exception as exc:  # noqa: BLE001 - reject this bucket
                    notes.append(f"{tokens}: {type(exc).__name__}: {exc}")
                    if is_new:
                        self._release_artifact(artifact_key)
                    if self._artifact_budget_exceeded:
                        break
                    continue
                if is_new and cuda:
                    _sync()
                    try:
                        self._record_artifact_bytes(
                            artifact_key,
                            max(
                                0,
                                tensorplay.cuda.memory_allocated() - allocated_before,
                            ),
                        )
                    except RuntimeError as exc:
                        notes.append(f"{tokens}: {exc}")
                        break
                    if self._over_memory_budget():
                        notes.append(f"{tokens}: device memory budget exhausted")
                        self._release_artifact(artifact_key)
                        break

                def reject_bucket(
                    message: str,
                    is_new: bool = is_new,
                    artifact_key: Any = artifact_key,
                ) -> None:
                    notes.append(message)
                    if is_new:
                        self._release_artifact(artifact_key)

                fallback = [
                    str(item.message)
                    for item in caught
                    if "Stax lowering failed" in str(item.message)
                ]
                if fallback:
                    reject_bucket(f"{tokens}: native lowering missing: {fallback[0]}")
                    break
                if tuple(int(dim) for dim in actual.shape) != tuple(
                    int(dim) for dim in reference.shape
                ):
                    reject_bucket(
                        f"{tokens}: compiled shape {tuple(actual.shape)} != "
                        f"eager shape {tuple(reference.shape)}"
                    )
                    continue
                error = _relative_error(actual, reference)
                if error > tolerance:
                    reject_bucket(
                        f"{tokens}: parity {error:.3e} exceeds "
                        f"tolerance {tolerance:.3e}"
                    )
                    continue
                if self.policy.verify_capture:
                    if not _capture_safe(compiled, value, params):
                        reject_bucket(
                            f"{tokens}: artifact cannot be captured and replayed "
                            "in a CUDA Graph"
                        )
                        continue
                    if self._over_memory_budget():
                        reject_bucket(
                            f"{tokens}: capture pushed the pass over the device "
                            "memory budget"
                        )
                        break
                speedup = _speedup_ratio(
                    _eager_callable(module),
                    lambda item, compiled=compiled: compiled(item, *params),
                    value,
                )
                if self.policy.min_speedup > 0.0 and speedup < self.policy.min_speedup:
                    reject_bucket(
                        f"{tokens}: speedup {speedup:.3f} below "
                        f"{self.policy.min_speedup:.3f}"
                    )
                    continue
                compiled_by_tokens[tokens] = compiled
                accepted_bytes += self._artifacts.get(artifact_key, (None, 0))[1]
                errors.append(error)
                speedups.append(speedup)

            if compiled_by_tokens and self._over_memory_budget():
                for tokens in compiled_by_tokens:
                    self._release_artifact((*decision_key, tokens))
                self._decisions[decision_key] = _Decision(
                    buckets={},
                    output_tail=None,
                    error=None,
                    speedup=None,
                    notes="compile device memory budget exceeded",
                )
                return _reject(
                    region,
                    "compile device memory budget exceeded after probing",
                    time.perf_counter() - started,
                )
            notes_text = "; ".join(notes[:6])
            if not compiled_by_tokens:
                self._decisions[decision_key] = _Decision(
                    buckets={},
                    output_tail=None,
                    error=None,
                    speedup=None,
                    notes=notes_text or "no token bucket was admitted",
                )
                return _reject(
                    region,
                    notes_text or "no token bucket was admitted",
                    time.perf_counter() - started,
                )
            accepted_tokens = tuple(sorted(compiled_by_tokens))
            evidence = RegionEvidence(
                region=region,
                accepted=True,
                reason=notes_text,
                compile_seconds=time.perf_counter() - started,
                max_relative_error=max(errors),
                # The worst accepted bucket: a region is never reported faster
                # than a shape it actually dispatches.
                speedup=min(speedups),
                dispatch_tokens=accepted_tokens[0],
                artifact_bytes=accepted_bytes,
                buckets=accepted_tokens,
            )
            self._decisions[decision_key] = _Decision(
                buckets=dict(compiled_by_tokens),
                output_tail=output_tail,
                error=max(errors),
                speedup=min(speedups),
                notes=notes_text,
            )
            self._install(
                module,
                compiled_by_tokens,
                params,
                output_tail,
                evidence,
            )
            return evidence
        except Exception as exc:  # noqa: BLE001 - cold path keeps eager
            return _reject(
                region,
                f"{type(exc).__name__}: {exc}",
                time.perf_counter() - started,
            )

    @staticmethod
    def _module_device(module: Any) -> Any:
        weight = getattr(module, "gate_up_weight", None)
        if weight is None:
            weight = getattr(module, "weight", None)
        if weight is None:
            weight = getattr(module, "down_proj_weight", None)
        if weight is None:
            weight = getattr(module, "down_weight", None)
        return getattr(weight, "device", None)

    @staticmethod
    def _module_dtype(module: Any) -> Any:
        weight = getattr(module, "gate_up_weight", None)
        if weight is None:
            weight = getattr(module, "weight", None)
        if weight is None:
            weight = getattr(module, "down_proj_weight", None)
        if weight is None:
            weight = getattr(module, "down_weight", None)
        return getattr(weight, "dtype", None)

    @staticmethod
    def _install(
        module: Any,
        compiled_by_tokens: dict[int, Callable[..., Any]],
        params: tuple[Any, ...],
        output_tail: tuple[int, ...] | None,
        evidence: RegionEvidence,
    ) -> None:
        original = module.forward
        parameters = _forward_parameters(module)
        accepts_decode = "decode" in parameters
        accepts_slots = "lora_slots" in parameters
        accepts_layer = "layer_index" in parameters

        def call_original(
            x: Any, decode: bool, lora_slots: Any, layer_index: int = 0
        ) -> Any:
            kwargs: dict[str, Any] = {}
            if accepts_decode:
                kwargs["decode"] = decode
            if accepts_slots:
                kwargs["lora_slots"] = lora_slots
            if accepts_layer:
                kwargs["layer_index"] = layer_index
            return original(x, **kwargs)

        def compiled_forward(
            x: Any,
            *,
            decode: bool = False,
            lora_slots: Any = None,
            layer_index: int = 0,
        ) -> Any:
            # Only precompiled exact bucket sizes run the compiled artifact.
            # Decode, other batch sizes, 3-D inputs and an adapter attached
            # after admission stay on the fused hand-written kernels.  Static
            # buckets also mean a captured decode graph can never contain a
            # compiled region.
            tokens = int(x.shape[0]) if x.ndim == 2 else -1
            compiled = None
            if not decode and tokens >= 0 and getattr(module, "lora", None) is None:
                compiled = compiled_by_tokens.get(tokens)
            if compiled is None:
                return call_original(x, decode, lora_slots, layer_index)
            try:
                result = compiled(x, *params)
                expected = (tokens, *(output_tail or ()))
                if tuple(int(dim) for dim in result.shape) != expected:
                    raise RuntimeError(
                        f"compiled output shape {tuple(result.shape)} != "
                        f"expected {expected}"
                    )
            except Exception as exc:  # noqa: BLE001 - serving must not fail
                # Retire the bucket instead of paying the failure on every
                # call, and record the first failure for the snapshot.
                compiled_by_tokens.pop(tokens, None)
                module._stax_buckets = tuple(sorted(compiled_by_tokens))
                evidence.buckets = module._stax_buckets
                if not compiled_by_tokens:
                    evidence.dispatch_tokens = None
                evidence.runtime_fallbacks += 1
                if not getattr(module, "_stax_degraded", False):
                    module._stax_degraded = True
                    log_event(
                        _LOGGER,
                        logging.ERROR,
                        "engine.compile.fallback",
                        region=evidence.region,
                        tokens=tokens,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                return call_original(x, decode, lora_slots, layer_index)
            evidence.runtime_calls += 1
            return result

        module._eager_forward = original
        module._stax_compiled = True
        module._stax_degraded = False
        module._stax_buckets = tuple(sorted(compiled_by_tokens))
        module.forward = compiled_forward


def compile_model(
    model: Any,
    *,
    policy: CompilePolicy,
    dtype: str,
    device: Any,
    enable_lora: bool = False,
    compile_fn: CompileFn | None = None,
) -> CompileReport:
    """Run the cold-path compile pass.  Never raises for a rejected region."""

    provider = StaxCompileProvider(policy, compile_fn=compile_fn)
    return provider.apply(
        model,
        dtype=dtype,
        device=device,
        enable_lora=enable_lora,
    )
