from __future__ import annotations

from pathlib import Path

import tensorplay
from tensorplay import nn

from megalm.kernels.bindings import KernelBindings, bind_kernel_plan
from megalm.kernels.ops import KernelPolicy
from megalm.loading import SafetensorsSource, WeightLoadReport
from megalm.models.oxr.config import OXRConfig
from megalm.models.oxr.vision import OXRVisualTokenizer
from megalm.models.qwen3 import Qwen3ForCausalLM
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache

_VISION_PREFIX = "visual_tokenizer.vit.vision_model."
_HEAD_PREFIX = "visual_tokenizer.head."
_VITE = "vte.weight"


class OXRModelError(RuntimeError):
    """Raised when an OXR checkpoint violates the native contract."""


def _vision_target(source_name: str) -> str:
    if source_name.startswith(_VISION_PREFIX):
        target = "visual." + source_name[len(_VISION_PREFIX) :]
        if target.startswith("visual.encoder.layers."):
            target = target.replace(
                "visual.encoder.layers.", "visual.layers.", 1
            )
        return target
    if source_name.startswith(_HEAD_PREFIX):
        suffix = source_name[len(_HEAD_PREFIX) :]
        if suffix == "0.weight":
            return "visual.head.weight"
        if suffix == "1.weight":
            return "visual.head_norm.weight"
        if suffix == "1.bias":
            return "visual.head_norm.bias"
        raise OXRModelError(f"unexpected OXR tokenizer head tensor: {source_name}")
    if source_name == _VITE:
        return "visual.visual_embedding.weight"
    raise OXRModelError(f"no native target for OXR tensor: {source_name}")


class OXRForConditionalGeneration(nn.Module):
    """Native OXR-1.0: SigLIP2-NAViT visual tokenizer + Qwen3 text decoder."""

    uses_mrope = False

    def __init__(
        self,
        config: OXRConfig,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.kernel_policy = kernel_policy
        if kernel_bindings is None:
            kernel_bindings = bind_kernel_plan(
                kernel_policy,
                device=self.device,
                dtype=dtype,
                model_type="qwen3",
            )
        self.kernel_bindings = kernel_bindings
        self.llm = Qwen3ForCausalLM(
            config.llm_config,
            device=self.device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
            use_decode_gemv=True,
        )
        self.visual = OXRVisualTokenizer(
            config.vision_config,
            visual_vocab_size=config.visual_vocab_size,
            hidden_size=config.hidden_size,
            device=self.device,
            dtype=dtype,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: tensorplay.device | str = "cuda",
        dtype: tensorplay.dtype = tensorplay.bfloat16,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
    ) -> tuple[OXRForConditionalGeneration, WeightLoadReport]:
        config = OXRConfig.from_path(model_path)
        model = cls(
            config,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
        report = model.load_weights(model_path)
        model.eval()
        return model, report

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def lm_head(self) -> tensorplay.Tensor:
        return self.llm.lm_head()

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        return self.llm(batch, cache)

    def configure_moe_workspace(self, max_batched_tokens: int) -> int:
        return self.llm.configure_moe_workspace(max_batched_tokens)

    def warmup_prefill_kernels(
        self,
        *,
        max_batched_tokens: int,
    ) -> tuple[int, ...]:
        return self.llm.warmup_prefill_kernels(
            max_batched_tokens=max_batched_tokens
        )

    def encode_images(
        self,
        pixel_values: tensorplay.Tensor,
        grid_thw: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        return self.visual(pixel_values, grid_thw)

    def input_embeddings(self, input_ids: tensorplay.Tensor) -> tensorplay.Tensor:
        """Embed text tokens and the OXR visual indicator sentinels.

        The visual-atom sentinel rows are overwritten by the engine with the
        cached tower output; indicator rows come from the visual embedding
        table exactly as in the reference ``merge_multimodal``.
        """

        embeddings = self.llm.model.embed_tokens(input_ids)
        indicators = self.visual.visual_embedding.weight[
            self.config.visual_vocab_size - self.config.indicator_count :
        ]
        starts = input_ids == self.config.image_start_token_id
        if bool(starts.any().item()):
            embeddings[starts] = indicators[0].to(embeddings.dtype)
        ends = input_ids == self.config.image_end_token_id
        if bool(ends.any().item()):
            embeddings[ends] = indicators[1].to(embeddings.dtype)
        return embeddings

    def load_weights(self, model_path: str | Path) -> WeightLoadReport:
        parameters = dict(self.named_parameters())
        loaded: set[str] = set()

        llm_source = SafetensorsSource(model_path, include_prefixes=("llm.",))
        llm_report = self.llm.load_weights(
            model_path,
            source=llm_source,
            key_prefix="llm.",
            strict=False,
        )
        # The Qwen3 loader packs q/k/v and gate/up projections, so its report
        # lists source names rather than all native targets; the dense LLM is
        # fully covered by the llm. prefix stream.
        loaded.update(name for name in parameters if name.startswith("llm."))

        @tensorplay.no_grad()
        def consume(source_name: str, tensor: tensorplay.Tensor) -> None:
            target_name = _vision_target(source_name)
            target = parameters.get(target_name)
            if target is None:
                raise OXRModelError(
                    f"checkpoint tensor has no native target: {source_name}"
                )
            expected = tuple(target.shape)
            actual = tuple(tensor.shape)
            if expected != actual:
                raise OXRModelError(
                    f"shape mismatch for {target_name}: expected {expected}, got {actual}"
                )
            target.copy_(tensor.to(device=self.device, dtype=self.dtype))
            loaded.add(target_name)

        vision_source = SafetensorsSource(
            model_path,
            include_prefixes=(_VISION_PREFIX[:-1], _HEAD_PREFIX[:-1], _VITE),
        )
        vision_report = vision_source.stream(consume)

        missing = sorted(set(parameters) - loaded)
        if missing:
            raise OXRModelError(
                f"checkpoint did not initialize parameters: {missing}"
            )
        return WeightLoadReport(
            tensor_count=llm_report.tensor_count + vision_report.tensor_count,
            total_bytes=llm_report.total_bytes + vision_report.total_bytes,
            source_names=llm_report.source_names + vision_report.source_names,
            target_names=llm_report.target_names + vision_report.target_names,
        )


__all__ = ["OXRForConditionalGeneration", "OXRModelError"]
