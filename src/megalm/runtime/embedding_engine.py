from __future__ import annotations

import logging
from dataclasses import dataclass

import tensorplay

from megalm.models.bert_embedding import BertEmbeddingConfig, load_bert_embedding_model

_LOGGER = logging.getLogger("megalm.embedding")


class EmbeddingEngineError(ValueError):
    """Raised when an embedding request cannot be executed by this engine."""


@dataclass(frozen=True, slots=True)
class EmbeddingEngineConfig:
    model_path: str
    device: str = "cuda"
    dtype: str = "float16"
    max_batch_size: int = 64
    max_sequence_length: int | None = None

    def __post_init__(self) -> None:
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise EmbeddingEngineError(f"unsupported embedding dtype {self.dtype!r}")
        if self.max_batch_size <= 0:
            raise EmbeddingEngineError("embedding max_batch_size must be positive")
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise EmbeddingEngineError("embedding max_sequence_length must be positive")


class EmbeddingEngine:
    """Non-generative embedding executor.

    One padded forward per batch, CLS/MEAN/LAST pooling and L2 normalization in
    the native model. There is no KV arena, no sampler, no CUDA decode graph and
    no token stream: inputs and outputs are dense vectors only.
    """

    def __init__(self, config: EmbeddingEngineConfig) -> None:
        self.config = config
        dtype = {
            "float16": tensorplay.float16,
            "bfloat16": tensorplay.bfloat16,
            "float32": tensorplay.float32,
        }[config.dtype]
        self.model, self.model_config, self.weight_report = load_bert_embedding_model(
            config.model_path, device=config.device, dtype=dtype
        )
        self.max_model_len = min(
            self.model_config.max_position_embeddings,
            config.max_sequence_length or self.model_config.max_position_embeddings,
        )
        _LOGGER.info(
            "embedding_engine_ready model=%s pooling=%s normalize=%s weights=%d max_len=%d",
            self.model_config.architecture,
            self.model_config.pooling_mode.value,
            self.model_config.normalize,
            self.weight_report.tensor_count,
            self.max_model_len,
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "task": "embed",
            "model": self.model_config.architecture,
            "pooling": self.model_config.pooling_mode.value,
            "normalize": self.model_config.normalize,
            "embedding_size": self.model_config.embedding_size,
            "max_model_len": self.max_model_len,
            "max_batch_size": self.config.max_batch_size,
            "weight_tensors": self.weight_report.tensor_count,
            "weight_bytes": self.weight_report.total_bytes,
        }

    def encode(
        self,
        token_ids: list[list[int]],
        *,
        dimensions: int | None = None,
        normalize: bool | None = None,
        truncate_prompt_tokens: int | None = None,
        truncation_side: str = "right",
    ) -> list[list[float]]:
        if not token_ids:
            raise EmbeddingEngineError("input must contain at least one text")
        if dimensions is not None and not 0 < dimensions <= self.model_config.embedding_size:
            raise EmbeddingEngineError(
                f"dimensions must be in [1, {self.model_config.embedding_size}]"
            )
        if truncation_side not in {"left", "right"}:
            raise EmbeddingEngineError("truncation_side must be left or right")
        prepared: list[list[int]] = []
        for ids in token_ids:
            if not ids:
                raise EmbeddingEngineError("each input must contain at least one token")
            if any(token < 0 for token in ids):
                raise EmbeddingEngineError("token ids must be non-negative")
            limit = truncate_prompt_tokens or self.max_model_len
            if limit > self.max_model_len:
                raise EmbeddingEngineError(
                    f"truncate_prompt_tokens {limit} exceeds max_model_len "
                    f"{self.max_model_len}"
                )
            if len(ids) > limit:
                ids = ids[:limit] if truncation_side == "right" else ids[-limit:]
            prepared.append(ids)
        output: list[list[float]] = []
        pad = self.model_config.pad_token_id
        chunk_size = self.config.max_batch_size
        for start in range(0, len(prepared), chunk_size):
            chunk = prepared[start : start + chunk_size]
            width = max(len(ids) for ids in chunk)
            padded = [ids + [pad] * (width - len(ids)) for ids in chunk]
            ids_tensor = tensorplay.tensor(
                padded, dtype=tensorplay.long, device=self.config.device
            )
            mask_tensor = tensorplay.tensor(
                [[1] * len(ids) + [0] * (width - len(ids)) for ids in chunk],
                dtype=tensorplay.long,
                device=self.config.device,
            )
            with tensorplay.no_grad():
                vectors = self.model(ids_tensor, mask_tensor)
            output.extend([[float(value) for value in row] for row in vectors.tolist()])
        if dimensions is not None:
            output = [row[:dimensions] for row in output]
        if normalize is not None and normalize != self.model_config.normalize:
            output = self._normalize(output, normalize)
        return output

    @staticmethod
    def _normalize(rows: list[list[float]], enabled: bool) -> list[list[float]]:
        if not enabled:
            return rows
        result: list[list[float]] = []
        for row in rows:
            norm = sum(value * value for value in row) ** 0.5
            result.append([value / norm for value in row] if norm else list(row))
        return result
