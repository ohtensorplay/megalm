from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class BertEmbeddingError(ValueError):
    """Raised when a BERT embedding checkpoint violates its declared contract."""


class PoolingMode(str, Enum):
    CLS = "cls"
    MEAN = "mean"
    LAST = "last"


@dataclass(frozen=True, slots=True)
class BertEmbeddingConfig:
    """Immutable BERT encoder contract, including Sentence-Transformers pooling."""

    model_type: str
    architecture: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    layer_norm_eps: float
    hidden_act: str
    pad_token_id: int
    type_vocab_size: int
    pooling_mode: PoolingMode
    normalize: bool
    use_pooler_dense: bool
    embedding_size: int

    def __post_init__(self) -> None:
        if not self.model_type or not self.architecture:
            raise BertEmbeddingError("model_type and architecture are required")
        if min(
            self.vocab_size,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
            self.max_position_embeddings,
            self.embedding_size,
        ) <= 0:
            raise BertEmbeddingError("BERT dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise BertEmbeddingError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if self.layer_norm_eps <= 0:
            raise BertEmbeddingError("layer_norm_eps must be positive")
        if self.hidden_act != "gelu":
            raise BertEmbeddingError(
                f"unsupported BERT activation {self.hidden_act!r}; only gelu"
            )
        if self.pad_token_id < 0 or self.type_vocab_size <= 0:
            raise BertEmbeddingError("invalid token type vocabulary")
        if self.embedding_size > self.hidden_size:
            raise BertEmbeddingError("embedding_size exceeds hidden_size")

    @classmethod
    def from_checkpoint(cls, model_path: str | Path) -> BertEmbeddingConfig:
        root = Path(model_path)
        config_path = root / "config.json" if root.is_dir() else root
        try:
            payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BertEmbeddingError(
                f"cannot read BERT config from {config_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise BertEmbeddingError("BERT config must be an object")
        architectures = payload.get("architectures") or ()
        architecture = (
            str(architectures[0]) if architectures else "BertModel"
        )
        if "bert" not in architecture.lower():
            raise BertEmbeddingError(
                f"architecture {architecture!r} is not a BERT encoder"
            )
        pooling, normalize = resolve_pooling_config(root)
        hidden_size = int(payload["hidden_size"])
        return cls(
            model_type=str(payload.get("model_type", "bert")),
            architecture=architecture,
            vocab_size=int(payload["vocab_size"]),
            hidden_size=hidden_size,
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            intermediate_size=int(payload["intermediate_size"]),
            max_position_embeddings=int(payload["max_position_embeddings"]),
            layer_norm_eps=float(payload.get("layer_norm_eps", 1e-12)),
            hidden_act=str(payload.get("hidden_act", "gelu")),
            pad_token_id=int(payload.get("pad_token_id", 0)),
            type_vocab_size=int(payload.get("type_vocab_size", 2)),
            pooling_mode=pooling,
            normalize=normalize,
            use_pooler_dense=False,
            embedding_size=int(payload.get("embedding_size", hidden_size)),
        )


def resolve_pooling_config(root: Path) -> tuple[PoolingMode, bool]:
    """Resolve Sentence-Transformers pooling, falling back to BERT's CLS token.

    Precedence follows the pinned vLLM reference: explicit ST ``modules.json``
    metadata, then the pooling config it points at, then the architecture
    default (CLS + L2 normalization for BGE/BERT encoders). Compact ST
    ``pooling_mode`` strings are intentionally not accepted.
    """

    modules_path = root / "modules.json"
    pooling: PoolingMode = PoolingMode.CLS
    normalize = True
    if modules_path.is_file():
        try:
            modules = json.loads(modules_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BertEmbeddingError(
                f"cannot read Sentence-Transformers modules from {modules_path}"
            ) from exc
        if not isinstance(modules, list):
            raise BertEmbeddingError("modules.json must contain a list")
        for module in modules:
            if not isinstance(module, dict):
                continue
            module_type = str(module.get("type", ""))
            if module_type.endswith("models.Pooling"):
                path = root / str(module.get("path", ""))
                config = json.loads((path / "config.json").read_text(encoding="utf-8"))
                if not isinstance(config, dict):
                    raise BertEmbeddingError("pooling config must be an object")
                selected = {
                    "pooling_mode_cls_token": PoolingMode.CLS,
                    "pooling_mode_mean_tokens": PoolingMode.MEAN,
                    "pooling_mode_last_tokens": PoolingMode.LAST,
                }
                modes = [
                    mode
                    for key, mode in selected.items()
                    if bool(config.get(key, False))
                ]
                if len(modes) != 1:
                    raise BertEmbeddingError(
                        "pooling config must select exactly one of CLS/MEAN/LAST"
                    )
                pooling = modes[0]
            elif module_type.endswith("models.Normalize"):
                normalize = True
    return pooling, normalize
