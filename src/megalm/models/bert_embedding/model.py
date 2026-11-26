from __future__ import annotations

import re
from pathlib import Path

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.loading import SafetensorsSource, WeightLoadReport
from megalm.models.bert_embedding.config import BertEmbeddingConfig, PoolingMode


class BertEmbeddingModelError(RuntimeError):
    """Raised when BERT weights cannot be loaded into the native model."""


class _BertLayer(nn.Module):
    def __init__(self, config: BertEmbeddingConfig, *, device, dtype) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True, device=device, dtype=dtype)
        self.attention_dense = nn.Linear(hidden, hidden, bias=True, device=device, dtype=dtype)
        self.attention_norm = nn.LayerNorm(hidden, eps=config.layer_norm_eps, device=device, dtype=dtype)
        self.intermediate = nn.Linear(hidden, config.intermediate_size, bias=True, device=device, dtype=dtype)
        self.output_dense = nn.Linear(config.intermediate_size, hidden, bias=True, device=device, dtype=dtype)
        self.output_norm = nn.LayerNorm(hidden, eps=config.layer_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        hidden: tensorplay.Tensor,
        attention_mask: tensorplay.Tensor | None,
    ) -> tensorplay.Tensor:
        batch, sequence, width = hidden.shape
        head_dim = width // self.num_attention_heads
        qkv = self.qkv(hidden).reshape(batch, sequence, 3, self.num_attention_heads, head_dim)
        query, key, value = qkv.unbind(2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scale = head_dim**-0.5
        scores = tensorplay.matmul(query, key.transpose(-1, -2)) * scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
        probabilities = F.softmax(scores, dim=-1)
        context = tensorplay.matmul(probabilities, value)
        context = context.permute(0, 2, 1, 3).reshape(batch, sequence, width)
        hidden = self.attention_norm(hidden + self.attention_dense(context))
        feed_forward = F.gelu(self.intermediate(hidden))
        return self.output_norm(hidden + self.output_dense(feed_forward))


class BertEmbeddingModel(nn.Module):
    """Native BERT encoder with deterministic CLS/MEAN/LAST sequence pooling.

    No sampling, no KV arena, no generation graph. All inputs of one request
    batch run through a single padded forward, and pooled vectors are returned
    in input order. Weights stream from safetensors one tensor at a time.
    """

    def __init__(
        self,
        config: BertEmbeddingConfig,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=dtype
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, device=device, dtype=dtype
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size, device=device, dtype=dtype
        )
        self.embeddings_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            _BertLayer(config, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        )
        self.pad_token_id = config.pad_token_id

    def forward(
        self,
        input_ids: tensorplay.Tensor,
        attention_mask: tensorplay.Tensor | None = None,
    ) -> tensorplay.Tensor:
        if input_ids.ndim != 2:
            raise BertEmbeddingModelError("input_ids must have shape [batch, sequence]")
        batch, sequence = input_ids.shape
        if sequence > self.config.max_position_embeddings:
            raise BertEmbeddingModelError(
                f"sequence {sequence} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
        if attention_mask is None:
            attention_mask = tensorplay.ones((batch, sequence), device=input_ids.device, dtype=tensorplay.long)
        positions = tensorplay.arange(sequence, device=input_ids.device)
        token_types = tensorplay.zeros((batch, sequence), device=input_ids.device, dtype=tensorplay.long)
        hidden = self.embeddings_norm(
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(token_types)
        )
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return self._pool(hidden, attention_mask)

    def _pool(
        self,
        hidden: tensorplay.Tensor,
        attention_mask: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        mode = self.config.pooling_mode
        if mode is PoolingMode.CLS:
            pooled = hidden[:, 0, :]
        elif mode is PoolingMode.LAST:
            lengths = attention_mask.to(tensorplay.long).sum(dim=1) - 1
            pooled = hidden[tensorplay.arange(hidden.shape[0], device=hidden.device), lengths]
        elif mode is PoolingMode.MEAN:
            mask = attention_mask.to(hidden.dtype)[:, :, None]
            summed = (hidden * mask).sum(dim=1)
            pooled = summed / mask.sum(dim=1).clamp(min=1)
        else:
            raise BertEmbeddingModelError(f"unsupported pooling mode {mode!r}")
        if self.config.normalize:
            pooled = F.normalize(pooled.float(), p=2, dim=-1).to(self.dtype)
        if self.config.embedding_size < self.config.hidden_size:
            pooled = pooled[..., : self.config.embedding_size]
        return pooled.contiguous()

    @tensorplay.no_grad()
    def load_weights(self, model_path: str | Path) -> WeightLoadReport:
        parameters = dict(self.named_parameters())
        loaded: set[str] = set()
        pending: dict[str, dict[str, tensorplay.Tensor]] = {}

        def copy(target_name: str, tensor: tensorplay.Tensor) -> None:
            try:
                target = parameters[target_name]
            except KeyError as exc:
                raise BertEmbeddingModelError(
                    f"checkpoint tensor has no native target: {target_name}"
                ) from exc
            if tuple(target.shape) != tuple(tensor.shape):
                raise BertEmbeddingModelError(
                    f"shape mismatch for {target_name}: expected "
                    f"{tuple(target.shape)}, got {tuple(tensor.shape)}"
                )
            target.copy_(tensor.to(device=self.device, dtype=self.dtype))
            loaded.add(target_name)

        def pack(source_name: str, field: str, tensor: tensorplay.Tensor) -> None:
            match = re.fullmatch(
                r"encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)\.(weight|bias)",
                source_name,
            )
            if match is None:
                raise BertEmbeddingModelError(f"unexpected qkv tensor {source_name!r}")
            layer, projection, kind = match.groups()
            bucket = pending.setdefault(f"{layer}:{kind}", {})
            bucket[projection] = tensor
            if set(bucket) == {"query", "key", "value"}:
                packed = tensorplay.cat(
                    (bucket["query"], bucket["key"], bucket["value"]), dim=0
                )
                copy(f"layers.{int(layer)}.qkv.{kind}", packed)
                del pending[f"{layer}:{kind}"]

        def consume(source_name: str, tensor: tensorplay.Tensor) -> None:
            # HF stores `position_ids` as a non-trainable bookkeeping buffer.
            if source_name == "embeddings.position_ids" or (
                source_name.startswith("pooler.")
                and not self.config.use_pooler_dense
            ):
                return
            if source_name == "embeddings.word_embeddings.weight":
                copy("word_embeddings.weight", tensor)
            elif source_name == "embeddings.position_embeddings.weight":
                copy("position_embeddings.weight", tensor)
            elif source_name == "embeddings.token_type_embeddings.weight":
                copy("token_type_embeddings.weight", tensor)
            elif source_name == "embeddings.LayerNorm.weight":
                copy("embeddings_norm.weight", tensor)
            elif source_name == "embeddings.LayerNorm.bias":
                copy("embeddings_norm.bias", tensor)
            elif ".attention.self." in source_name:
                pack(source_name, "qkv", tensor)
            else:
                match = re.fullmatch(
                    r"encoder\.layer\.(\d+)\.(?P<rest>.+)", source_name
                )
                if match is None:
                    raise BertEmbeddingModelError(
                        f"checkpoint tensor has no native target: {source_name}"
                    )
                layer, rest = int(match.group(1)), match.group("rest")
                mapping = {
                    "attention.output.dense.weight": "attention_dense.weight",
                    "attention.output.dense.bias": "attention_dense.bias",
                    "attention.output.LayerNorm.weight": "attention_norm.weight",
                    "attention.output.LayerNorm.bias": "attention_norm.bias",
                    "intermediate.dense.weight": "intermediate.weight",
                    "intermediate.dense.bias": "intermediate.bias",
                    "output.dense.weight": "output_dense.weight",
                    "output.dense.bias": "output_dense.bias",
                    "output.LayerNorm.weight": "output_norm.weight",
                    "output.LayerNorm.bias": "output_norm.bias",
                }
                target = mapping.get(rest)
                if target is None:
                    raise BertEmbeddingModelError(
                        f"checkpoint tensor has no native target: {source_name}"
                    )
                copy(f"layers.{layer}.{target}", tensor)

        source = SafetensorsSource(model_path)
        report = source.stream(consume, device="cpu")
        if pending:
            raise BertEmbeddingModelError(
                f"incomplete fused QKV weights in checkpoint: {sorted(pending)}"
            )
        expected = len(parameters)
        if len(loaded) != expected:
            missing = sorted(set(parameters) - loaded)
            raise BertEmbeddingModelError(
                f"checkpoint loaded {len(loaded)}/{expected} native tensors; missing {missing}"
            )
        return report


def load_bert_embedding_model(
    model_path: str | Path,
    *,
    device: tensorplay.device | str,
    dtype: tensorplay.dtype,
) -> tuple[BertEmbeddingModel, BertEmbeddingConfig, WeightLoadReport]:
    config = BertEmbeddingConfig.from_checkpoint(model_path)
    model = BertEmbeddingModel(config, device=device, dtype=dtype)
    report = model.load_weights(model_path)
    model.eval()
    return model, config, report
