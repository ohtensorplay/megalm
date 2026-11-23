from __future__ import annotations

import math
from pathlib import Path

import tensorplay
import tensorplay.nn.functional as F

from megalm.models.inkling2.model import rms_norm


class Inkling2DSparkPredictor:
    """Single-step DSpark draft for Inkling-2 (``mtp.safetensors``).

    The draft consumes the four target features captured by the target model
    (layer visits 5, 23, 41 plus the final-norm output) and proposes one token.
    The engine's native MTP path verifies that token against the target, so an
    imperfect draft only lowers the acceptance rate — it cannot change output.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        embed_weight: tensorplay.Tensor,
        embed_norm_weight: tensorplay.Tensor,
        lm_head_weight: tensorplay.Tensor,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        hidden_size: int = 2304,
        num_layers: int = 3,
        num_heads: int = 24,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 9216,
        eps: float = 1e-6,
        rope_theta: float = 1e6,
        softcap: float = 30.0,
        max_slots: int = 32,
    ) -> None:
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.hidden = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps
        self.softcap = softcap
        self.rank = 256
        self.embed_weight = embed_weight
        self.embed_norm_weight = embed_norm_weight
        self.lm_head_weight = lm_head_weight

        from safetensors import safe_open

        path = Path(model_path) / "mtp.safetensors"
        with safe_open(str(path), framework="pt") as source:
            raw = {key: source.get_tensor(key) for key in source.keys()}

        def tensor(name: str) -> tensorplay.Tensor:
            value = raw[name]
            return tensorplay.tensor(
                value, dtype=dtype, device=self.device
            )

        self.fuse_alpha = tensorplay.softmax(
            tensorplay.tensor(
                raw["fuse_weight"], dtype=tensorplay.float32, device=self.device
            ),
            dim=-1,
        )
        self.mask_embed = tensor("mask_embed")
        self.norm_weight = tensor("norm.weight")
        self.layers: list[dict[str, tensorplay.Tensor]] = []
        for index in range(num_layers):
            prefix = f"layers.{index}."
            attn = prefix + "self_attn."
            self.layers.append(
                {
                    "input_norm": tensor(prefix + "input_layernorm.weight"),
                    "post_norm": tensor(prefix + "post_attention_layernorm.weight"),
                    "dflare_norm": tensor(prefix + "dflare_norm.weight"),
                    "q": tensor(attn + "q_proj.weight"),
                    "k": tensor(attn + "k_proj.weight"),
                    "v": tensor(attn + "v_proj.weight"),
                    "k_ctx": tensor(attn + "k_proj_ctx.weight"),
                    "v_ctx": tensor(attn + "v_proj_ctx.weight"),
                    "o": tensor(attn + "o_proj.weight"),
                    "q_norm": tensor(attn + "q_norm.weight"),
                    "k_norm": tensor(attn + "k_norm.weight"),
                    "k_norm_ctx": tensor(attn + "k_norm_ctx.weight"),
                    "gate": tensor(prefix + "mlp.gate_proj.weight"),
                    "up": tensor(prefix + "mlp.up_proj.weight"),
                    "down": tensor(prefix + "mlp.down_proj.weight"),
                }
            )
        self.markov_dflare = tensor("markov_head.dflare_norm.weight")
        self.markov_w1 = tensor("markov_head.markov_w1.weight")
        self.rnn_proj = tensorplay.tensor(
            raw["markov_head.rnn_proj.weight"],
            dtype=tensorplay.float32,
            device=self.device,
        )
        self.rnn_bias = tensorplay.tensor(
            raw["markov_head.rnn_proj.bias"],
            dtype=tensorplay.float32,
            device=self.device,
        )
        hidden_out = tensorplay.tensor(
            raw["markov_head.hidden_out.weight"],
            dtype=tensorplay.float32,
            device=self.device,
        )
        hidden_bias = tensorplay.tensor(
            raw["markov_head.hidden_out.bias"],
            dtype=tensorplay.float32,
            device=self.device,
        )
        self.fold_weight = tensorplay.matmul(
            lm_head_weight.float(), hidden_out
        )
        self.fold_bias = tensorplay.matmul(
            lm_head_weight.float(), hidden_bias
        )
        inv_freq = 1.0 / (
            rope_theta
            ** (
                tensorplay.arange(
                    0, head_dim, 2, dtype=tensorplay.float32, device=self.device
                )
                / head_dim
            )
        )
        self.register_buffer("_inv_freq", inv_freq)
        self.state = tensorplay.zeros(
            (max_slots, self.rank), dtype=dtype, device=self.device
        )

    def register_buffer(self, name: str, value: tensorplay.Tensor) -> None:
        setattr(self, name, value)

    def _rope(self, x: tensorplay.Tensor, positions: tensorplay.Tensor) -> tensorplay.Tensor:
        """Neox RoPE on ``[tokens, heads, head_dim]``."""

        angles = (
            positions.float().reshape(-1, 1)
            * self._inv_freq.reshape(1, -1)
        )
        half = x.shape[-1] // 2
        cos = tensorplay.cos(angles).to(x.dtype).reshape(-1, 1, half)
        sin = tensorplay.sin(angles).to(x.dtype).reshape(-1, 1, half)
        first = x[..., :half]
        second = x[..., half:]
        return tensorplay.cat(
            (first * cos - second * sin, second * cos + first * sin),
            dim=-1,
        )

    def _mix(self, features: tensorplay.Tensor, row: int, weight: tensorplay.Tensor) -> tensorplay.Tensor:
        mixed = tensorplay.sum(
            features.float() * self.fuse_alpha[row].reshape(1, -1, 1),
            dim=1,
        )
        return rms_norm(
            mixed, weight, self.eps, output_dtype=self.dtype
        )

    def _head_norm(
        self,
        x: tensorplay.Tensor,
        weight: tensorplay.Tensor,
        num_heads: int,
    ) -> tensorplay.Tensor:
        shaped = x.reshape(x.shape[0], num_heads, self.head_dim)
        return rms_norm(
            shaped,
            weight.reshape(1, 1, self.head_dim),
            self.eps,
            output_dtype=self.dtype,
        ).reshape(x.shape[0], num_heads * self.head_dim)

    def _run_layers(
        self,
        x: tensorplay.Tensor,
        features: tensorplay.Tensor,
        positions: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        tokens = x.shape[0]
        residual: tensorplay.Tensor | None = None
        for index, weights in enumerate(self.layers):
            context = self._mix(features[:tokens], index, weights["dflare_norm"])
            ctx_k = tensorplay.matmul(
                context, weights["k_ctx"].transpose(0, 1)
            ).reshape(tokens, self.num_kv_heads, self.head_dim)
            ctx_k = self._head_norm(
                ctx_k.reshape(tokens, self.num_kv_heads * self.head_dim),
                weights["k_norm_ctx"],
                self.num_kv_heads,
            ).reshape(tokens, self.num_kv_heads, self.head_dim)
            ctx_k = self._rope(ctx_k, positions)
            ctx_v = tensorplay.matmul(
                context, weights["v_ctx"].transpose(0, 1)
            ).reshape(tokens, self.num_kv_heads, self.head_dim)

            if residual is None:
                residual = x
                hidden = rms_norm(
                    x, weights["input_norm"], self.eps, output_dtype=self.dtype
                )
            else:
                residual = residual + x
                hidden = rms_norm(
                    residual,
                    weights["input_norm"],
                    self.eps,
                    output_dtype=self.dtype,
                )
            q = self._head_norm(
                tensorplay.matmul(hidden, weights["q"].transpose(0, 1)),
                weights["q_norm"],
                self.num_heads,
            ).reshape(tokens, self.num_heads, self.head_dim)
            k = self._head_norm(
                tensorplay.matmul(hidden, weights["k"].transpose(0, 1)),
                weights["k_norm"],
                self.num_kv_heads,
            ).reshape(tokens, self.num_kv_heads, self.head_dim)
            v = tensorplay.matmul(
                hidden, weights["v"].transpose(0, 1)
            ).reshape(tokens, self.num_kv_heads, self.head_dim)
            q = self._rope(q, positions)
            k = self._rope(k, positions)
            all_k = tensorplay.cat((ctx_k, k), dim=0)
            all_v = tensorplay.cat((ctx_v, v), dim=0)
            repeat = self.num_heads // self.num_kv_heads
            all_k = all_k.repeat_interleave(repeat, dim=1)
            all_v = all_v.repeat_interleave(repeat, dim=1)
            logits = tensorplay.einsum(
                "thd,shd->ths", q, all_k
            ) * (self.head_dim**-0.5)
            probs = tensorplay.softmax(logits, dim=-1).to(self.dtype)
            attended = tensorplay.einsum("ths,shd->thd", probs, all_v).reshape(
                tokens, self.num_heads * self.head_dim
            )
            residual = residual + tensorplay.matmul(
                attended, weights["o"].transpose(0, 1)
            )
            hidden = rms_norm(
                residual, weights["post_norm"], self.eps, output_dtype=self.dtype
            )
            gate_up = tensorplay.cat(
                (
                    tensorplay.matmul(hidden, weights["gate"].transpose(0, 1)),
                    tensorplay.matmul(hidden, weights["up"].transpose(0, 1)),
                ),
                dim=-1,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate.float()) * up.float()
            x = tensorplay.matmul(
                activated.to(self.dtype), weights["down"].transpose(0, 1)
            )
        hidden = rms_norm(
            residual + x, self.norm_weight, self.eps, output_dtype=self.dtype
        )
        return hidden

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        batch,
        target_hidden_states: tensorplay.Tensor,
        cache,
        *,
        features: tensorplay.Tensor,
        slot: int,
    ) -> tensorplay.Tensor:
        del cache
        tokens = batch.input_ids.shape[0]
        if features.shape[0] < tokens or features.shape[1] != 4:
            raise ValueError("DSpark needs four target features per token")
        positions = batch.positions
        x = rms_norm(
            F.embedding(batch.input_ids, self.embed_weight),
            self.embed_norm_weight,
            self.eps,
            output_dtype=self.dtype,
        )
        hidden = self._run_layers(x, features, positions)
        base_logits = tensorplay.matmul(hidden, self.lm_head_weight.transpose(0, 1))
        f_rnn = self._mix(features[:tokens], self.num_layers, self.markov_dflare)
        context_input = tensorplay.cat(
            (hidden[tokens - 1 : tokens].float(), f_rnn[tokens - 1 : tokens].float()),
            dim=-1,
        )
        context_bias = (
            tensorplay.matmul(
                context_input, self.rnn_proj[:, 2 * self.rank :].transpose(0, 1)
            )
            + self.rnn_bias
        )
        last = batch.input_ids[tokens - 1 : tokens].long()
        prev = tensorplay.index_select(self.markov_w1, 0, last)
        state_input = tensorplay.cat(
            (self.state[slot : slot + 1].float(), prev.float()), dim=-1
        )
        raw = context_bias + tensorplay.matmul(
            state_input, self.rnn_proj[:, : 2 * self.rank].transpose(0, 1)
        )
        gate_raw, candidate_raw, out_raw = raw.chunk(3, dim=-1)
        gate = tensorplay.sigmoid(gate_raw)
        candidate = tensorplay.tanh(candidate_raw)
        updated = gate * self.state[slot : slot + 1].float() + (1.0 - gate) * candidate
        self.state[slot : slot + 1] = updated.to(self.dtype)
        latent = tensorplay.tanh(out_raw)
        bias = self.fold_bias + tensorplay.matmul(
            latent, self.fold_weight.transpose(0, 1)
        )
        logits = base_logits.float() + bias
        logits = self.softcap * tensorplay.tanh(logits / self.softcap)
        return logits

    def draft_block(
        self,
        anchor_id: int,
        features: tensorplay.Tensor,
        anchor_position: int,
        slot: int,
        count: int = 7,
    ) -> tuple[int, ...]:
        """Propose ``count`` tokens from one anchor-plus-masks draft block."""

        block = count + 1
        anchor = rms_norm(
            F.embedding(
                tensorplay.tensor(
                    (anchor_id,), dtype=tensorplay.int64, device=self.device
                ),
                self.embed_weight,
            ),
            self.embed_norm_weight,
            self.eps,
            output_dtype=self.dtype,
        )
        masks = self.mask_embed.reshape(1, -1).expand(count, -1)
        block_input = tensorplay.cat((anchor, masks), dim=0)
        block_features = features.reshape(1, 4, self.hidden).expand(
            block, -1, -1
        )
        positions = tensorplay.arange(
            anchor_position,
            anchor_position + block,
            dtype=tensorplay.int64,
            device=self.device,
        )
        hidden = self._run_layers(block_input, block_features, positions)
        base_logits = tensorplay.matmul(
            hidden, self.lm_head_weight.transpose(0, 1)
        ).float()
        anchor_features = features.reshape(1, 4, self.hidden)
        f_rnn = self._mix(anchor_features, self.num_layers, self.markov_dflare)
        context_bias = tensorplay.matmul(
            tensorplay.cat(
                (hidden[1:block].float(), f_rnn.expand(count, -1).float()),
                dim=-1,
            ),
            self.rnn_proj[:, 2 * self.rank :].transpose(0, 1),
        ) + self.rnn_bias
        state = self.state[slot : slot + 1].float()
        previous = anchor_id
        drafts: list[int] = []
        for step in range(count):
            prev = tensorplay.index_select(
                self.markov_w1,
                0,
                tensorplay.tensor(
                    (previous,), dtype=tensorplay.int64, device=self.device
                ),
            )
            state_input = tensorplay.cat((state, prev.float()), dim=-1)
            raw = context_bias[step] + tensorplay.matmul(
                state_input, self.rnn_proj[:, : 2 * self.rank].transpose(0, 1)
            )
            gate_raw, candidate_raw, out_raw = raw.chunk(3, dim=-1)
            gate = tensorplay.sigmoid(gate_raw)
            candidate = tensorplay.tanh(candidate_raw)
            state = gate * state + (1.0 - gate) * candidate
            latent = tensorplay.tanh(out_raw)
            bias = self.fold_bias + tensorplay.matmul(
                latent, self.fold_weight.transpose(0, 1)
            )
            logits = base_logits[step + 1] + bias
            logits = self.softcap * tensorplay.tanh(logits / self.softcap)
            previous = int(tensorplay.argmax(logits, dim=-1).item())
            drafts.append(previous)
        self.state[slot : slot + 1] = state.to(self.dtype)
        return tuple(drafts)

    def compute_logits(
        self,
        hidden_states: tensorplay.Tensor,
        lm_head_weight: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        del lm_head_weight
        return hidden_states
