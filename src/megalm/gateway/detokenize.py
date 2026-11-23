from __future__ import annotations

from megalm.runtime.tokenizer import FrontendError, ModelTokenizer


class IncrementalDecoder:
    """Frontend-owned detokenizer state for one streamed generation."""

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        *,
        skip_special_tokens: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._skip_special_tokens = skip_special_tokens
        self._decode_stream = tokenizer.decode_stream(
            skip_special_tokens=skip_special_tokens
        )
        self._token_ids: list[int] = []
        self._emitted = ""

    @property
    def token_count(self) -> int:
        return len(self._token_ids)

    @property
    def text(self) -> str:
        return self._emitted

    def push(self, token_ids: tuple[int, ...]) -> str:
        self._token_ids.extend(token_ids)
        delta = self._decode_stream.push(token_ids)
        self._emitted += delta
        return delta

    def finish(self) -> str:
        """Flush an incomplete byte sequence and verify the native stream."""

        decoded = self._tokenizer.decode(
            self._token_ids,
            skip_special_tokens=self._skip_special_tokens,
        )
        if not decoded.startswith(self._emitted):
            raise FrontendError(
                "incremental tokenizer output diverged from full decode"
            )
        delta = decoded[len(self._emitted) :]
        self._emitted = decoded
        return delta


__all__ = ["IncrementalDecoder"]
