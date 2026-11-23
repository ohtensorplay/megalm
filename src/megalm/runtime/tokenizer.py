from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer, decoders


class FrontendError(ValueError):
    """Raised when local tokenizer/template inputs are invalid."""


# Inkling typed content-block markers. They are ordinary special tokens, so the
# gateway can only use them for output parsing when the decoder preserves them.
INKLING_MESSAGE_MODEL = "<|message_model|>"
INKLING_CONTENT_TEXT = "<|content_text|>"
INKLING_CONTENT_THINKING = "<|content_thinking|>"
INKLING_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
INKLING_END_MESSAGE = "<|end_message|>"
INKLING_MODEL_END_SAMPLING = "<|content_model_end_sampling|>"

# OXR visual-indicator sentinels. These ids are unused by the OXR tokenizer;
# the native model replaces their embeddings with vte indicator rows.
_OXR_IMAGE_START_ID = 151931
_OXR_IMAGE_END_ID = 151932


@dataclass(frozen=True, slots=True)
class MultimodalEncoding:
    input_ids: tuple[int, ...]
    token_type_ids: tuple[int, ...]
    image_count: int
    video_count: int = 0
    audio_count: int = 0


class ModelDecodeStream:
    """Thin stateful wrapper around the tokenizer's native stream decoder."""

    __slots__ = ("_stream", "_tokenizer")

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        skip_special_tokens: bool,
    ) -> None:
        self._tokenizer = tokenizer
        self._stream = decoders.DecodeStream(
            skip_special_tokens=skip_special_tokens
        )

    def push(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        if len(token_ids) == 1:
            value: int | list[int] = int(token_ids[0])
        else:
            value = [int(token) for token in token_ids]
        return self._stream.step(self._tokenizer, value) or ""


class ModelTokenizer:
    """Tokenizer + trusted local template frontend with no Transformers dependency."""

    @staticmethod
    def _strip_generation_tags(source: str) -> str:
        return source.replace("{% generation %}", "").replace(
            "{% endgeneration %}", ""
        ).replace("{%- generation -%}", "").replace(
            "{%- endgeneration -%}", ""
        )

    def __init__(self, model_path: str | Path) -> None:
        root = Path(model_path)
        try:
            self._tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            tokenizer_config = json.loads(
                (root / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            template_path = root / "chat_template.jinja"
            if template_path.is_file():
                template_source = template_path.read_text(encoding="utf-8")
            else:
                template_source = tokenizer_config.get("chat_template")
                if not isinstance(template_source, str):
                    template_payload = json.loads(
                        (root / "chat_template.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    template_source = template_payload.get("chat_template")
                if not isinstance(template_source, str):
                    raise FrontendError(
                        f"model has no supported chat template under {root}"
                    )
        except (OSError, json.JSONDecodeError) as exc:
            raise FrontendError(f"cannot load tokenizer assets from {root}") from exc
        self._template_source = template_source
        self._model_name = root.name.lower()
        self._model_type = ""
        config_path = root / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                config = {}
            if isinstance(config, dict):
                self._model_type = str(config.get("model_type", "")).lower()
        environment = Environment(
            autoescape=False,
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        environment.globals["raise_exception"] = _raise_template_error
        # vLLM-style chat templates call ``tojson`` with sort_keys/separators,
        # which Jinja2's builtin filter does not accept. Override it so tool
        # declarations and structured messages render with stable key order.
        environment.filters["tojson"] = _template_tojson
        # Inkling-2 templates use HF's ``generation`` block.  Jinja does not
        # know it, so strip the block markers; the inner content renders.
        template_source = self._strip_generation_tags(template_source)
        self._template = environment.from_string(template_source)
        self.bos_token_id = self._optional_special_id(
            tokenizer_config,
            "bos_token",
        )
        self.eos_token_id = self._special_id(tokenizer_config, "eos_token")
        self.pad_token_id = self._special_id(tokenizer_config, "pad_token")
        self.image_token_id = self._first_token_id(
            "<|image_pad|>",
            "<|IMAGE|>",
        )
        self.video_token_id = self._first_token_id(
            "<|video_pad|>",
            "<|VIDEO|>",
        )
        self.audio_token_id = self._first_token_id("<|AUDIO|>")
        self.vision_start_token_id = self._first_token_id(
            "<|vision_start|>",
            "<|vision_bos|>",
        )
        self.vision_end_token_id = self._first_token_id(
            "<|vision_end|>",
            "<|vision_eos|>",
        )
        self.audio_start_token_id = self._first_token_id("<|audio_bos|>")
        self.audio_end_token_id = self._first_token_id("<|audio_eos|>")

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        """Return the id of `token`, or ``None`` when it is not in the vocab."""

        return self._tokenizer.token_to_id(token)

    def special_token_kinds(self) -> dict[int, str]:
        """Map every special-token id to the coarse kind used by the parser.

        ``message`` / ``text`` / ``thinking`` are Inkling block markers; every
        other special token is reported as ``other`` so the Rust parser can
        stop its backward prompt scan at the first unrelated marker.
        """

        kinds: dict[int, str] = {}
        for token_id, added in self._tokenizer.get_added_tokens_decoder().items():
            if not added.special:
                continue
            content = added.content
            if content == INKLING_MESSAGE_MODEL:
                kinds[token_id] = "message"
            elif content == INKLING_CONTENT_TEXT:
                kinds[token_id] = "text"
            elif content == INKLING_CONTENT_THINKING:
                kinds[token_id] = "thinking"
            else:
                kinds[token_id] = "other"
        return kinds

    def output_grammar(self) -> str:
        """Return the assistant-output grammar the tokenizer can express.

        ``inkling`` models delimit reasoning, text and tool calls with special
        tokens, so the decoder must keep them and the Rust block parser owns the
        grammar. Everything else uses the plain-text ``<think>`` splitter.
        """

        required = (
            INKLING_MESSAGE_MODEL,
            INKLING_CONTENT_TEXT,
            INKLING_CONTENT_THINKING,
            INKLING_INVOKE_TOOL_JSON,
            INKLING_END_MESSAGE,
            INKLING_MODEL_END_SAMPLING,
        )
        if all(
            self._tokenizer.token_to_id(marker) is not None for marker in required
        ):
            return "inkling"
        return "text"

    def tool_call_parser(self) -> tuple[str, bool]:
        """Return the Rust tool-call parser name and its special-token need.

        Detection uses the chat template first, then the model directory name
        and ``config.json`` model type. ``MEGALM_TOOL_PARSER`` overrides both.
        """

        override = os.environ.get("MEGALM_TOOL_PARSER")
        if override:
            name = override.strip().lower()
            if name in ("", "none", "off", "generic", "generic_json", "json"):
                return "generic_json", False
            known = {
                "deepseek_v32",
                "deepseek_v4",
                "hermes",
                "granite4",
                "qwen3_xml",
                "qwen3_coder",
                "seed_oss",
                "glm45",
                "glm47",
                "hy_v3",
                "hy4",
                "minimax_m2",
                "kimi_k2",
                "nemotron_v3",
                "mistral",
                "phi4mini",
                "internlm2",
                "llama3_json",
                "llama4_json",
            }
            if name not in known:
                raise FrontendError(
                    f"unknown MEGALM_TOOL_PARSER `{name}`; "
                    f"expected one of {sorted(known)}"
                )
            preserve = name in {
                "deepseek_v32",
                "deepseek_v4",
                "internlm2",
                "kimi_k2",
            }
            return name, preserve

        template = self._template_source
        name = self._model_name
        model_type = self._model_type
        dsml = "\uff5cDSML\uff5c"
        if dsml in template and "function_calls" in template:
            return "deepseek_v32", True
        if dsml in template and "tool_calls" in template:
            return "deepseek_v4", True

        if "[TOOL_CALLS]" in template:
            return "mistral", False
        if "functools[" in template:
            return "phi4mini", False
        if "<|action_start|><|plugin|>" in template or "internlm2" in name:
            return "internlm2", True
        if "<seed:tool_call>" in template:
            return "seed_oss", False
        if "<minimax:tool_call>" in template:
            return "minimax_m2", False
        if "<|tool_calls_section_begin|>" in template or "kimi-k2" in name:
            return "kimi_k2", True
        if "<tool_sep>" in template or "hy3" in name or "hy4" in name:
            return "hy_v3", False
        if "<function=" in template and "<parameter=" in template:
            return "qwen3_coder", False
        if "<arg_key>" in template:
            if "glm-4.5" in name or "glm-4.6" in name:
                return "glm45", False
            return "glm47", False

        if "qwen3-coder" in name or "qwen2.5-coder" in name or "qwen3.5" in name:
            return "qwen3_coder", False
        if "nemotron" in name:
            return "nemotron_v3", False
        if "granite-4" in name or "granite4" in name:
            return "granite4", False
        if "qwen" in name or "qwen" in model_type:
            return "qwen3_xml", False
        if "mistral" in name or "mixtral" in name:
            return "mistral", False
        if "llama-4" in name:
            return "llama4_json", False
        if "llama-3" in name:
            return "llama3_json", False
        if "glm-5" in name or "glm-4.7" in name:
            return "glm47", False
        if "glm-4.5" in name or "glm-4.6" in name:
            return "glm45", False
        if "phi-4-mini" in name or "phi4mini" in name:
            return "phi4mini", False

        if "<tool_call>" in template:
            lowered = template.lower()
            if "qwen" in model_type or "qwen" in lowered:
                return "qwen3_xml", False
            return "hermes", False
        return "generic_json", False

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str) or not text:
            raise FrontendError("text must be a non-empty string")
        return tuple(self._tokenizer.encode(text).ids)

    def encode_batch(self, texts: Sequence[str]) -> tuple[tuple[int, ...], ...]:
        normalized = tuple(texts)
        if not normalized or any(
            not isinstance(text, str) or not text for text in normalized
        ):
            raise FrontendError(
                "tokenizer batch must contain non-empty strings"
            )
        return tuple(
            tuple(encoding.ids)
            for encoding in self._tokenizer.encode_batch(list(normalized))
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        return self._tokenizer.decode(
            [int(token) for token in token_ids],
            skip_special_tokens=skip_special_tokens,
        )

    def decode_stream(
        self,
        *,
        skip_special_tokens: bool = False,
    ) -> ModelDecodeStream:
        return ModelDecodeStream(
            self._tokenizer,
            skip_special_tokens=skip_special_tokens,
        )

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        open_thinking: bool = False,
    ) -> tuple[int, ...]:
        return self.encode(
            self.render_chat(
                messages,
                tools=tools,
                open_thinking=open_thinking,
            )
        )

    def render_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        open_thinking: bool = False,
    ) -> str:
        normalized = self._normalize_messages(messages)
        return self._template.render(
            messages=normalized,
            tools=list(tools) if tools else None,
            add_generation_prompt=True,
            open_thinking=open_thinking,
            enable_thinking=open_thinking,
            add_vision_id=False,
            add_audio_id=False,
        )

    def encode_multimodal_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        image_feature_counts: Sequence[int],
        video_frame_feature_counts: Sequence[Sequence[int]] = (),
        video_timestamps: Sequence[Sequence[float]] = (),
        audio_feature_counts: Sequence[int] = (),
        tools: Sequence[Mapping[str, Any]] | None = None,
        open_thinking: bool = False,
        video_timestamp_tokens: bool = True,
    ) -> MultimodalEncoding:
        counts = tuple(int(count) for count in image_feature_counts)
        if any(count <= 0 for count in counts):
            raise FrontendError("image feature counts must be positive")
        audio_counts = tuple(
            int(count) for count in audio_feature_counts
        )
        if any(count <= 0 for count in audio_counts):
            raise FrontendError("audio feature counts must be positive")
        frame_counts = tuple(
            tuple(int(count) for count in video)
            for video in video_frame_feature_counts
        )
        timestamps = tuple(
            tuple(float(value) for value in video)
            for video in video_timestamps
        )
        if len(frame_counts) != len(timestamps):
            raise FrontendError(
                "video frame counts and timestamps differ"
            )
        if any(
            not video
            or len(video) != len(video_timestamps)
            or any(count <= 0 for count in video)
            for video, video_timestamps in zip(
                frame_counts,
                timestamps,
            )
        ):
            raise FrontendError("video frame metadata is invalid")
        rendered = self.render_chat(
            messages,
            tools=tools,
            open_thinking=open_thinking,
        )
        if getattr(self, "_model_type", "") == "oxr" and "<image>" in rendered:
            # OXR's chat template emits a literal ``<image>`` marker. Map it to
            # the tokenizer's image placeholder before the media expansion.
            rendered = rendered.replace("<image>", "<|image_pad|>")
        if counts and self.image_token_id is None:
            raise FrontendError("tokenizer has no image placeholder token")
        if frame_counts and self.video_token_id is None:
            raise FrontendError("tokenizer has no video placeholder token")
        if (counts or frame_counts) and (
            self.vision_start_token_id is None
            or self.vision_end_token_id is None
        ):
            raise FrontendError("tokenizer has no vision boundary tokens")
        if audio_counts and (
            self.audio_token_id is None
            or self.audio_start_token_id is None
            or self.audio_end_token_id is None
        ):
            raise FrontendError("tokenizer has no audio placeholder tokens")
        input_ids = self._expand_media_placeholders(
            self.encode(rendered),
            image_feature_counts=counts,
            video_frame_feature_counts=frame_counts,
            video_timestamps=timestamps,
            audio_feature_counts=audio_counts,
            video_timestamp_tokens=video_timestamp_tokens,
        )
        token_type_ids = tuple(
            (
                1
                if token_id == self.image_token_id
                else 2
                if token_id == self.video_token_id
                else 3
                if token_id == self.audio_token_id
                else 0
            )
            for token_id in input_ids
        )
        if token_type_ids.count(1) != sum(counts):
            raise FrontendError(
                "expanded image placeholders were not preserved by the tokenizer"
            )
        if token_type_ids.count(2) != sum(
            sum(video) for video in frame_counts
        ):
            raise FrontendError(
                "expanded video placeholders were not preserved by the tokenizer"
            )
        if token_type_ids.count(3) != sum(audio_counts):
            raise FrontendError(
                "expanded audio placeholders were not preserved by the tokenizer"
            )
        return MultimodalEncoding(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            image_count=len(counts),
            video_count=len(frame_counts),
            audio_count=len(audio_counts),
        )

    def _expand_media_placeholders(
        self,
        input_ids: tuple[int, ...],
        *,
        image_feature_counts: tuple[int, ...],
        video_frame_feature_counts: tuple[tuple[int, ...], ...],
        video_timestamps: tuple[tuple[float, ...], ...],
        audio_feature_counts: tuple[int, ...],
        video_timestamp_tokens: bool,
    ) -> tuple[int, ...]:
        if getattr(self, "_model_type", "") == "oxr" and self.image_token_id is not None:
            # OXR wraps every image with two visual-indicator sentinels that
            # resolve to rows of the visual embedding table. The engine later
            # replaces the pad rows with the cached visual-token embeddings.
            expanded: list[int] = []
            image_index = 0
            for token_id in input_ids:
                if token_id == self.image_token_id:
                    if image_index >= len(image_feature_counts):
                        raise FrontendError(
                            "chat has more image placeholders than image inputs"
                        )
                    expanded.append(_OXR_IMAGE_START_ID)
                    expanded.extend(
                        [token_id] * image_feature_counts[image_index]
                    )
                    expanded.append(_OXR_IMAGE_END_ID)
                    image_index += 1
                else:
                    expanded.append(token_id)
            if image_index != len(image_feature_counts):
                raise FrontendError(
                    "image inputs and chat image placeholders differ"
                )
            return tuple(expanded)

        start = self.vision_start_token_id
        end = self.vision_end_token_id
        audio_start = self.audio_start_token_id
        audio_end = self.audio_end_token_id
        image_index = 0
        video_index = 0
        audio_index = 0
        expanded: list[int] = []
        cursor = 0
        while cursor < len(input_ids):
            triplet = input_ids[cursor : cursor + 3]
            if (
                self.image_token_id is not None
                and start is not None
                and end is not None
                and triplet
                == (start, self.image_token_id, end)
            ):
                if image_index >= len(image_feature_counts):
                    raise FrontendError(
                        "chat has more image placeholders than image inputs"
                    )
                expanded.append(start)
                expanded.extend(
                    [self.image_token_id]
                    * image_feature_counts[image_index]
                )
                expanded.append(end)
                image_index += 1
                cursor += 3
                continue
            if (
                self.video_token_id is not None
                and start is not None
                and end is not None
                and triplet
                == (start, self.video_token_id, end)
            ):
                if video_index >= len(video_frame_feature_counts):
                    raise FrontendError(
                        "chat has more video placeholders than video inputs"
                    )
                if video_timestamp_tokens:
                    for frame_count, timestamp in zip(
                        video_frame_feature_counts[video_index],
                        video_timestamps[video_index],
                    ):
                        expanded.extend(
                            self._tokenizer.encode(
                                f"<{timestamp:.1f} seconds>",
                                add_special_tokens=False,
                            ).ids
                        )
                        expanded.append(start)
                        expanded.extend(
                            [self.video_token_id] * frame_count
                        )
                        expanded.append(end)
                else:
                    expanded.append(start)
                    expanded.extend(
                        [self.video_token_id]
                        * sum(video_frame_feature_counts[video_index])
                    )
                    expanded.append(end)
                video_index += 1
                cursor += 3
                continue
            if (
                self.audio_token_id is not None
                and audio_start is not None
                and audio_end is not None
                and triplet
                == (audio_start, self.audio_token_id, audio_end)
            ):
                if audio_index >= len(audio_feature_counts):
                    raise FrontendError(
                        "chat has more audio placeholders than audio inputs"
                    )
                expanded.append(audio_start)
                expanded.extend(
                    [self.audio_token_id]
                    * audio_feature_counts[audio_index]
                )
                expanded.append(audio_end)
                audio_index += 1
                cursor += 3
                continue
            expanded.append(input_ids[cursor])
            cursor += 1
        if image_index != len(image_feature_counts):
            raise FrontendError(
                "image inputs and chat image placeholders differ"
            )
        if video_index != len(video_frame_feature_counts):
            raise FrontendError(
                "video inputs and chat video placeholders differ"
            )
        if audio_index != len(audio_feature_counts):
            raise FrontendError(
                "audio inputs and chat audio placeholders differ"
            )
        return tuple(expanded)

    @staticmethod
    def _normalize_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not messages:
            raise FrontendError("chat requires at least one message")
        normalized: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if not isinstance(role, str) or not role:
                raise FrontendError("every chat message requires a role")
            if not isinstance(content, str):
                if not isinstance(content, Sequence) or isinstance(
                    content,
                    (str, bytes, bytearray),
                ):
                    raise FrontendError(
                        "chat content must be text or a content-part array"
                    )
                parts: list[dict[str, Any]] = []
                for part in content:
                    if not isinstance(part, Mapping):
                        raise FrontendError(
                            "chat content parts must be objects"
                        )
                    parts.append(dict(part))
                content = parts
            item = dict(message)
            item["role"] = role
            item["content"] = content
            if role == "assistant" and isinstance(item.get("tool_calls"), Sequence) and not isinstance(
                item["tool_calls"], (str, bytes, bytearray)
            ):
                normalized_calls: list[dict[str, Any]] = []
                for call in item["tool_calls"]:
                    if not isinstance(call, Mapping):
                        raise FrontendError(
                            "assistant tool_calls entries must be objects"
                        )
                    call_copy = dict(call)
                    function = call_copy.get("function")
                    if isinstance(function, Mapping):
                        function_copy = dict(function)
                        arguments = function_copy.get("arguments")
                        if isinstance(arguments, str):
                            if not arguments.strip():
                                function_copy["arguments"] = {}
                            else:
                                try:
                                    parsed = json.loads(arguments)
                                except json.JSONDecodeError as exc:
                                    raise FrontendError(
                                        "assistant tool call arguments must be valid JSON"
                                    ) from exc
                                if not isinstance(parsed, dict):
                                    raise FrontendError(
                                        "assistant tool call arguments must be a JSON object"
                                    )
                                function_copy["arguments"] = parsed
                        call_copy["function"] = function_copy
                    normalized_calls.append(call_copy)
                item["tool_calls"] = normalized_calls
            # HF chat templates often access these OpenAI optional fields
            # directly under StrictUndefined. Give every message the stable
            # shape expected by those templates without enabling tool calls.
            item.setdefault("reasoning_content", None)
            item.setdefault("tool_calls", [])
            normalized.append(item)
        return normalized

    def _first_token_id(self, *tokens: str) -> int | None:
        return next(
            (
                token_id
                for token in tokens
                if (token_id := self._tokenizer.token_to_id(token))
                is not None
            ),
            None,
        )

    def _special_id(self, config: dict[str, Any], key: str) -> int:
        token = config.get(key)
        if isinstance(token, dict):
            token = token.get("content")
        if not isinstance(token, str):
            raise FrontendError(f"tokenizer config has no string {key}")
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise FrontendError(f"special token {token!r} is absent from vocabulary")
        return token_id

    def _optional_special_id(
        self,
        config: dict[str, Any],
        key: str,
    ) -> int | None:
        if config.get(key) is None:
            return None
        return self._special_id(config, key)


def _template_tojson(
    value: Any,
    *,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    """Jinja filter compatible with vLLM's extended ``tojson`` filter."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=separators,
    )


def _raise_template_error(message: object) -> None:
    raise FrontendError(str(message))


__all__ = [
    "FrontendError",
    "ModelDecodeStream",
    "ModelTokenizer",
    "MultimodalEncoding",
]
