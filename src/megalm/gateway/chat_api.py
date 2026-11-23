from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from megalm.contracts.request import (
    AudioOutputOptions,
    FinishReason,
    OutputModality,
    SamplingParams,
    StructuredOutput,
)
from megalm.gateway.chat_media import parse_media_chat
from megalm.gateway.detokenize import IncrementalDecoder
from megalm.gateway.media_fetch import (
    MEDIA_FETCH_CONFIG,
    MEDIA_FETCH_STATS,
    MediaFetchError,
    resolve_remote_media,
)
from megalm.gateway.replica_types import GatewayError, ReplicaFailure
from megalm.gateway.responses import MegaLMJSONResponse
from megalm.megalm_rs import (
    PyInklingParser,
    PyJsonFormatValidator,
    PyOutputParser,
    reasoning_boundary,
    sse_audio_chunk,
    sse_content_chunk,
    sse_content_chunk_with_logprobs,
    sse_finish_chunk,
    sse_raw_json,
    sse_reasoning_chunk,
    sse_role_chunk,
    sse_tool_call_chunk,
    sse_tool_call_delta_chunk,
    sse_usage_chunk,
)
from megalm.observability import log_event, observe_latency
from megalm.observability_otel import record_request_latency, start_child_span
from megalm.runtime.scheduler import SchedulerOverloaded
from megalm.runtime.tokenizer import FrontendError, ModelTokenizer

_LOGGER = logging.getLogger("megalm.gateway.chat")

# Qwen3-style reasoning markers emitted as plain tokens. Models whose chat
# template ends the generation prompt with the start marker continue inside
# the reasoning block without re-emitting it; the parser is seeded accordingly.
THINKING_START_MARKER = "<think>"
THINKING_END_MARKER = "</think>"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Annotated[str, Field(min_length=1)]
    content: str | list[dict[str, Any]]
    name: str | None = None


class ChatAudioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice: Annotated[str, Field(min_length=1)] = "Chelsie"
    format: Literal["wav"] = "wav"


class ChatCompletionRequest(BaseModel):
    """Text chat subset shared by the built-in UI and OpenAI clients."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: Annotated[list[ChatMessage], Field(min_length=1)]
    max_tokens: Annotated[int | None, Field(gt=0)] = None
    max_completion_tokens: Annotated[int | None, Field(gt=0)] = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: Annotated[float, Field(ge=0)] = 0.0
    top_p: Annotated[float, Field(gt=0, le=1)] = 1.0
    top_k: Annotated[int, Field(ge=0)] = 0
    min_p: Annotated[float, Field(ge=0, le=1)] = 0.0
    repetition_penalty: Annotated[float, Field(gt=0)] = 1.0
    frequency_penalty: Annotated[float, Field(ge=-2, le=2)] = 0.0
    presence_penalty: Annotated[float, Field(ge=-2, le=2)] = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] = Field(default_factory=list)
    ignore_eos: bool = False
    n: Annotated[int, Field(gt=0, le=128)] = 1
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    modalities: list[Literal["text", "audio"]] | None = None
    audio: ChatAudioRequest | None = None
    open_thinking: bool = False
    logprobs: bool = False
    top_logprobs: Annotated[int | None, Field(ge=0, le=20)] = None
    cancel_id: Annotated[str | None, Field(min_length=1)] = None

    def resolved_max_tokens(self) -> int:
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError(
                "max_tokens and max_completion_tokens must agree"
            )
        return self.max_completion_tokens or self.max_tokens or 256

    def sampling(
        self, structured_output: StructuredOutput | None = None
    ) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop_token_ids=tuple(self.stop_token_ids),
            ignore_eos=self.ignore_eos,
            logprobs=self.logprobs,
            structured_output=structured_output,
        )


def _normalized_tools(
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
) -> tuple[tuple[dict[str, Any], ...], str | None]:
    """Validate OpenAI tool inputs and return template-ready tools.

    Returns the normalized tool specs plus an optional forced function name.
    Only ``function`` tools are supported; ``tool_choice`` accepts ``auto``,
    ``none``, ``required`` and ``{"type": "function", ...}`` selectors.
    """
    if tools is None and tool_choice is None:
        return (), None
    if not tools:
        raise ValueError("tool_choice requires a non-empty tools list")

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise ValueError("only function tools are supported")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError("each tool requires a function object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("each tool function requires a name")
        if name in names:
            raise ValueError(f"duplicate tool function name {name!r}")
        names.add(name)
        entry = dict(tool)
        entry["type"] = "function"
        entry["function"] = dict(function)
        normalized.append(entry)

    forced: str | None = None
    if tool_choice is None or tool_choice == "auto":
        pass
    elif tool_choice == "none":
        return (), None
    elif tool_choice == "required":
        pass
    elif isinstance(tool_choice, dict):
        if tool_choice.get("type", "function") != "function":
            raise ValueError("tool_choice only supports function selectors")
        selector = tool_choice.get("function")
        if not isinstance(selector, dict) or not isinstance(
            selector.get("name"), str
        ):
            raise ValueError("tool_choice function requires a name")
        forced = selector["name"]
        if forced not in names:
            raise ValueError(f"unknown tool function {forced!r}")
    else:
        raise ValueError("unsupported tool_choice value")
    return tuple(normalized), forced


def _normalized_response_format(
    response_format: dict[str, Any] | None,
) -> tuple[str, str | None]:
    """Validate response_format and return (mode, schema_json).

    ``mode`` is ``json_object`` or ``json_schema``. The returned schema JSON
    string is fed to the Rust validator; prompt guidance is added separately
    by the route. This is best-effort output shaping, not constrained
    decoding.
    """
    if response_format is None:
        return "", None
    kind = response_format.get("type")
    if kind == "json_object":
        return "json_object", None
    if kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict):
            raise ValueError("json_schema response_format requires a json_schema object")
        schema = spec.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("json_schema response_format requires a schema object")
        return "json_schema", orjson.dumps(schema).decode("utf-8")
    raise ValueError(f"unsupported response_format type {kind!r}")


def _format_instruction(mode: str, schema_json: str | None) -> str:
    if mode == "json_object":
        return (
            "Respond with a single valid JSON object and nothing else. "
            "Do not wrap it in prose or Markdown fences."
        )
    return (
        "Respond with a single valid JSON object matching this JSON Schema, "
        "and nothing else. Do not wrap it in prose or Markdown fences.\n"
        f"JSON Schema: {schema_json}"
    )


async def _fit_text_history(
    messages: tuple[dict[str, Any], ...],
    *,
    tokenizer: ModelTokenizer,
    tokenizer_batcher: Any,
    open_thinking: bool,
    max_prompt_tokens: int | None,
    tools: tuple[dict[str, Any], ...] = (),
) -> tuple[tuple[int, ...], int, bool]:
    """Encode a chat and discard complete oldest turns only when required.

    The third result tells whether the prompt leaves an open reasoning block:
    the Rust core scans token IDs backwards for the last `<think>`/`</think>`
    boundary, stopping at the first unrelated special token (the reference
    Qwen3 semantics). A tokenizer without dedicated think tokens falls back to
    the rendered-text suffix check.
    """

    think_start_id = tokenizer.token_to_id(THINKING_START_MARKER)
    think_end_id = tokenizer.token_to_id(THINKING_END_MARKER)
    token_grammar = think_start_id is not None and think_end_id is not None
    special_ids = tuple(tokenizer.special_token_kinds()) if token_grammar else ()

    async def encode(start: int, system_count: int) -> tuple[tuple[int, ...], bool]:
        retained = messages[:system_count] + messages[start:]
        rendered = tokenizer.render_chat(
            retained,
            tools=tools or None,
            open_thinking=open_thinking,
        )
        token_ids = await tokenizer_batcher.encode(rendered)
        if token_grammar:
            boundary = reasoning_boundary(
                list(token_ids),
                think_start_id,
                think_end_id,
                list(special_ids),
            )
            in_thinking = boundary if boundary is not None else False
        else:
            in_thinking = rendered.rstrip().endswith(THINKING_START_MARKER)
        return token_ids, in_thinking

    system_count = 0
    while (
        system_count < len(messages)
        and messages[system_count].get("role") == "system"
    ):
        system_count += 1

    starts = [system_count]
    starts.extend(
        index
        for index in range(system_count + 1, len(messages))
        if messages[index].get("role") == "user"
    )
    token_ids, ends_with_thinking = await encode(starts[0], system_count)
    if max_prompt_tokens is None or len(token_ids) <= max_prompt_tokens:
        return token_ids, 0, ends_with_thinking

    low = 1
    high = len(starts) - 1
    best: tuple[tuple[int, ...], int, bool] | None = None
    while low <= high:
        middle = (low + high) // 2
        start = starts[middle]
        candidate, candidate_thinking = await encode(start, system_count)
        if len(candidate) <= max_prompt_tokens:
            best = (candidate, start - system_count, candidate_thinking)
            high = middle - 1
        else:
            low = middle + 1

    if best is None:
        newest, _ = await encode(starts[-1], system_count)
        raise ValueError(
            "system prompt and latest turn require "
            f"{len(newest)} prompt tokens, but only {max_prompt_tokens} "
            "remain after reserving output tokens"
        )
    return best


def _choice_request_id(base: str | None, index: int) -> str | None:
    """Distinct replica request id per parallel-sampling choice."""

    if base is None:
        return None
    return base if index == 0 else f"{base}-c{index}"


def install_chat_routes(
    app: FastAPI,
    *,
    replica: Any,
    tokenizer: ModelTokenizer,
    tokenizer_batcher: Any,
    admission_batcher: Any,
    omni_admission_batcher: Any,
    served_model_name: str,
    builtin_tool_executor: Any = None,
) -> None:
    """Install chat routes without coupling HTTP types to Engine Core."""

    # Inkling-style models delimit output sections with special tokens, so the
    # frontend keeps them and the Rust block parser owns the grammar. The
    # marker table is fixed for the lifetime of the Replica.
    grammar = tokenizer.output_grammar()
    marker_kinds = tuple(tokenizer.special_token_kinds().items())
    _, tool_preserve_special = tokenizer.tool_call_parser()
    preserve_special_tokens = grammar == "inkling" or tool_preserve_special

    async def _dispatch_chat(
        payload: ChatCompletionRequest,
    ) -> MegaLMJSONResponse | StreamingResponse:
        """Run one normalized chat request through the shared Gateway core.

        Native protocol adapters translate into ``ChatCompletionRequest`` and
        call this same function, so tool parsing and sampling stay identical.
        """
        if payload.model != served_model_name:
            raise HTTPException(status_code=404, detail="Unknown model")
        try:
            tools, forced_tool = _normalized_tools(
                payload.tools,
                payload.tool_choice,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        request_preserve_special_tokens = preserve_special_tokens and (
            grammar == "inkling" or bool(tools)
        )
        try:
            format_mode, format_schema = _normalized_response_format(
                payload.response_format
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        format_validator = (
            PyJsonFormatValidator(format_schema)
            if format_mode
            else None
        )
        # Engine-level constrained decoding applies to plain-text grammar
        # without tools: Inkling blocks and tool-call payloads have their own
        # structure, and a JSON grammar over the whole generation would fight
        # those markers. Those requests keep the prompt + validator path.
        structured_output: StructuredOutput | None = None
        if format_mode:
            if grammar == "inkling" or tools:
                log_event(
                    _LOGGER,
                    logging.INFO,
                    "constrained_decoding_skipped",
                    model=payload.model,
                    reason="inkling grammar or tools requested",
                )
            else:
                try:
                    structured_output = StructuredOutput(
                        kind=format_mode, schema_json=format_schema
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
        if payload.top_logprobs:
            raise HTTPException(
                status_code=400,
                detail=(
                    "top_logprobs is not implemented by this Gateway; only"
                    " the sampled token logprob is available"
                ),
            )
        modalities = tuple(
            OutputModality(item)
            for item in (payload.modalities or ["text"])
        )
        if not modalities or len(set(modalities)) != len(modalities):
            raise HTTPException(
                status_code=400,
                detail="modalities must be unique and non-empty",
            )
        wants_audio = OutputModality.AUDIO in modalities
        if wants_audio != (payload.audio is not None):
            raise HTTPException(
                status_code=400,
                detail=(
                    "audio options must be supplied exactly when "
                    "the audio modality is requested"
                ),
            )
        descriptor = await replica.describe()
        supported_outputs = set(
            descriptor.get("output_modalities", ("text",))
        )
        if any(item.value not in supported_outputs for item in modalities):
            raise HTTPException(
                status_code=400,
                detail=(
                    "the loaded Replica does not support the requested "
                    "output modalities"
                ),
            )
        audio_capability = descriptor.get("audio_output")
        if wants_audio and isinstance(audio_capability, dict):
            voices = tuple(audio_capability.get("voices", ()))
            assert payload.audio is not None
            if voices and payload.audio.voice not in voices:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"unknown voice {payload.audio.voice!r}; "
                        f"available={voices}"
                    ),
                )
        audio_output = (
            None
            if payload.audio is None
            else AudioOutputOptions(
                voice=payload.audio.voice,
                format=payload.audio.format,
            )
        )
        if isinstance(payload.stop, str):
            stop_sequences: tuple[str, ...] = (
                () if not payload.stop else (payload.stop,)
            )
        elif payload.stop:
            stop_sequences = tuple(
                stop for stop in payload.stop if stop
            )
        else:
            stop_sequences = ()
        include_usage = bool(
            (payload.stream_options or {}).get("include_usage")
        )
        max_new_tokens = payload.resolved_max_tokens()
        max_model_len = descriptor.get("max_model_len")
        max_prompt_tokens = (
            max_model_len - max_new_tokens
            if isinstance(max_model_len, int)
            else None
        )
        if max_prompt_tokens is not None and max_prompt_tokens <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"max_tokens={max_new_tokens} must be smaller than "
                    f"the Replica context capacity {max_model_len}"
                ),
            )
        request_count = max(1, payload.n)
        # Prompt tail token ids seed the Inkling block parser; multimodal
        # requests do not expose them, so the parser starts between blocks.
        prompt_token_ids: tuple[int, ...] = ()
        initial_in_thinking = False
        base_request_id = payload.cancel_id
        for _ in range(request_count):
            replica.frontend_request_started()
        dropped_messages = 0
        streams: list[Any] = []
        submit_failed = False
        try:
            await asyncio.sleep(0)
            messages = tuple(
                message.model_dump(exclude_none=True)
                for message in payload.messages
            )
            if format_mode:
                messages = (
                    {
                        "role": "system",
                        "content": _format_instruction(
                            format_mode, format_schema
                        ),
                    },
                    *messages,
                )
            has_non_text_part = any(
                isinstance(message["content"], list)
                and any(
                    not isinstance(part, dict)
                    or part.get("type") != "text"
                    for part in message["content"]
                )
                for message in messages
            )
            if has_non_text_part:
                messages, fetch_report = await resolve_remote_media(
                    messages,
                    config=MEDIA_FETCH_CONFIG,
                    stats=MEDIA_FETCH_STATS,
                )
                if fetch_report.fetched:
                    log_event(
                        _LOGGER,
                        logging.INFO,
                        "chat_media_remote_resolved",
                        fetched=fetch_report.fetched,
                        bytes=fetch_report.bytes_downloaded,
                        hosts=list(fetch_report.hosts),
                    )
                media = await asyncio.to_thread(
                    parse_media_chat,
                    messages,
                )
                for choice_index in range(request_count):
                    streams.append(
                        await omni_admission_batcher.submit(
                            media.messages,
                            media.images,
                            videos=media.videos,
                            audios=media.audios,
                            max_new_tokens=max_new_tokens,
                            sampling=payload.sampling(structured_output),
                            request_id=_choice_request_id(
                                base_request_id, choice_index
                            ),
                            open_thinking=payload.open_thinking,
                            output_modalities=modalities,
                            audio_output=audio_output,
                            tools=tools,
                        )
                    )
            else:
                with start_child_span(
                    "tokenizer.encode",
                    **{"gen_ai.request.model": payload.model},
                ):
                    token_ids, dropped_messages, initial_in_thinking = (
                        await _fit_text_history(
                            messages,
                            tokenizer=tokenizer,
                            tokenizer_batcher=tokenizer_batcher,
                            open_thinking=payload.open_thinking,
                            max_prompt_tokens=max_prompt_tokens,
                            tools=tools,
                        )
                    )
                if dropped_messages:
                    log_event(
                        _LOGGER,
                        logging.INFO,
                        "chat_history_trimmed",
                        dropped_messages=dropped_messages,
                        prompt_tokens=len(token_ids),
                        max_new_tokens=max_new_tokens,
                        context_capacity=max_model_len,
                        open_thinking=payload.open_thinking,
                    )
                prompt_token_ids = token_ids
                with start_child_span(
                    "gateway.admit",
                    **{
                        "megalm.choice_count": request_count,
                        "gen_ai.usage.input_tokens": len(token_ids),
                    },
                ) as admit_span:
                    for choice_index in range(request_count):
                        stream = await admission_batcher.submit(
                            token_ids,
                            max_new_tokens=max_new_tokens,
                            sampling=payload.sampling(structured_output),
                            request_id=_choice_request_id(
                                base_request_id, choice_index
                            ),
                            output_modalities=modalities,
                            audio_output=audio_output,
                        )
                        streams.append(stream)
                        if admit_span is not None:
                            admit_span.set_attribute(
                                f"megalm.request_id.c{choice_index}",
                                stream.request_id,
                            )
        except SchedulerOverloaded as exc:
            submit_failed = True
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except MediaFetchError as exc:
            submit_failed = True
            raise HTTPException(
                status_code=exc.status_code, detail=exc.message
            ) from exc
        except (FrontendError, ValueError) as exc:
            submit_failed = True
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GatewayError as exc:
            submit_failed = True
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            for _ in range(request_count):
                replica.frontend_request_finished()
            if submit_failed:
                for created_stream in streams:
                    replica.cancel(created_stream.request_id)

        created = int(time.time())
        single_stream = streams[0]
        if request_count == 1 and payload.stream:
            return StreamingResponse(
                _stream_chat(
                    stream=single_stream,
                    replica=replica,
                    tokenizer=tokenizer,
                    model=served_model_name,
                    created=created,
                    modalities=modalities,
                    stop_sequences=stop_sequences,
                    include_usage=include_usage,
                    tools=tools,
                    forced_tool=forced_tool,
                    format_validator=format_validator,
                    initial_in_thinking=initial_in_thinking,
                    prompt_token_ids=prompt_token_ids,
                    preserve_special_tokens=request_preserve_special_tokens,
                    marker_kinds=marker_kinds,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-MegaLM-Dropped-Messages": str(dropped_messages),
                },
            )
        if request_count == 1:
            response = await _collect_chat(
                    stream=single_stream,
                    replica=replica,
                    tokenizer=tokenizer,
                    model=served_model_name,
                    created=created,
                    modalities=modalities,
                    stop_sequences=stop_sequences,
                    tools=tools,
                    forced_tool=forced_tool,
                    format_validator=format_validator,
                    initial_in_thinking=initial_in_thinking,
                    prompt_token_ids=prompt_token_ids,
                    preserve_special_tokens=request_preserve_special_tokens,
                    marker_kinds=marker_kinds,
                )
            response["megalm"] = {"dropped_messages": dropped_messages}
            return MegaLMJSONResponse(response)
        if payload.stream:
            return StreamingResponse(
                _stream_chat_multi(
                    streams=streams,
                    replica=replica,
                    tokenizer=tokenizer,
                    model=served_model_name,
                    created=created,
                    modalities=modalities,
                    stop_sequences=stop_sequences,
                    include_usage=include_usage,
                    tools=tools,
                    forced_tool=forced_tool,
                    format_validator=format_validator,
                    initial_in_thinking=initial_in_thinking,
                    prompt_token_ids=prompt_token_ids,
                    preserve_special_tokens=request_preserve_special_tokens,
                    marker_kinds=marker_kinds,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-MegaLM-Dropped-Messages": str(dropped_messages),
                },
            )
        response = await _collect_chat_multi(
            streams=streams,
            replica=replica,
            tokenizer=tokenizer,
            model=served_model_name,
            created=created,
            modalities=modalities,
            stop_sequences=stop_sequences,
            tools=tools,
            forced_tool=forced_tool,
            format_validator=format_validator,
            prompt_token_ids=prompt_token_ids,
            preserve_special_tokens=request_preserve_special_tokens,
            marker_kinds=marker_kinds,
        )
        response["megalm"] = {"dropped_messages": dropped_messages}
        return MegaLMJSONResponse(response)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        payload: ChatCompletionRequest,
    ) -> MegaLMJSONResponse | StreamingResponse:
        return await _dispatch_chat(payload)

    from megalm.gateway.agent_protocols import install_agent_routes

    install_agent_routes(
        app,
        dispatch_chat=_dispatch_chat,
        served_model_name=served_model_name,
        tokenizer=tokenizer,
        tokenizer_batcher=tokenizer_batcher,
        builtin_tool_executor=builtin_tool_executor,
    )


TOOL_ARGUMENT_FRAGMENT_CHARS = 16


def _stream_tool_arguments() -> bool:
    """Whether the gateway emits tool arguments as incremental SSE deltas.

    Default stays the legacy single complete chunk so existing clients see no
    change. Opt in with ``MEGALM_STREAM_TOOL_ARGUMENTS=1``.
    """

    return os.environ.get("MEGALM_STREAM_TOOL_ARGUMENTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def _stream_chat(
    *,
    stream: Any,
    replica: Any,
    tokenizer: ModelTokenizer,
    model: str,
    created: int,
    modalities: tuple[OutputModality, ...],
    stop_sequences: tuple[str, ...] = (),
    include_usage: bool = False,
    tools: tuple[dict[str, Any], ...] = (),
    forced_tool: str | None = None,
    format_validator: PyJsonFormatValidator | None = None,
    initial_in_thinking: bool = False,
    prompt_token_ids: tuple[int, ...] = (),
    preserve_special_tokens: bool = False,
    marker_kinds: tuple[tuple[int, str], ...] = (),
    index: int = 0,
    response_id: str | None = None,
    usage_sink: list[tuple[int, int]] | None = None,
) -> AsyncIterator[bytes]:
    chat_id = response_id or stream.request_id
    choice_index = index
    started_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    decoder = IncrementalDecoder(
        tokenizer,
        skip_special_tokens=not preserve_special_tokens,
    )
    parser = _make_output_parser(
        stream,
        tokenizer=tokenizer,
        stop_sequences=stop_sequences,
        tools=tools,
        forced_tool=forced_tool,
        initial_in_thinking=initial_in_thinking,
        prompt_token_ids=prompt_token_ids,
        preserve_special_tokens=preserve_special_tokens,
        marker_kinds=marker_kinds,
    )
    fragment_tool_args = _stream_tool_arguments() and isinstance(parser, PyOutputParser)
    fragment_index: dict[str, int] = {}
    token_count = 0
    tool_call_count = 0
    emitted_tool_names: list[str] = []
    terminal = False

    def drain_tool_calls(calls: list[tuple[str, str, str]]) -> list[bytes]:
        nonlocal tool_call_count
        chunks: list[bytes] = []
        for call_id, name, arguments in calls:
            chunks.append(
                sse_tool_call_chunk(
                    chat_id,
                    model,
                    created,
                    choice_index,
                    call_id,
                    tool_call_count,
                    name,
                    arguments,
                )
            )
            tool_call_count += 1
            emitted_tool_names.append(name)
        return chunks

    def drain_tool_deltas(
        deltas: list[tuple[str, str | None, str]],
    ) -> list[bytes]:
        nonlocal tool_call_count
        chunks: list[bytes] = []
        for call_id, name, arguments in deltas:
            if name is not None:
                index = tool_call_count
                fragment_index[call_id] = index
                tool_call_count += 1
                emitted_tool_names.append(name)
                chunks.append(
                    sse_tool_call_delta_chunk(
                        chat_id,
                        model,
                        created,
                        choice_index,
                        index,
                        call_id,
                        name,
                        arguments,
                    )
                )
            else:
                chunks.append(
                    sse_tool_call_delta_chunk(
                        chat_id,
                        model,
                        created,
                        choice_index,
                        fragment_index.get(call_id, 0),
                        None,
                        None,
                        arguments,
                    )
                )
        return chunks

    stream_text_parts: list[str] = []

    def check_format() -> None:
        if format_validator is None or not stream_text_parts:
            return
        try:
            format_validator.validate("".join(stream_text_parts))
        except ValueError as exc:
            log_event(
                _LOGGER,
                logging.WARNING,
                "response_format_invalid",
                request_id=stream.request_id,
                error=str(exc),
            )

    try:
        yield sse_role_chunk(chat_id, model, created, choice_index)
        while True:
            item = await stream.events.get()
            if isinstance(item, ReplicaFailure):
                raise RuntimeError("Replica chat stream failed") from item.error
            token_count += len(item.token_ids)
            if item.token_ids and OutputModality.TEXT in modalities:
                if first_token_ns is None:
                    first_token_ns = time.perf_counter_ns()
                delta = decoder.push(item.token_ids)
                if fragment_tool_args:
                    content, reasoning, deltas, stopped = parser.push_fragments(
                        delta, TOOL_ARGUMENT_FRAGMENT_CHARS
                    )
                    tool_chunks = drain_tool_deltas(deltas)
                else:
                    content, reasoning, calls, stopped = parser.push(delta)
                    tool_chunks = drain_tool_calls(calls)
                for chunk in tool_chunks:
                    yield chunk
                if reasoning:
                    yield sse_reasoning_chunk(
                        chat_id,
                        model,
                        created,
                        reasoning,
                        choice_index,
                    )
                if content or item.logprobs:
                    entries_json: str | None = None
                    if item.logprobs:
                        entries_json = orjson.dumps(
                            _logprob_entries(
                                tokenizer,
                                item.token_ids,
                                item.logprobs,
                            )
                        ).decode("utf-8")
                    if content:
                        stream_text_parts.append(content)
                    if entries_json is not None:
                        yield sse_content_chunk_with_logprobs(
                            chat_id,
                            model,
                            created,
                            content,
                            choice_index,
                            entries_json,
                        )
                    elif content:
                        yield sse_content_chunk(
                            chat_id,
                            model,
                            created,
                            content,
                            choice_index,
                        )
                if stopped:
                    check_format()
                    _record_frontend_latency(
                        replica,
                        started_ns=started_ns,
                        first_token_ns=first_token_ns,
                        finished_ns=time.perf_counter_ns(),
                        token_count=token_count,
                    )
                    terminal = True
                    replica.cancel(stream.request_id)
                    if usage_sink is not None:
                        usage_sink.append((stream.prompt_tokens, token_count))
                    yield sse_finish_chunk(
                        chat_id,
                        model,
                        created,
                        choice_index,
                        "tool_calls"
                        if tool_call_count
                        else "stop",
                        (stream.prompt_tokens, token_count)
                        if include_usage
                        else None,
                    )
                    if include_usage:
                        yield sse_usage_chunk(
                            stream.request_id,
                            model,
                            created,
                            stream.prompt_tokens,
                            token_count,
                        )
                    yield b"data: [DONE]\n\n"
                    return
            for chunk in item.audio:
                yield sse_audio_chunk(
                    chat_id,
                    model,
                    created,
                    base64.b64encode(chunk.data).decode("ascii"),
                    chunk.format,
                    choice_index,
                )
            if item.finish_reason is not None:
                if item.error is not None:
                    terminal = True
                    _record_frontend_latency(
                        replica,
                        started_ns=started_ns,
                        first_token_ns=first_token_ns,
                        finished_ns=time.perf_counter_ns(),
                        token_count=token_count,
                    )
                    yield sse_raw_json(
                        orjson.dumps(
                            {
                                "error": {
                                    "code": item.error.code,
                                    "message": item.error.message,
                                }
                            }
                        ).decode("utf-8")
                    )
                    yield b"data: [DONE]\n\n"
                    return
                tail_reasoning = ""
                if OutputModality.TEXT in modalities:
                    tail = decoder.finish()
                    if fragment_tool_args:
                        tail_content, tail_reasoning, deltas, _ = (
                            parser.push_fragments(
                                tail, TOOL_ARGUMENT_FRAGMENT_CHARS
                            )
                        )
                        for chunk in drain_tool_deltas(deltas):
                            yield chunk
                        final_content, final_reasoning, deltas = parser.finish_fragments(
                            TOOL_ARGUMENT_FRAGMENT_CHARS
                        )
                        for chunk in drain_tool_deltas(deltas):
                            yield chunk
                    else:
                        tail_content, tail_reasoning, calls, _ = parser.push(tail)
                        for chunk in drain_tool_calls(calls):
                            yield chunk
                        final_content, final_reasoning, calls = parser.finish()
                        for chunk in drain_tool_calls(calls):
                            yield chunk
                    tail = tail_content + final_content
                    tail_reasoning = tail_reasoning + final_reasoning
                else:
                    tail = ""
                if tail_reasoning:
                    yield sse_reasoning_chunk(
                        chat_id,
                        model,
                        created,
                        tail_reasoning,
                        choice_index,
                    )
                if tail:
                    stream_text_parts.append(tail)
                    yield sse_content_chunk(
                        chat_id,
                        model,
                        created,
                        tail,
                        choice_index,
                    )
                finish_reason = (
                    "tool_calls"
                    if tool_call_count
                    else _finish_reason(item.finish_reason)
                )
                _check_forced_tool(
                    forced_tool,
                    emitted_tool_names,
                    stream.request_id,
                )
                check_format()
                _record_frontend_latency(
                    replica,
                    started_ns=started_ns,
                    first_token_ns=first_token_ns,
                    finished_ns=time.perf_counter_ns(),
                    token_count=token_count,
                )
                terminal = True
                if usage_sink is not None:
                    usage_sink.append((stream.prompt_tokens, token_count))
                yield sse_finish_chunk(
                    chat_id,
                    model,
                    created,
                    choice_index,
                    finish_reason,
                    (stream.prompt_tokens, token_count)
                    if include_usage
                    else None,
                )
                if include_usage:
                    yield sse_usage_chunk(
                        chat_id,
                        model,
                        created,
                        stream.prompt_tokens,
                        token_count,
                    )
                yield b"data: [DONE]\n\n"
                return
    finally:
        _record_parser_telemetry(
            replica,
            parser,
            request_id=stream.request_id,
            streaming=True,
        )
        if not terminal:
            replica.cancel(stream.request_id)


async def _collect_chat(
    *,
    stream: Any,
    replica: Any,
    tokenizer: ModelTokenizer,
    model: str,
    created: int,
    modalities: tuple[OutputModality, ...],
    stop_sequences: tuple[str, ...] = (),
    tools: tuple[dict[str, Any], ...] = (),
    forced_tool: str | None = None,
    format_validator: PyJsonFormatValidator | None = None,
    initial_in_thinking: bool = False,
    prompt_token_ids: tuple[int, ...] = (),
    preserve_special_tokens: bool = False,
    marker_kinds: tuple[tuple[int, str], ...] = (),
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    decoder = IncrementalDecoder(
        tokenizer,
        skip_special_tokens=not preserve_special_tokens,
    )
    parser = _make_output_parser(
        stream,
        tokenizer=tokenizer,
        stop_sequences=stop_sequences,
        tools=tools,
        forced_tool=forced_tool,
        initial_in_thinking=initial_in_thinking,
        prompt_token_ids=prompt_token_ids,
        preserve_special_tokens=preserve_special_tokens,
        marker_kinds=marker_kinds,
    )
    token_ids: list[int] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    logprob_entries: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    emitted_tool_names: list[str] = []
    audio_chunks: list[bytes] = []
    audio_format = "wav"
    stopped_by_parser = False
    terminal = False

    def record_calls(calls: list[tuple[str, str, str]]) -> None:
        for call_id, name, arguments in calls:
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )
            emitted_tool_names.append(name)

    try:
        while True:
            item = await stream.events.get()
            if isinstance(item, ReplicaFailure):
                raise RuntimeError("Replica chat request failed") from item.error
            token_ids.extend(item.token_ids)
            for chunk in item.audio:
                audio_chunks.append(chunk.data)
                audio_format = chunk.format
            if (
                item.token_ids
                and OutputModality.TEXT in modalities
            ):
                if first_token_ns is None:
                    first_token_ns = time.perf_counter_ns()
                delta = decoder.push(item.token_ids)
                content, reasoning, calls, stopped = parser.push(delta)
                record_calls(calls)
                if item.logprobs:
                    logprob_entries.extend(
                        _logprob_entries(
                            tokenizer,
                            item.token_ids,
                            item.logprobs,
                        )
                    )
                if reasoning:
                    reasoning_parts.append(reasoning)
                if content:
                    text_parts.append(content)
                if stopped:
                    stopped_by_parser = True
                    terminal = True
                    replica.cancel(stream.request_id)
                    break
            if item.finish_reason is None:
                continue
            if item.error is not None:
                terminal = True
                _record_frontend_latency(
                    replica,
                    started_ns=started_ns,
                    first_token_ns=first_token_ns,
                    finished_ns=time.perf_counter_ns(),
                    token_count=len(token_ids),
                )
                raise HTTPException(
                    status_code=500,
                    detail=item.error.message,
                )
            tail_reasoning = ""
            if OutputModality.TEXT in modalities:
                tail = decoder.finish()
                tail_content, tail_reasoning, calls, _ = parser.push(tail)
                record_calls(calls)
                final_content, final_reasoning, calls = parser.finish()
                record_calls(calls)
                if tail_content + final_content:
                    text_parts.append(tail_content + final_content)
                tail_reasoning = tail_reasoning + final_reasoning
            if tail_reasoning:
                reasoning_parts.append(tail_reasoning)
            _record_frontend_latency(
                replica,
                started_ns=started_ns,
                first_token_ns=first_token_ns,
                finished_ns=time.perf_counter_ns(),
                token_count=len(token_ids),
            )
            terminal = True
            break
        if tool_calls:
            finish_reason = "tool_calls"
        elif stopped_by_parser:
            finish_reason = "stop"
        else:
            finish_reason = _finish_reason(item.finish_reason)
        _check_forced_tool(forced_tool, emitted_tool_names, stream.request_id)
        if format_validator is not None and OutputModality.TEXT in modalities:
            try:
                format_validator.validate("".join(text_parts))
            except ValueError as exc:
                log_event(
                    _LOGGER,
                    logging.WARNING,
                    "response_format_invalid",
                    request_id=stream.request_id,
                    error=str(exc),
                )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": (
                "".join(text_parts)
                if OutputModality.TEXT in modalities
                else None
            ),
            **(
                {}
                if not audio_chunks
                else {
                    "audio": {
                        "data": base64.b64encode(
                            b"".join(audio_chunks)
                        ).decode("ascii"),
                        "format": audio_format,
                    }
                }
            ),
        }
        joined_reasoning = "".join(reasoning_parts)
        if joined_reasoning:
            message["reasoning_content"] = joined_reasoning
        if tool_calls:
            # OpenAI clients expect `content` to be absent (null) when the
            # assistant turn carries only tool calls.
            if not message["content"]:
                message["content"] = None
            message["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-{stream.request_id}",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": (
                        {"content": logprob_entries}
                        if logprob_entries
                        else None
                    ),
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage(stream.prompt_tokens, len(token_ids)),
        }
    finally:
        _record_parser_telemetry(
            replica,
            parser,
            request_id=stream.request_id,
            streaming=False,
        )
        if not terminal:
            replica.cancel(stream.request_id)


def _make_output_parser(
    stream: Any,
    *,
    tokenizer: ModelTokenizer,
    stop_sequences: tuple[str, ...],
    tools: tuple[dict[str, Any], ...],
    forced_tool: str | None,
    initial_in_thinking: bool = False,
    prompt_token_ids: tuple[int, ...] = (),
    preserve_special_tokens: bool = False,
    marker_kinds: tuple[tuple[int, str], ...] = (),
) -> PyOutputParser | PyInklingParser:
    """Build the streaming output parser for the served model's grammar.

    ``preserve_special_tokens`` is set by the route from the tokenizer's
    declared output grammar: Inkling-style models carry block markers that the
    Rust parser needs, while plain-text models keep the ``<think>`` splitter.
    """

    if preserve_special_tokens and tokenizer.output_grammar() == "inkling":
        return PyInklingParser(
            stream.request_id,
            list(stop_sequences),
            list(prompt_token_ids),
            list(marker_kinds),
        )
    selected_tools = [
        tool
        for tool in tools
        if forced_tool is None or tool["function"]["name"] == forced_tool
    ]
    tool_names = [tool["function"]["name"] for tool in selected_tools]
    parser_name, _ = tokenizer.tool_call_parser()
    if not selected_tools:
        parser_name = "generic_json"
    tools_json = None
    if parser_name not in ("", "generic_json", "json", "none"):
        tools_json = orjson.dumps(
            [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description"),
                    "parameters": tool["function"].get("parameters") or {},
                    "strict": tool["function"].get("strict"),
                }
                for tool in selected_tools
            ]
        ).decode("utf-8")
    return PyOutputParser(
        stream.request_id,
        list(stop_sequences),
        tool_names,
        THINKING_START_MARKER,
        THINKING_END_MARKER,
        initial_in_thinking,
        parser_name,
        tools_json,
        forced_tool,
    )


def _logprob_entries(
    tokenizer: ModelTokenizer,
    token_ids: tuple[int, ...],
    logprobs: tuple[float, ...],
) -> list[dict[str, Any]]:
    """OpenAI `logprobs.content` entries for one aligned token batch."""

    if len(token_ids) != len(logprobs):
        return []
    return [
        {
            "token": tokenizer.decode((token_id,)),
            "logprob": float(value),
            "top_logprobs": [],
        }
        for token_id, value in zip(token_ids, logprobs)
    ]


def _check_forced_tool(
    forced_tool: str | None,
    emitted_names: list[str],
    request_id: str,
) -> None:
    """Log a warning when a forced tool request produced no matching call."""

    if forced_tool and forced_tool not in emitted_names:
        log_event(
            _LOGGER,
            logging.WARNING,
            "forced_tool_not_emitted",
            request_id=request_id,
            expected=forced_tool,
            emitted=list(emitted_names),
        )


def _record_parser_telemetry(
    replica: Any,
    parser: Any,
    *,
    request_id: str,
    streaming: bool,
) -> None:
    """Fold one request's Rust parser counters into replica metrics and emit
    structured diagnostics for any block the parser had to degrade.

    A degraded block is surfaced as visible text so the stream never breaks,
    but it must be observable: we log the parser, failure kind, byte position
    and a raw context snippet, mirroring the reference parser's error context.
    """

    # Drain diagnostics first: OutputParser folds their count into `stats`.
    diagnostics = parser.take_diagnostics()
    stats = parser.stats()
    metrics = getattr(replica, "metrics", None)
    if metrics is not None:
        metrics.parser_pushes = getattr(metrics, "parser_pushes", 0) + stats[0]
        metrics.parser_finishes = getattr(metrics, "parser_finishes", 0) + stats[1]
        metrics.parser_calls = getattr(metrics, "parser_calls", 0) + stats[2]
        metrics.parser_diagnostics = (
            getattr(metrics, "parser_diagnostics", 0) + stats[3]
        )
        metrics.parser_bytes = getattr(metrics, "parser_bytes", 0) + stats[4]

    for name, kind, message, position, context, expected in diagnostics:
        log_event(
            _LOGGER,
            logging.WARNING,
            "parser_degraded",
            request_id=request_id,
            parser=name,
            kind=kind,
            message=message,
            position=position,
            context=context,
            expected=expected,
            streaming=streaming,
        )


def _record_frontend_latency(
    replica: Any,
    *,
    started_ns: int,
    first_token_ns: int | None,
    finished_ns: int,
    token_count: int,
) -> None:
    """Aggregate TTFT and time-per-output-token into replica metrics.

    Replica metrics classes expose ``ttft_ms_total``/``ttft_samples`` and the
    ``*_ms_histogram`` counters; lightweight test doubles lacking them are
    skipped silently. Recorded for both embedded and process replicas so
    ``megalm_replica_*`` is topology-independent.
    """
    metrics = getattr(replica, "metrics", None)
    if metrics is None or not hasattr(metrics, "ttft_samples"):
        return
    e2e_ms = (finished_ns - started_ns) / 1_000_000
    e2e_hist = getattr(metrics, "e2e_ms_histogram", None)
    if e2e_hist is not None:
        observe_latency(e2e_hist, e2e_ms)
    if first_token_ns is not None:
        ttft_ms = (first_token_ns - started_ns) / 1_000_000
        metrics.ttft_ms_total += ttft_ms
        metrics.ttft_samples += 1
        ttft_hist = getattr(metrics, "ttft_ms_histogram", None)
        if ttft_hist is not None:
            observe_latency(ttft_hist, ttft_ms)
        if token_count > 1:
            tpot_ms = (
                (finished_ns - first_token_ns) / (token_count - 1) / 1_000_000
            )
            metrics.tpot_ms_total += tpot_ms
            metrics.tpot_samples += 1
            tpot_hist = getattr(metrics, "tpot_ms_histogram", None)
            if tpot_hist is not None:
                observe_latency(tpot_hist, tpot_ms)
    record_request_latency(
        ttft_ms=(
            (first_token_ns - started_ns) / 1_000_000
            if first_token_ns is not None
            else None
        ),
        tpot_ms=(
            (finished_ns - first_token_ns) / (token_count - 1) / 1_000_000
            if first_token_ns is not None and token_count > 1
            else None
        ),
        e2e_ms=e2e_ms,
        replica_mode=getattr(replica, "replica_mode", "embedded"),
    )


async def _stream_chat_multi(
    *,
    streams: list[Any],
    replica: Any,
    tokenizer: ModelTokenizer,
    model: str,
    created: int,
    modalities: tuple[OutputModality, ...],
    stop_sequences: tuple[str, ...] = (),
    include_usage: bool = False,
    tools: tuple[dict[str, Any], ...] = (),
    forced_tool: str | None = None,
    format_validator: PyJsonFormatValidator | None = None,
    initial_in_thinking: bool = False,
    prompt_token_ids: tuple[int, ...] = (),
    preserve_special_tokens: bool = False,
    marker_kinds: tuple[tuple[int, str], ...] = (),
) -> AsyncIterator[bytes]:
    """Interleave N parallel-sampling choices into one SSE stream.

    Each choice runs the full single-choice pipeline (detokenize, stop
    matching, tool-call parsing) and pushes its frames into a shared queue;
    chunks carry the per-choice ``index``. One usage trailer with summed
    counts is emitted when every choice is terminal.
    """

    shared_id = streams[0].request_id
    count = len(streams)
    queue: asyncio.Queue = asyncio.Queue()
    usage_sink: list[tuple[int, int]] = []

    async def pump(choice_index: int, stream: Any) -> None:
        try:
            async for payload in _stream_chat(
                stream=stream,
                replica=replica,
                tokenizer=tokenizer,
                model=model,
                created=created,
                modalities=modalities,
                stop_sequences=stop_sequences,
                include_usage=False,
                tools=tools,
                forced_tool=forced_tool,
                format_validator=format_validator,
                initial_in_thinking=initial_in_thinking,
                prompt_token_ids=prompt_token_ids,
                preserve_special_tokens=preserve_special_tokens,
                marker_kinds=marker_kinds,
                index=choice_index,
                response_id=shared_id,
                usage_sink=usage_sink,
            ):
                if payload == b"data: [DONE]\n\n":
                    break
                await queue.put(payload)
            await queue.put(None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(exc)

    tasks = [
        asyncio.create_task(pump(i, stream))
        for i, stream in enumerate(streams)
    ]
    finished = 0
    terminal = False
    try:
        while finished < count:
            item = await queue.get()
            if isinstance(item, BaseException):
                raise item
            if item is None:
                finished += 1
                continue
            yield item
        total_prompt = sum(prompt for prompt, _ in usage_sink)
        total_completion = sum(completion for _, completion in usage_sink)
        if include_usage:
            yield sse_usage_chunk(
                shared_id,
                model,
                created,
                total_prompt,
                total_completion,
            )
        yield b"data: [DONE]\n\n"
        terminal = True
    finally:
        for task in tasks:
            task.cancel()
        if not terminal:
            for stream in streams:
                replica.cancel(stream.request_id)


async def _collect_chat_multi(
    *,
    streams: list[Any],
    replica: Any,
    tokenizer: ModelTokenizer,
    model: str,
    created: int,
    modalities: tuple[OutputModality, ...],
    stop_sequences: tuple[str, ...] = (),
    tools: tuple[dict[str, Any], ...] = (),
    forced_tool: str | None = None,
    format_validator: PyJsonFormatValidator | None = None,
    prompt_token_ids: tuple[int, ...] = (),
    preserve_special_tokens: bool = False,
    marker_kinds: tuple[tuple[int, str], ...] = (),
) -> dict[str, Any]:
    tasks = [
        asyncio.create_task(
            _collect_chat(
                stream=stream,
                replica=replica,
                tokenizer=tokenizer,
                model=model,
                created=created,
                modalities=modalities,
                stop_sequences=stop_sequences,
                tools=tools,
                forced_tool=forced_tool,
                format_validator=format_validator,
                prompt_token_ids=prompt_token_ids,
                preserve_special_tokens=preserve_special_tokens,
                marker_kinds=marker_kinds,
            )
        )
        for stream in streams
    ]
    try:
        responses = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
    merged_choices = []
    prompt_total = 0
    completion_total = 0
    for index, response in enumerate(responses):
        choice = dict(response["choices"][0])
        choice["index"] = index
        merged_choices.append(choice)
        usage = response["usage"]
        prompt_total += usage["prompt_tokens"]
        completion_total += usage["completion_tokens"]
    return {
        "id": responses[0]["id"],
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": merged_choices,
        "usage": {
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "total_tokens": prompt_total + completion_total,
        },
    }


def _finish_reason(reason: FinishReason) -> str:
    return {
        FinishReason.STOP: "stop",
        FinishReason.LENGTH: "length",
        FinishReason.CANCELLED: "stop",
        FinishReason.REJECTED: "error",
        FinishReason.ERROR: "error",
    }[reason]


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "install_chat_routes",
]
