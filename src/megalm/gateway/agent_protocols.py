"""Native Anthropic Messages and OpenAI Responses protocol adapters.

Both endpoints translate their wire format into the gateway's existing
``ChatCompletionRequest`` and reuse the same submission/parsing pipeline as
``/v1/chat/completions``. Only the request and response shells are protocol
specific; tool parsing, sampling and streaming stay in the Rust-backed core.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from megalm.gateway.responses import MegaLMJSONResponse

BuiltinToolExecutor = Callable[
    [list[dict[str, Any]], "ResponsesRequest"], list[dict[str, Any]] | None
]

_CHAT_REQUEST: Any = None


def _chat_request() -> Any:
    """Lazy import avoids a module cycle with ``gateway.chat_api``."""

    global _CHAT_REQUEST
    if _CHAT_REQUEST is None:
        from megalm.gateway.chat_api import ChatCompletionRequest

        _CHAT_REQUEST = ChatCompletionRequest
    return _CHAT_REQUEST


# Responses API is stateful: ``store`` / ``previous_response_id`` and the
# retrieve / cancel endpoints operate on this in-process response store.
_RESPONSE_STORE: dict[str, dict[str, Any]] = {}
_RESPONSE_EVENTS: dict[str, list[bytes]] = {}


# ---------------------------------------------------------------------------
# Anthropic Messages request model
# ---------------------------------------------------------------------------


class AnthropicToolChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None


class AnthropicTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AnthropicContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None
    source: dict[str, Any] | None = None
    id: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    thinking: str | None = None
    signature: str | None = None


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant", "system"]
    content: str | list[AnthropicContentBlock]


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(gt=0)
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None
    output_config: dict[str, Any] | None = None


class AnthropicCountTokensRequest(BaseModel):
    """``messages.count_tokens`` has no ``max_tokens`` field."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None


# ---------------------------------------------------------------------------
# OpenAI Responses request model
# ---------------------------------------------------------------------------


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = "auto"
    text: dict[str, Any] | None = None
    parallel_tool_calls: bool = True
    background: bool = False
    store: bool = True
    previous_response_id: str | None = None
    request_id: str | None = None
    stop: str | list[str] | None = None


# ---------------------------------------------------------------------------
# Anthropic -> chat translation
# ---------------------------------------------------------------------------


def _anthropic_text(blocks: str | list[dict[str, Any]] | None) -> str:
    if not blocks:
        return ""
    if isinstance(blocks, str):
        if blocks.startswith("x-anthropic-billing-header"):
            return ""
        return blocks
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, BaseModel):
            block = block.model_dump()
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            if block["text"].startswith("x-anthropic-billing-header"):
                continue
            parts.append(block["text"])
    return "".join(parts)


def _image_url(source: dict[str, Any] | None) -> str:
    if not source:
        return ""
    if source.get("type") == "url":
        return str(source.get("url", ""))
    media_type = source.get("media_type", "image/jpeg")
    return f"data:{media_type};base64,{source.get('data', '')}"


def _anthropic_tool_result(block: AnthropicContentBlock) -> list[dict[str, Any]]:
    """Convert a tool_result block into tool + optional image messages."""

    text = ""
    image_urls: list[str] = []
    references: list[dict[str, Any]] = []
    content = block.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "text":
                parts.append(item.get("text", ""))
            elif kind == "image":
                url = _image_url(item.get("source"))
                if url:
                    image_urls.append(url)
            elif kind == "tool_reference":
                name = item.get("tool_name") or item.get("name")
                if name:
                    references.append({"type": "tool_reference", "name": name})
        text = "\n".join(parts)

    result: list[dict[str, Any]] = [
        {
            "role": "tool",
            "tool_call_id": block.tool_use_id or "",
            "content": text,
        }
    ]
    for url in image_urls:
        result.append(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            }
        )
    if references:
        # Our chat templates cannot render structured tool_reference parts;
        # keep the hint as text so the information is not silently dropped.
        names = ", ".join(str(item.get("name")) for item in references)
        result[0]["content"] = (
            str(result[0]["content"]) + f"\nTool reference: {names}"
        )
    return result


def _anthropic_tool_choice(choice: AnthropicToolChoice | None) -> Any:
    if choice is None:
        return None
    if choice.type == "auto":
        return "auto"
    if choice.type == "any":
        return "required"
    if choice.type == "none":
        return "none"
    return {"type": "function", "function": {"name": choice.name}}


def _anthropic_response_format(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not config:
        return None
    fmt = config.get("format")
    if not isinstance(fmt, dict):
        return None
    kind = fmt.get("type")
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        schema = fmt.get("schema", fmt.get("schema_"))
        if isinstance(schema, dict):
            return {"type": "json_schema", "json_schema": {"schema": schema}}
    return None


def anthropic_to_chat(request: AnthropicMessagesRequest) -> Any:
    """Translate an Anthropic Messages request into ``ChatCompletionRequest``."""

    chat_messages: list[dict[str, Any]] = []
    system_text = _anthropic_text(request.system)
    tools: list[dict[str, Any]] = []

    for message in request.messages:
        if message.role == "system":
            inline = _anthropic_text(message.content)
            if inline:
                system_text += inline
            continue

        content_parts: list[dict[str, Any]] = []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning: list[str] = []
        tool_results: list[dict[str, Any]] = []

        blocks = (
            [AnthropicContentBlock(type="text", text=message.content)]
            if isinstance(message.content, str)
            else message.content
        )
        for block in blocks:
            if block.type == "text" and block.text:
                text_parts.append(block.text)
                content_parts.append({"type": "text", "text": block.text})
            elif block.type == "image" and block.source:
                url = _image_url(block.source)
                if url:
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
            elif block.type == "thinking" and block.thinking is not None:
                reasoning.append(block.thinking)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id or f"toolu_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": orjson.dumps(block.input or {}).decode(
                                "utf-8"
                            ),
                        },
                    }
                )
            elif block.type == "tool_result":
                tool_results.extend(_anthropic_tool_result(block))
            # redacted_thinking is accepted but carries no usable text.

        chat_message: dict[str, Any] = {"role": message.role}
        if text_parts and len(content_parts) == len(text_parts):
            chat_message["content"] = "".join(text_parts)
        elif content_parts:
            chat_message["content"] = content_parts
        elif tool_calls:
            chat_message["content"] = ""
        if tool_calls:
            chat_message["tool_calls"] = tool_calls
        if reasoning:
            chat_message["reasoning"] = "".join(reasoning)
        if "content" in chat_message or tool_calls:
            chat_messages.append(chat_message)
        chat_messages.extend(tool_results)

    if system_text:
        chat_messages.insert(0, {"role": "system", "content": system_text})

    for tool in request.tools or []:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
        )

    stream = bool(getattr(request, "stream", False))
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": chat_messages,
        "max_tokens": getattr(request, "max_tokens", None) or 1,
        "stream": stream,
        "stop": getattr(request, "stop_sequences", None),
        "tools": tools or None,
        "tool_choice": _anthropic_tool_choice(request.tool_choice),
        "response_format": _anthropic_response_format(
            getattr(request, "output_config", None)
        ),
    }
    output_config = getattr(request, "output_config", None)
    effort = output_config.get("effort") if output_config else None
    if effort is not None:
        payload["reasoning_effort"] = effort
        payload["open_thinking"] = effort != "none"
    for field in ("temperature", "top_p", "top_k"):
        value = getattr(request, field, None)
        if value is not None:
            payload[field] = value
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return _chat_request().model_validate(payload)


def _builtin_tools(request: ResponsesRequest) -> list[dict[str, Any]]:
    """Tool declarations that need an executor instead of model-side calling."""

    return [
        tool
        for tool in request.tools
        if isinstance(tool, dict) and tool.get("type") not in (None, "function")
    ]


# ---------------------------------------------------------------------------
# Responses -> chat translation
# ---------------------------------------------------------------------------


def _responses_tool_choice(choice: Any) -> Any:
    if choice is None:
        return None
    if choice == "auto":
        return "auto"
    if choice == "none":
        return "none"
    if choice == "required":
        return "required"
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "function" and isinstance(choice.get("name"), str):
            return {"type": "function", "function": {"name": choice["name"]}}
        if kind == "allowed_tools":
            mode = choice.get("mode", "auto")
            if mode == "auto":
                return "auto"
            if mode == "none":
                return "none"
            named = [
                item.get("name")
                for item in choice.get("tools", [])
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
            if mode == "required" and len(named) == 1:
                return {"type": "function", "function": {"name": named[0]}}
    raise ValueError(f"unsupported Responses tool_choice {choice!r}")


def _responses_response_format(text: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(text, dict):
        return None
    fmt = text.get("format")
    if not isinstance(fmt, dict):
        return None
    if fmt.get("type") == "json_object":
        return {"type": "json_object"}
    if fmt.get("type") == "json_schema":
        schema = fmt.get("schema", fmt.get("schema_"))
        if isinstance(schema, dict):
            return {"type": "json_schema", "json_schema": {"schema": schema}}
    return None


def _item_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def _responses_input_messages(
    items: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(items, str):
        return [{"role": "user", "content": items}] if items else []
    messages: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("type", "message")
        role = item.get("role")
        if kind == "function_call":
            tool_call = {
                "id": item.get("call_id") or f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments") or "{}",
                },
            }
            previous = messages[-1] if messages else None
            if previous and previous.get("role") == "assistant":
                previous.setdefault("tool_calls", []).append(tool_call)
            else:
                messages.append(
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]}
                )
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": _item_text(item.get("output")),
                }
            )
        elif kind == "reasoning":
            messages.append(
                {"role": "assistant", "content": "", "reasoning": _item_text(item.get("content"))}
            )
        elif role in ("user", "assistant", "system", "tool") or kind == "message":
            text = _item_text(item.get("content"))
            message_role = role or "user"
            message: dict[str, Any] = {"role": message_role, "content": text}
            if message_role == "tool" and item.get("call_id"):
                message["tool_call_id"] = item["call_id"]
            messages.append(message)
    return messages


def responses_to_chat(
    request: ResponsesRequest,
    previous_response: dict[str, Any] | None = None,
) -> Any:
    """Translate an OpenAI Responses request into ``ChatCompletionRequest``."""

    messages: list[dict[str, Any]] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})
    if previous_response is not None:
        # Per the spec, prior instructions do NOT carry over; only the
        # assistant's output message text is prepended as conversation history.
        for item in previous_response.get("output") or []:
            if isinstance(item, dict) and item.get("type") == "message":
                text = _item_text(item.get("content"))
                if text:
                    messages.append({"role": "assistant", "content": text})
    messages.extend(_responses_input_messages(request.input))

    tools: list[dict[str, Any]] = []
    for tool in request.tools:
        if tool.get("type") != "function":
            continue
        function: dict[str, Any] = {
            "name": tool.get("name", ""),
            "parameters": tool.get("parameters")
            or tool.get("input_schema")
            or {"type": "object", "properties": {}},
        }
        if tool.get("description") is not None:
            function["description"] = tool["description"]
        tools.append({"type": "function", "function": function})

    tool_choice = request.tool_choice
    if not tools and tool_choice not in (None, "none"):
        tool_choice = None

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_output_tokens,
        "stream": request.stream,
        "stop": request.stop,
        "tools": tools or None,
        "tool_choice": _responses_tool_choice(tool_choice),
        "response_format": _responses_response_format(request.text),
        "cancel_id": request.request_id,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stream:
        payload["stream_options"] = {"include_usage": True}
    return _chat_request().model_validate(payload)


# ---------------------------------------------------------------------------
# Chat response -> native response bodies
# ---------------------------------------------------------------------------

_STOP_TO_ANTHROPIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _chat_message(body: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    choices = body.get("choices") or []
    if not choices:
        return {}, None
    choice = choices[0]
    return choice.get("message") or {}, choice.get("finish_reason")


def _tool_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = orjson.loads(arguments)
        except orjson.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def chat_to_anthropic(
    body: dict[str, Any], *, model: str, request_id: str | None = None
) -> dict[str, Any]:
    """Convert a non-streaming chat completion body to Anthropic's shape."""

    message, finish_reason = _chat_message(body)
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content.append(
            {
                "type": "thinking",
                "thinking": reasoning,
                "signature": uuid.uuid4().hex,
            }
        )
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex}",
                "name": function.get("name", ""),
                "input": _tool_input(function.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})

    usage = body.get("usage") or {}
    return {
        "id": request_id or body.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": body.get("model") or model,
        "stop_reason": _STOP_TO_ANTHROPIC.get(finish_reason or ""),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _response_usage(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def _response_skeleton(
    request: ResponsesRequest,
    *,
    model: str,
    output: list[dict[str, Any]],
    status: str,
    usage: dict[str, Any] | None,
    created_at: int | None = None,
) -> dict[str, Any]:
    tools = list(request.tools)
    tool_choice = request.tool_choice
    if tools and tool_choice is None:
        tool_choice = "auto"
    return {
        "id": request.request_id or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": created_at or int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": request.parallel_tool_calls,
        "temperature": request.temperature or 0.0,
        "top_p": request.top_p or 1.0,
        "tool_choice": tool_choice,
        "tools": tools,
        "background": request.background,
        "max_output_tokens": request.max_output_tokens or 256,
        "service_tier": "auto",
        "truncation": "disabled",
        "instructions": request.instructions,
        "previous_response_id": request.previous_response_id,
        "usage": usage,
    }


def _response_output_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        output.append(
            {
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex}",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": reasoning}],
                "status": "completed",
            }
        )
    if message.get("content"):
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": message["content"],
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": call.get("id") or f"call_{index}",
                "name": function.get("name", ""),
                "namespace": None,
                "arguments": function.get("arguments") or "{}",
                "status": "completed",
            }
        )
    return output


def chat_to_responses(
    body: dict[str, Any],
    *,
    request: ResponsesRequest,
    model: str,
    extra_output_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert a non-streaming chat completion body to Responses' shape."""

    message, _ = _chat_message(body)
    output = list(extra_output_items or ()) + _response_output_items(message)
    return _response_skeleton(
        request,
        model=model,
        output=output,
        status="completed",
        usage=_response_usage(body),
        created_at=body.get("created"),
    )


# ---------------------------------------------------------------------------
# Streaming conversion helpers
# ---------------------------------------------------------------------------


def _event_frame(event_type: str, payload: dict[str, Any]) -> bytes:
    data = orjson.dumps(payload).decode("utf-8")
    return f"event: {event_type}\ndata: {data}\n\n".encode()


def _parse_openai_frame(raw: bytes) -> dict[str, Any] | None:
    if not raw.startswith(b"data:"):
        return None
    text = raw[5:].decode("utf-8", errors="replace").strip()
    if not text or text == "[DONE]":
        return None
    try:
        payload = orjson.loads(text)
    except orjson.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _usage_from_chunk(chunk: dict[str, Any], current: dict[str, int]) -> None:
    usage = chunk.get("usage")
    if isinstance(usage, dict) and usage:
        if "prompt_tokens" in usage:
            current["input_tokens"] = usage["prompt_tokens"]
        if "completion_tokens" in usage:
            current["output_tokens"] = usage["completion_tokens"]


async def stream_chat_to_anthropic(
    source: AsyncIterator[bytes],
    *,
    model: str,
    request_id: str | None = None,
) -> AsyncIterator[bytes]:
    """Translate OpenAI chat SSE frames into Anthropic Messages SSE events."""

    message_id = request_id or f"msg_{uuid.uuid4().hex}"
    started = False
    block_kind: str | None = None
    block_index = 0
    tool_index_to_id: dict[int, str] = {}
    finish_reason: str | None = None
    usage = {"input_tokens": 0, "output_tokens": 0}

    def start_block(block: dict[str, Any]) -> bytes:
        nonlocal block_kind
        block_kind = block["type"]
        return _event_frame(
            "content_block_start",
            {"type": "content_block_start", "index": block_index, "content_block": block},
        )

    def stop_block() -> list[bytes]:
        nonlocal block_kind, block_index
        if block_kind is None:
            return []
        frames: list[bytes] = []
        if block_kind == "thinking":
            frames.append(
                _event_frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "signature_delta",
                            "signature": uuid.uuid4().hex,
                        },
                    },
                )
            )
        frames.append(
            _event_frame(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_index},
            )
        )
        block_kind = None
        block_index += 1
        return frames

    async for raw in source:
        chunk = _parse_openai_frame(raw)
        if chunk is None:
            continue
        if not started:
            started = True
            yield _event_frame(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": chunk.get("model") or model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        _usage_from_chunk(chunk, usage)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if choices[0].get("finish_reason"):
            finish_reason = choices[0]["finish_reason"]

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            if block_kind != "thinking":
                for frame in stop_block():
                    yield frame
                yield start_block({"type": "thinking", "thinking": ""})
            yield _event_frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning},
                },
            )

        content = delta.get("content")
        if content:
            if block_kind != "text":
                for frame in stop_block():
                    yield frame
                yield start_block({"type": "text", "text": ""})
            yield _event_frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": content},
                },
            )

        for call in delta.get("tool_calls") or []:
            index = call.get("index", 0)
            call_id = call.get("id")
            function = call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments") or ""
            if call_id:
                tool_index_to_id[index] = call_id
                if name is not None:
                    for frame in stop_block():
                        yield frame
                    yield start_block(
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": {},
                        }
                    )
            if arguments and block_kind == "tool_use":
                yield _event_frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        },
                    },
                )

    if started:
        for frame in stop_block():
            yield frame
        delta: dict[str, Any] = {
            "stop_reason": _STOP_TO_ANTHROPIC.get(finish_reason or "stop", "end_turn")
        }
        yield _event_frame(
            "message_delta",
            {"type": "message_delta", "delta": delta, "usage": usage},
        )
        yield _event_frame("message_stop", {"type": "message_stop"})


def _incrementing_event(event_type: str, state: dict[str, Any], **fields: Any) -> bytes:
    payload = {
        "type": event_type,
        "sequence_number": state["sequence"],
        **fields,
    }
    state["sequence"] += 1
    return _event_frame(event_type, payload)


async def stream_chat_to_responses(
    source: AsyncIterator[bytes],
    *,
    request: ResponsesRequest,
    model: str,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
    extra_output_items: list[dict[str, Any]] | None = None,
) -> AsyncIterator[bytes]:
    """Translate OpenAI chat SSE frames into OpenAI Responses SSE events."""

    if request.request_id is None:
        request = request.model_copy(
            update={"request_id": f"resp_{uuid.uuid4().hex}"}
        )
    state: dict[str, Any] = {"sequence": 0}
    started = False
    extra_emitted = False
    output_index = 0
    block_kind: str | None = None
    item_id = ""
    part: dict[str, Any] = {}
    items: list[dict[str, Any]] = list(extra_output_items or ())
    tool_index_to_id: dict[int, str] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}

    def close_block() -> list[bytes]:
        nonlocal block_kind, output_index
        if block_kind is None:
            return []
        frames: list[bytes] = []
        if block_kind == "message":
            frames.append(
                _incrementing_event(
                    "response.output_text.done",
                    state,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=0,
                    text=part.get("text", ""),
                )
            )
            frames.append(
                _incrementing_event(
                    "response.content_part.done",
                    state,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=0,
                    part=part,
                )
            )
        elif block_kind == "reasoning":
            frames.append(
                _incrementing_event(
                    "response.reasoning_text.done",
                    state,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=0,
                    text=part.get("text", ""),
                )
            )
            frames.append(
                _incrementing_event(
                    "response.reasoning_part.done",
                    state,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=0,
                    part=part,
                )
            )
        elif block_kind == "function_call":
            frames.append(
                _incrementing_event(
                    "response.function_call_arguments.done",
                    state,
                    item_id=item_id,
                    output_index=output_index,
                    arguments=part.get("arguments", ""),
                )
            )
        frames.append(
            _incrementing_event(
                "response.output_item.done",
                state,
                output_index=output_index,
                item=items[-1],
            )
        )
        block_kind = None
        output_index += 1
        return frames

    async for raw in source:
        chunk = _parse_openai_frame(raw)
        if chunk is None:
            continue
        if not started:
            started = True
            skeleton = _response_skeleton(
                request, model=model, output=[], status="queued", usage=None
            )
            yield _incrementing_event("response.created", state, response=skeleton)
            skeleton["status"] = "in_progress"
            yield _incrementing_event("response.in_progress", state, response=skeleton)
        if extra_output_items and not extra_emitted:
            extra_emitted = True
            for item in extra_output_items:
                yield _incrementing_event(
                    "response.output_item.added",
                    state,
                    output_index=output_index,
                    item=item,
                )
                yield _incrementing_event(
                    "response.output_item.done",
                    state,
                    output_index=output_index,
                    item=item,
                )
                output_index += 1
        _usage_from_chunk(chunk, usage)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            if block_kind != "reasoning":
                for frame in close_block():
                    yield frame
                item_id = f"rs_{uuid.uuid4().hex}"
                item = {
                    "type": "reasoning",
                    "id": item_id,
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": ""}],
                    "status": "in_progress",
                }
                items.append(item)
                yield _incrementing_event(
                    "response.output_item.added",
                    state,
                    output_index=output_index,
                    item=item,
                )
                part = item["content"][0]
                yield _incrementing_event(
                    "response.reasoning_part.added",
                    state,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=0,
                    part=part,
                )
                block_kind = "reasoning"
            part["text"] = part.get("text", "") + reasoning
            yield _incrementing_event(
                "response.reasoning_text.delta",
                state,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                delta=reasoning,
            )

        content = delta.get("content")
        if content:
            if block_kind != "message":
                for frame in close_block():
                    yield frame
                item_id = f"msg_{uuid.uuid4().hex}"
                item = {
                    "type": "message",
                    "id": item_id,
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [
                        {"type": "output_text", "text": "", "annotations": []}
                    ],
                }
                items.append(item)
                yield _incrementing_event(
                    "response.output_item.added",
                    state,
                    output_index=output_index,
                    item=item,
                )
                part = item["content"][0]
                yield _incrementing_event(
                    "response.content_part.added",
                    state,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=0,
                    part=part,
                )
                block_kind = "message"
            part["text"] = part.get("text", "") + content
            yield _incrementing_event(
                "response.output_text.delta",
                state,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                delta=content,
            )

        for call in delta.get("tool_calls") or []:
            index = call.get("index", 0)
            call_id = call.get("id")
            function = call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments") or ""
            if call_id:
                tool_index_to_id[index] = call_id
                if name is not None:
                    for frame in close_block():
                        yield frame
                    item_id = f"fc_{uuid.uuid4().hex}"
                    item = {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": name,
                        "namespace": None,
                        "arguments": "",
                        "status": "in_progress",
                    }
                    items.append(item)
                    yield _incrementing_event(
                        "response.output_item.added",
                        state,
                        output_index=output_index,
                        item=item,
                    )
                    part = item
                    block_kind = "function_call"
            if arguments and block_kind == "function_call":
                part["arguments"] = part.get("arguments", "") + arguments
                yield _incrementing_event(
                    "response.function_call_arguments.delta",
                    state,
                    item_id=item_id,
                    output_index=output_index,
                    delta=arguments,
                )

    if started:
        for frame in close_block():
            yield frame
        for item in items:
            item["status"] = "completed"
        response_usage = {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        response = _response_skeleton(
            request,
            model=model,
            output=items,
            status="completed",
            usage=response_usage,
        )
        if on_complete is not None:
            on_complete(response)
        yield _incrementing_event("response.completed", state, response=response)


# ---------------------------------------------------------------------------
# Route installation
# ---------------------------------------------------------------------------


def _anthropic_error(status_code: int, message: str, kind: str) -> MegaLMJSONResponse:
    return MegaLMJSONResponse(
        {
            "type": "error",
            "error": {"type": kind, "message": message},
        },
        status_code=status_code,
    )


def _openai_error(status_code: int, message: str, kind: str) -> MegaLMJSONResponse:
    return MegaLMJSONResponse(
        {
            "error": {
                "message": message,
                "type": kind,
                "param": None,
                "code": None,
            }
        },
        status_code=status_code,
    )


def _error_kind(exc: HTTPException) -> str:
    if exc.status_code in (400, 404, 422):
        return "invalid_request_error"
    if exc.status_code == 401:
        return "authentication_error"
    if exc.status_code == 429:
        return "rate_limit_error"
    return "api_error"


def install_agent_routes(
    app: FastAPI,
    *,
    dispatch_chat: Callable[[Any], Awaitable[Any]],
    served_model_name: str,
    tokenizer: Any = None,
    tokenizer_batcher: Any = None,
    builtin_tool_executor: BuiltinToolExecutor | None = None,
) -> None:
    """Install Anthropic and Responses routes on top of the chat dispatcher.

    ``builtin_tool_executor`` handles Responses tools such as
    ``web_search_preview`` or ``code_interpreter``, mirroring vLLM's external
    tool server. Without one, such requests are rejected instead of silently
    dropping the tool declaration.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> MegaLMJSONResponse:
        from fastapi.exception_handlers import request_validation_exception_handler

        message = "; ".join(
            str(error.get("msg", "invalid request")) for error in exc.errors()
        )
        if request.url.path.startswith("/v1/messages"):
            return _anthropic_error(400, message, "invalid_request_error")
        if request.url.path.startswith("/v1/responses"):
            return _openai_error(400, message, "invalid_request_error")
        return await request_validation_exception_handler(request, exc)

    @app.post("/v1/messages", response_model=None)
    async def anthropic_messages(request: AnthropicMessagesRequest) -> Any:
        try:
            payload = anthropic_to_chat(request)
            response = await dispatch_chat(payload)
        except HTTPException as exc:
            return _anthropic_error(
                exc.status_code, str(exc.detail), _error_kind(exc)
            )
        except ValueError as exc:
            return _anthropic_error(400, str(exc), "invalid_request_error")
        if isinstance(response, StreamingResponse):
            return StreamingResponse(
                stream_chat_to_anthropic(
                    response.body_iterator,
                    model=served_model_name,
                    request_id=f"msg_{uuid.uuid4().hex}",
                ),
                media_type="text/event-stream",
            )
        body = orjson.loads(response.body)
        return MegaLMJSONResponse(
            chat_to_anthropic(body, model=served_model_name)
        )

    @app.post("/v1/messages/count_tokens", response_model=None)
    async def anthropic_count_tokens(request: AnthropicCountTokensRequest) -> Any:
        if tokenizer is None or tokenizer_batcher is None:
            return _anthropic_error(501, "token counting unavailable", "api_error")
        try:
            payload = anthropic_to_chat(request)
            messages = tuple(
                message.model_dump(exclude_none=True)
                for message in payload.messages
            )
            rendered = tokenizer.render_chat(
                messages,
                tools=payload.tools or None,
            )
            token_ids = await tokenizer_batcher.encode(rendered)
        except (ValueError, HTTPException) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            return _anthropic_error(400, str(detail), "invalid_request_error")
        return MegaLMJSONResponse({"input_tokens": len(token_ids)})

    def _not_found(response_id: str) -> MegaLMJSONResponse:
        return _openai_error(
            404, f"Response '{response_id}' not found.", "invalid_request_error"
        )

    def _store_response(body: dict[str, Any]) -> None:
        _RESPONSE_STORE[body["id"]] = body

    @app.post("/v1/responses", response_model=None)
    async def openai_responses(request: ResponsesRequest) -> Any:
        previous_response: dict[str, Any] | None = None
        if request.previous_response_id:
            previous_response = _RESPONSE_STORE.get(request.previous_response_id)
            if previous_response is None:
                return _not_found(request.previous_response_id)
        builtins = _builtin_tools(request)
        extra_output_items: list[dict[str, Any]] = []
        if builtins:
            if builtin_tool_executor is None:
                kinds = ", ".join(
                    str(tool.get("type")) for tool in builtins
                )
                return _openai_error(
                    400,
                    f"built-in tools ({kinds}) require a tool server",
                    "invalid_request_error",
                )
            executed = builtin_tool_executor(builtins, request)
            if not executed:
                return _openai_error(
                    400,
                    "the tool server did not return a result for built-in tools",
                    "invalid_request_error",
                )
            extra_output_items = executed
        try:
            payload = responses_to_chat(request, previous_response)
            response = await dispatch_chat(payload)
        except HTTPException as exc:
            return _openai_error(exc.status_code, str(exc.detail), _error_kind(exc))
        except ValueError as exc:
            return _openai_error(400, str(exc), "invalid_request_error")
        if isinstance(response, StreamingResponse):
            captured: dict[str, Any] = {}
            frames: list[bytes] = []

            def _capture(body: dict[str, Any]) -> None:
                captured["body"] = body
                if request.store:
                    _store_response(body)

            async def _recorded() -> AsyncIterator[bytes]:
                async for frame in stream_chat_to_responses(
                    response.body_iterator,
                    request=request,
                    model=served_model_name,
                    on_complete=_capture,
                    extra_output_items=extra_output_items,
                ):
                    frames.append(frame)
                    yield frame
                if request.store and "body" in captured:
                    _RESPONSE_EVENTS[captured["body"]["id"]] = list(frames)

            return StreamingResponse(
                _recorded(), media_type="text/event-stream"
            )
        body = orjson.loads(response.body)
        native = chat_to_responses(
            body,
            request=request,
            model=served_model_name,
            extra_output_items=extra_output_items,
        )
        if request.store:
            _store_response(native)
        return MegaLMJSONResponse(native)

    @app.get("/v1/responses/{response_id}", response_model=None)
    async def retrieve_responses(
        response_id: str,
        stream: bool = False,
        starting_after: int | None = None,
    ) -> Any:
        stored = _RESPONSE_STORE.get(response_id)
        if stored is None:
            return _not_found(response_id)
        if not stream:
            return MegaLMJSONResponse(stored)

        recorded = _RESPONSE_EVENTS.get(response_id)
        if not recorded:
            recorded = [
                _event_frame(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "sequence_number": 0,
                        "response": stored,
                    },
                )
            ]

        def _sequence(frame: bytes) -> int:
            text = frame.decode("utf-8", errors="replace")
            for row in text.splitlines():
                if row.startswith("data: "):
                    try:
                        payload = orjson.loads(row[len("data: ") :])
                    except orjson.JSONDecodeError:
                        return -1
                    value = payload.get("sequence_number")
                    return value if isinstance(value, int) else -1
            return -1

        replay_frames = [
            frame
            for frame in recorded
            if starting_after is None or _sequence(frame) > starting_after
        ]

        async def replay() -> AsyncIterator[bytes]:
            for frame in replay_frames:
                yield frame

        return StreamingResponse(replay(), media_type="text/event-stream")

    @app.post("/v1/responses/{response_id}/cancel", response_model=None)
    async def cancel_responses(response_id: str) -> Any:
        stored = _RESPONSE_STORE.get(response_id)
        if stored is None:
            return _not_found(response_id)
        stored["status"] = "cancelled"
        return MegaLMJSONResponse(stored)


__all__ = [
    "AnthropicCountTokensRequest",
    "AnthropicMessagesRequest",
    "BuiltinToolExecutor",
    "ResponsesRequest",
    "anthropic_to_chat",
    "chat_to_anthropic",
    "chat_to_responses",
    "install_agent_routes",
    "responses_to_chat",
    "stream_chat_to_anthropic",
    "stream_chat_to_responses",
]
