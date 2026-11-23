"""Jev-style structured-decision endpoints.

Mainstream runtimes (vLLM ``/v1/systemone``, SGLang ``/v1/decisions``) answer
typed questions without generating text: one read per question renders the
shared state plus the question's labelled options through the chat template,
then reads the next-token log-probabilities of the answer labels.  MegaLM uses
the same label-read engine path (``SamplingParams.score_labels``), so this
module is only the HTTP/frontend half of that contract.

No model weights or custom checkpoint are required: any chat model whose
answer labels are single tokens at the answer position can serve decisions.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from megalm.contracts.request import SamplingParams
from megalm.gateway.replica_types import ReplicaFailure

# SGLang increments this when the prompt wording changes. Answers are pinned
# by clients sending the same value, so a wording change must bump it here.
PROMPT_FORMAT_VERSION = 1

MAX_QUESTIONS = 64
MAX_CHOICES = 26
MAX_LEVELS = 10
CHOICE_LABELS = tuple(chr(ord("A") + index) for index in range(MAX_CHOICES))
LEVEL_LABELS = tuple(str(index) for index in range(MAX_LEVELS))
NOUL_LABELS = ("yes", "no")


class DecisionFrontendError(ValueError):
    """A decision request cannot be served by the loaded chat model."""


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _check_name(name: str, what: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise DecisionFrontendError(f"{what} must be a non-empty string")
    if any(
        character in "\r\n\t" or ord(character) < 32 for character in name
    ):
        raise DecisionFrontendError(
            f"{what} must not contain control or line-break characters"
        )
    return name.strip()


class SystemOneQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "noul", "score"]
    instructions: str | None = None
    criteria: dict[str, str | None] | list[str] | None = None


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    state: str | dict[str, Any] | list[Any] | None = None
    questions: dict[str, SystemOneQuestion] = Field(min_length=1)
    chat_template_kwargs: dict[str, Any] | None = None


class DecisionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: Literal["choice", "score", "yes_no", "predicate"]
    question: str = Field(min_length=1)
    options: list[dict[str, Any]] | None = None
    choices: list[dict[str, Any]] | None = None
    levels: list[Any] | None = None
    yes: str | None = None
    no: str | None = None


class DecisionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    input: str | dict[str, Any] | list[Any] | None = None
    state: str | dict[str, Any] | list[Any] | None = None
    questions: list[DecisionItem] = Field(min_length=1)
    temperature: float = 1.0
    chat_template_kwargs: dict[str, Any] | None = None
    prompt_format_version: int | None = None
    images: list[Any] | None = None


def _check_template_kwargs(kwargs: dict[str, Any] | None) -> None:
    if not kwargs:
        return
    if "chat_template" in kwargs:
        raise DecisionFrontendError("request chat templates are not accepted")
    if "enable_thinking" in kwargs and kwargs["enable_thinking"] not in {
        False,
        None,
    }:
        raise DecisionFrontendError("decision reads require thinking disabled")


def _render_labels(
    state: str,
    instructions: str | None,
    lines: list[str],
    closing: str,
) -> str:
    parts = [state, ""]
    if instructions:
        parts.append(f"Question: {instructions}")
    parts.extend(lines)
    parts.append(closing)
    return "\n".join(parts)


def _system_one_prompt(
    state: str,
    question: SystemOneQuestion,
) -> tuple[str, tuple[str, ...]]:
    """Build the user message and its ordered answer labels."""

    instructions = question.instructions
    criteria = question.criteria
    if question.type == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise DecisionFrontendError(
                "a choice question needs at least one named option"
            )
        if len(criteria) > MAX_CHOICES:
            raise DecisionFrontendError(
                f"a choice question allows at most {MAX_CHOICES} options"
            )
        lines: list[str] = []
        options: list[str] = []
        for name, description in criteria.items():
            options.append(_check_name(name, "choice option name"))
            line = f"{CHOICE_LABELS[len(options) - 1]}: {name}"
            if description:
                line += f" - {_text(description)}"
            lines.append(line)
        lowered = {name.casefold() for name in options}
        if len(lowered) != len(options):
            raise DecisionFrontendError("choice option names must be unique")
        labels = CHOICE_LABELS[: len(options)]
        prompt = _render_labels(
            state,
            instructions,
            lines,
            "Answer with the letter of one option only.",
        )
        return prompt, labels
    if question.type == "score":
        if (
            not isinstance(criteria, list)
            or len(criteria) < 2
            or len(criteria) > MAX_LEVELS
        ):
            raise DecisionFrontendError(
                "a score question needs 2 to 10 ordered levels"
            )
        levels = [_text(level) for level in criteria]
        if any(not level.strip() for level in levels):
            raise DecisionFrontendError("score levels must not be blank")
        lines = [
            f"{LEVEL_LABELS[index]}: {level}"
            for index, level in enumerate(levels)
        ]
        prompt = _render_labels(
            state,
            instructions,
            lines,
            "Answer with the number of one level only.",
        )
        return prompt, LEVEL_LABELS[: len(levels)]
    # noul
    if criteria is not None and not isinstance(criteria, dict):
        raise DecisionFrontendError("noul criteria must be an object or null")
    descriptions = criteria or {}
    unknown = set(descriptions) - {"true", "false"}
    if unknown:
        raise DecisionFrontendError(
            f"unknown noul criteria keys: {sorted(unknown)}"
        )
    lines = []
    for label, key in (("yes", "true"), ("no", "false")):
        description = descriptions.get(key)
        if description:
            lines.append(f"{label}: {_text(description)}")
    if not instructions and not lines:
        raise DecisionFrontendError(
            "a noul question needs instructions or true/false descriptions"
        )
    prompt = _render_labels(
        state,
        instructions,
        lines,
        "Answer with one label only.",
    )
    return prompt, NOUL_LABELS


def _decision_prompt(
    state: str,
    item: DecisionItem,
) -> tuple[str, tuple[str, ...]]:
    instructions = item.question
    if item.type in {"choice", "predicate"}:
        raw = item.options if item.options is not None else item.choices
        if not raw or len(raw) > MAX_CHOICES:
            raise DecisionFrontendError(
                f"a choice needs 1 to {MAX_CHOICES} options"
            )
        lines: list[str] = []
        names: list[str] = []
        for option in raw:
            if not isinstance(option, dict) or "value" not in option:
                raise DecisionFrontendError(
                    "choice options need a string value"
                )
            value = option["value"]
            if not isinstance(value, (str, bool)):
                raise DecisionFrontendError(
                    "choice option values must be strings or booleans"
                )
            name = _check_name(str(value), "choice value")
            names.append(name)
            line = f"{CHOICE_LABELS[len(names) - 1]}: {name}"
            description = option.get("description")
            if description:
                line += f" - {_text(description)}"
            lines.append(line)
        if len({name.casefold() for name in names}) != len(names):
            raise DecisionFrontendError("choice values must be unique")
        prompt = _render_labels(
            state,
            instructions,
            lines,
            "Answer with the letter of one option only.",
        )
        return prompt, CHOICE_LABELS[: len(names)]
    if item.type == "score":
        levels = item.levels or []
        if not 2 <= len(levels) <= MAX_LEVELS:
            raise DecisionFrontendError(
                "a score question needs 2 to 10 levels"
            )
        rendered = [
            f"{level.get('label')}"
            if isinstance(level, dict) and "label" in level
            else _text(level)
            for level in levels
        ]
        if any(not level.strip() for level in rendered):
            raise DecisionFrontendError("score levels must not be blank")
        lines = [
            f"{LEVEL_LABELS[index]}: {level}"
            for index, level in enumerate(rendered)
        ]
        prompt = _render_labels(
            state,
            instructions,
            lines,
            "Answer with the number of one level only.",
        )
        return prompt, LEVEL_LABELS[: len(levels)]
    lines = []
    if item.yes:
        lines.append(f"yes: {_text(item.yes)}")
    if item.no:
        lines.append(f"no: {_text(item.no)}")
    prompt = _render_labels(
        state,
        instructions,
        lines,
        "Answer with one label only.",
    )
    return prompt, NOUL_LABELS


def _label_token_ids(
    tokenizer: Any,
    rendered: str,
    labels: tuple[str, ...],
    question_id: str,
) -> tuple[int, ...]:
    base = tokenizer.encode(rendered)
    token_ids: list[int] = []
    for label in labels:
        candidate = tokenizer.encode(rendered + label)
        if (
            len(candidate) != len(base) + 1
            or candidate[: len(base)] != base
        ):
            raise DecisionFrontendError(
                f"question {question_id!r}: answer label {label!r} is not a "
                "single token at the answer position"
            )
        token_ids.append(candidate[-1])
    if len(set(token_ids)) != len(token_ids):
        raise DecisionFrontendError(
            f"question {question_id!r}: answer labels are not distinct tokens"
        )
    return tuple(token_ids)


def _probabilities(
    logprobs: tuple[float, ...],
    temperature: float,
) -> tuple[tuple[float, ...], float]:
    if not logprobs:
        raise DecisionFrontendError("a decision read returned no label scores")
    if any(math.isnan(value) or math.isinf(value) for value in logprobs):
        raise DecisionFrontendError("a decision read returned invalid scores")
    scaled = [value / temperature for value in logprobs]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    probabilities = tuple(value / total for value in weights)
    label_mass = min(1.0, sum(math.exp(value) for value in logprobs))
    return probabilities, label_mass


async def _read(
    *,
    tokenizer: Any,
    admission_batcher: Any,
    prompt: str,
    labels: tuple[str, ...],
    temperature: float,
    request_id: str,
    max_model_len: int,
) -> tuple[tuple[float, ...], float, int]:
    """Render, submit and reduce one decision read."""

    rendered = tokenizer.render_chat(
        [{"role": "user", "content": prompt}],
        open_thinking=False,
    )
    token_ids = _label_token_ids(tokenizer, rendered, labels, request_id)
    base_ids = tokenizer.encode(rendered)
    if len(base_ids) + 1 > max_model_len:
        raise DecisionFrontendError(
            "decision prompt exceeds the model context length"
        )
    stream = await admission_batcher.submit(
        base_ids,
        max_new_tokens=1,
        sampling=SamplingParams(
            temperature=0.0,
            score_labels=token_ids,
        ),
        request_id=request_id,
    )
    label_logprobs: tuple[float, ...] = ()
    for _ in range(16):
        item = await stream.events.get()
        if isinstance(item, ReplicaFailure):
            raise RuntimeError("decision read failed") from item.error  # noqa: TRY004
        if item.token_ids:
            if item.token_ids != token_ids:
                raise DecisionFrontendError(
                    "decision read returned unexpected label ids"
                )
            label_logprobs += tuple(item.logprobs)
        if item.finish_reason is not None:
            break
    else:
        raise RuntimeError("decision read did not terminate")
    if len(label_logprobs) != len(labels):
        raise DecisionFrontendError(
            "decision read returned an incomplete label distribution"
        )
    probabilities, label_mass = _probabilities(label_logprobs, temperature)
    return probabilities, label_mass, len(base_ids)


def _system_one_answer(
    question: SystemOneQuestion,
    labels: tuple[str, ...],
    probabilities: tuple[float, ...],
    label_mass: float,
) -> dict[str, Any]:
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    confidence = probabilities[best] * label_mass
    if question.type == "choice":
        names = list((question.criteria or {}).keys())
        return {
            "type": "choice",
            "choice": names[best],
            "probabilities": {
                name: float(probabilities[index])
                for index, name in enumerate(names)
            },
            "confidence": float(confidence),
        }
    if question.type == "score":
        levels = [str(level) for level in (question.criteria or [])]
        score = sum(
            index * probability
            for index, probability in enumerate(probabilities)
        )
        return {
            "type": "score",
            "score": float(score),
            "legend": {
                str(index): level for index, level in enumerate(levels)
            },
            "probabilities": {
                str(index): float(probability)
                for index, probability in enumerate(probabilities)
            },
            "confidence": float(confidence),
        }
    yes = probabilities[0]
    return {
        "type": "noul",
        "noul": float(yes),
        "probabilities": {
            "yes": float(yes),
            "no": float(probabilities[1]),
        },
        "confidence": float(confidence),
    }


def _decision_answer(
    item: DecisionItem,
    labels: tuple[str, ...],
    probabilities: tuple[float, ...],
    label_mass: float,
) -> dict[str, Any]:
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    answer: dict[str, Any] = {
        "type": item.type,
        "label_mass": float(label_mass),
    }
    if item.type in {"choice", "predicate"}:
        raw = item.options if item.options is not None else item.choices
        values = [str(option.get("value")) for option in raw or []]
        answer["choice"] = values[best]
        answer["probabilities"] = {
            value: float(probability)
            for value, probability in zip(values, probabilities)
        }
    elif item.type == "score":
        answer["probabilities"] = {
            str(index): float(probability)
            for index, probability in enumerate(probabilities)
        }
    else:
        answer["probabilities"] = {
            label: float(probability)
            for label, probability in zip(NOUL_LABELS, probabilities)
        }
    if item.type == "score":
        answer["score"] = float(
            sum(
                index * probability
                for index, probability in enumerate(probabilities)
            )
        )
    elif item.type not in {"choice", "predicate"}:
        answer["choice"] = labels[best]
    return answer


def install_decision_routes(
    app: FastAPI,
    *,
    replica: Any,
    tokenizer: Any,
    tokenizer_batcher: Any,
    admission_batcher: Any,
    served_model_name: str,
) -> None:
    """Mount the decision endpoints on the gateway app."""

    async def _prepare() -> int:
        descriptor = await replica.describe()
        max_model_len = int(descriptor["max_model_len"])
        return max_model_len

    @app.post("/v1/systemone")
    async def systemone(payload: SystemOneRequest) -> dict[str, Any]:
        try:
            _check_template_kwargs(payload.chat_template_kwargs)
        except DecisionFrontendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if len(payload.questions) > MAX_QUESTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"at most {MAX_QUESTIONS} questions per request",
            )
        state = _text(payload.state) if payload.state is not None else ""
        max_model_len = await _prepare()
        request_id_root = uuid4().hex
        prepared: list[tuple[str, SystemOneQuestion, str, tuple[str, ...]]] = []
        for question_id, question in payload.questions.items():
            if not question_id.strip():
                raise HTTPException(
                    status_code=400, detail="question ids must not be blank"
                )
            try:
                prompt, labels = _system_one_prompt(state, question)
                rendered = tokenizer.render_chat(
                    [{"role": "user", "content": prompt}],
                    open_thinking=False,
                )
                _label_token_ids(
                    tokenizer, rendered, labels, question_id
                )
            except DecisionFrontendError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            prepared.append((question_id, question, prompt, labels))
        try:
            reads = await asyncio.gather(
                *(
                    _read(
                        tokenizer=tokenizer,
                        admission_batcher=admission_batcher,
                        prompt=prompt,
                        labels=labels,
                        temperature=1.0,
                        request_id=f"decision-{request_id_root}-{index}",
                        max_model_len=max_model_len,
                    )
                    for index, (_, _, prompt, labels) in enumerate(prepared)
                )
            )
        except DecisionFrontendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        answers: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        input_tokens = 0
        for (question_id, question, _, labels), (
            probabilities,
            label_mass,
            prompt_tokens,
        ) in zip(prepared, reads):
            answers[question_id] = _system_one_answer(
                question, labels, probabilities, label_mass
            )
            diagnostics[question_id] = {
                "label_mass": float(label_mass),
                "argmax_is_label": True,
            }
            input_tokens += prompt_tokens
        return {
            "id": f"decision-{request_id_root}",
            "object": "structured_decision",
            "created": int(time.time()),
            "model": served_model_name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "diagnostics": diagnostics,
        }

    @app.post("/v1/decisions")
    async def decisions(payload: DecisionsRequest) -> dict[str, Any]:
        try:
            _check_template_kwargs(payload.chat_template_kwargs)
        except DecisionFrontendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if payload.prompt_format_version not in (None, PROMPT_FORMAT_VERSION):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unsupported prompt_format_version "
                    f"{payload.prompt_format_version}"
                ),
            )
        if payload.images:
            raise HTTPException(
                status_code=400,
                detail="decision image inputs are not supported",
            )
        if len(payload.questions) > MAX_QUESTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"at most {MAX_QUESTIONS} questions per request",
            )
        if not math.isfinite(payload.temperature) or payload.temperature <= 0:
            raise HTTPException(
                status_code=400, detail="temperature must be positive"
            )
        state = _text(
            payload.input if payload.input is not None else payload.state
        )
        if not state.strip():
            raise HTTPException(status_code=400, detail="input must not be blank")
        max_model_len = await _prepare()
        request_id_root = uuid4().hex
        prepared: list[tuple[DecisionItem, str, tuple[str, ...]]] = []
        seen: set[str] = set()
        for item in payload.questions:
            if item.id in seen:
                raise HTTPException(
                    status_code=400,
                    detail=f"duplicate question id {item.id!r}",
                )
            seen.add(item.id)
            try:
                prompt, labels = _decision_prompt(state, item)
                rendered = tokenizer.render_chat(
                    [{"role": "user", "content": prompt}],
                    open_thinking=False,
                )
                _label_token_ids(tokenizer, rendered, labels, item.id)
            except DecisionFrontendError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            prepared.append((item, prompt, labels))
        try:
            reads = await asyncio.gather(
                *(
                    _read(
                        tokenizer=tokenizer,
                        admission_batcher=admission_batcher,
                        prompt=prompt,
                        labels=labels,
                        temperature=payload.temperature,
                        request_id=f"decision-{request_id_root}-{index}",
                        max_model_len=max_model_len,
                    )
                    for index, (_, prompt, labels) in enumerate(prepared)
                )
            )
        except DecisionFrontendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        answers: dict[str, Any] = {}
        input_tokens = 0
        for (item, _, labels), (
            probabilities,
            label_mass,
            prompt_tokens,
        ) in zip(prepared, reads):
            answers[item.id] = _decision_answer(
                item, labels, probabilities, label_mass
            )
            input_tokens += prompt_tokens
        return {
            "object": "decisions",
            "model": served_model_name,
            "prompt_format_version": PROMPT_FORMAT_VERSION,
            "answers": answers,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": 0,
            },
        }


__all__ = [
    "PROMPT_FORMAT_VERSION",
    "DecisionFrontendError",
    "install_decision_routes",
]
