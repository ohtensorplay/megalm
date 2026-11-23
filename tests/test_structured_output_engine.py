"""End-to-end engine tests for constrained decoding.

These load the built-in checkpoint, so they are skipped without CUDA. They
verify that a constrained request bypasses the greedy CUDA graph, that the
grammar mask reaches the sampler, and that the generated text is valid JSON.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
import tensorplay

from megalm.contracts.request import (
    EngineError,
    OutputModality,
    SamplingParams,
    StructuredOutput,
)
from megalm.runtime.engine import Engine, EngineConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(),
    reason="first-party engine integration requires CUDA",
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def engine():
    config = EngineConfig(
        model_path=PROJECT_ROOT / "model",
        max_cache_tokens=256,
        max_model_len=256,
        max_num_sequences=2,
        max_num_batched_tokens=64,
        prefill_chunk_size=16,
        enable_cuda_graph=True,
        cuda_graph_batch_sizes=(1, 2),
    )
    instance = Engine(config)
    yield instance
    del instance
    gc.collect()
    tensorplay.cuda.empty_cache()


def test_constrained_generation_yields_valid_json_object(engine) -> None:
    assert engine._grammar.available
    results = engine.generate_chat(
        ["输出一个JSON对象"],
        max_new_tokens=48,
        sampling=SamplingParams(
            structured_output=StructuredOutput("json_object")
        ),
    )
    text = engine.tokenizer.decode(
        results[0].token_ids, skip_special_tokens=True
    ).strip()
    value = json.loads(text)
    assert isinstance(value, dict)


def test_constrained_generation_respects_json_schema(engine) -> None:
    results = engine.generate_chat(
        ["用JSON给出一个人的姓名和年龄"],
        max_new_tokens=64,
        sampling=SamplingParams(
            structured_output=StructuredOutput(
                "json_schema", json.dumps(_SCHEMA)
            )
        ),
    )
    text = engine.tokenizer.decode(
        results[0].token_ids, skip_special_tokens=True
    ).strip()
    value = json.loads(text)
    assert set(_SCHEMA["required"]).issubset(value)
    assert isinstance(value["name"], str)
    assert isinstance(value["age"], int)


def test_rejected_admission_does_not_leak_grammar_state(engine) -> None:
    """A validation failure after begin() must roll the matcher back."""

    sampling = SamplingParams(structured_output=StructuredOutput("json_object"))
    with pytest.raises(EngineError):
        engine.add_request(
            engine.tokenizer.encode_chat(
                [{"role": "user", "content": "hi"}]
            ),
            max_new_tokens=4,
            sampling=sampling,
            request_id="reject-audio",
            output_modalities=(OutputModality.AUDIO,),
        )
    assert engine._grammar.get("reject-audio") is None


def test_parallel_choices_have_independent_matchers(engine) -> None:
    token_ids = engine.tokenizer.encode_chat(
        [{"role": "user", "content": "输出一个JSON对象"}]
    )
    sampling = SamplingParams(structured_output=StructuredOutput("json_object"))
    ids = []
    for index in range(2):
        request_id = f"choice-{index}"
        ids.append(request_id)
        engine.add_request(
            token_ids,
            max_new_tokens=48,
            sampling=sampling,
            request_id=request_id,
        )
        assert engine._grammar.get(request_id) is not None
    results = engine.run_until_complete(ids)
    for result in results:
        text = engine.tokenizer.decode(
            result.token_ids, skip_special_tokens=True
        ).strip()
        assert isinstance(json.loads(text), dict)
    for request_id in ids:
        assert engine._grammar.get(request_id) is None


def test_logprobs_and_structured_output_are_compatible(engine) -> None:
    results = engine.generate_chat(
        ["输出一个JSON对象"],
        max_new_tokens=48,
        sampling=SamplingParams(
            structured_output=StructuredOutput("json_object"),
            logprobs=True,
        ),
    )
    text = engine.tokenizer.decode(
        results[0].token_ids, skip_special_tokens=True
    ).strip()
    assert isinstance(json.loads(text), dict)


def test_structured_metrics_are_recorded(engine) -> None:
    before = engine.metrics.snapshot()
    engine.generate_chat(
        ["输出一个JSON对象"],
        max_new_tokens=16,
        sampling=SamplingParams(
            structured_output=StructuredOutput("json_object")
        ),
    )
    after = engine.metrics.snapshot()
    assert after["structured_requests"] == before["structured_requests"] + 1
    assert after["structured_mask_steps"] > before["structured_mask_steps"]
    assert after["structured_masked_rows"] > before["structured_masked_rows"]
