"""Engine-level constrained decoding (xgrammar) tests.

These tests exercise the real xgrammar library when it is installed and the
fail-open paths when it is absent. No GPU or model weights are required: the
grammar matcher is CPU-side and the logits mask is applied to a small CPU
tensor, which is the same code path used for device tensors apart from the
Triton launch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from megalm.contracts.request import (
    RequestContractError,
    SamplingParams,
    StructuredOutput,
)
from megalm.kernels.grammar_mask import apply_grammar_bitmask
from megalm.runtime import structured as structured_module
from megalm.runtime.structured import (
    GrammarManager,
    resolve_structured_output,
)
from megalm.runtime.tokenizer import ModelTokenizer

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city", "days"],
    "additionalProperties": False,
}


def _manager() -> GrammarManager:
    tokenizer = ModelTokenizer(Path("model"))
    return GrammarManager(tokenizer, vocab_size=201024)


class _NoWarnings:
    """Minimal logger stub for ``log_event`` (which calls ``logger.log``)."""

    def __init__(self) -> None:
        self.events: list[tuple[int, str, dict]] = []

    def log(self, level: int, event: str, *, extra: dict | None = None):
        self.events.append((level, event, extra or {}))


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_structured_output_validates_and_canonicalizes_schema():
    first = StructuredOutput("json_schema", json.dumps(SCHEMA, sort_keys=True))
    second = StructuredOutput("json_schema", json.dumps(SCHEMA))
    assert first.schema_json == second.schema_json
    assert SamplingParams(structured_output=first).structured_output == first


def test_structured_output_rejects_bad_inputs():
    with pytest.raises(RequestContractError):
        StructuredOutput("regex")
    with pytest.raises(RequestContractError):
        StructuredOutput("json_schema")
    with pytest.raises(RequestContractError):
        StructuredOutput("json_schema", schema_json="not json")
    with pytest.raises(RequestContractError):
        StructuredOutput("json_object", schema_json="{}")


# ---------------------------------------------------------------------------
# Grammar manager
# ---------------------------------------------------------------------------


def test_manager_compiles_and_caches_grammars():
    manager = _manager()
    assert manager.available
    first = manager.begin("r1", StructuredOutput("json_schema", json.dumps(SCHEMA)))
    assert first is not None
    assert manager.get("r1") is first
    # Canonicalized schema text => cache hit, same compiled grammar object.
    first_compiled = manager._cache[next(iter(manager._cache))]
    manager.begin("r2", StructuredOutput("json_schema", json.dumps(SCHEMA)))
    second_compiled = manager._cache[next(iter(manager._cache))]
    assert first_compiled is second_compiled
    manager.release("r1")
    assert manager.get("r1") is None


def test_bitmask_allows_only_grammar_valid_tokens():
    manager = _manager()
    tokenizer = ModelTokenizer(Path("model"))
    backend = tokenizer._tokenizer
    grammar = manager.begin(
        "r1", StructuredOutput("json_schema", json.dumps(SCHEMA))
    )
    assert grammar is not None
    bitmask = manager.allocate_bitmask(1)
    assert grammar.fill_bitmask(bitmask, 0)
    allowed = set()
    for token_id in range(manager.vocab_size):
        if (int(bitmask[0, token_id // 32]) >> (token_id % 32)) & 1:
            allowed.add(token_id)
    # The first token of a JSON object must be `{`.
    assert backend.token_to_id("{") in allowed
    assert backend.token_to_id("hello") not in allowed


def test_accept_token_rejects_invalid_continuation():
    manager = _manager()
    tokenizer = ModelTokenizer(Path("model"))
    backend = tokenizer._tokenizer
    grammar = manager.begin("r1", StructuredOutput("json_object"))
    assert grammar is not None
    open_brace = backend.token_to_id("{")
    assert open_brace is not None and grammar.accept_token(open_brace)
    # A closing bracket cannot follow an opening object brace.
    assert not grammar.accept_token(backend.token_to_id("]"))


def test_unsupported_schema_fails_open():
    manager = _manager()
    bad = StructuredOutput(
        "json_schema",
        json.dumps({"type": "string", "pattern": "["}),
    )
    assert manager.begin("r1", bad) is None


def test_missing_xgrammar_degrades_without_raising(monkeypatch):
    monkeypatch.setattr(structured_module, "_xgrammar", None)
    tokenizer = ModelTokenizer(Path("model"))
    manager = GrammarManager(tokenizer, vocab_size=201024)
    assert not manager.available
    assert manager.begin("r1", StructuredOutput("json_object")) is None


def test_gate_skips_speculative_and_starts_matcher():
    manager = _manager()
    sampling = SamplingParams(structured_output=StructuredOutput("json_object"))
    logger = _NoWarnings()
    skipped = resolve_structured_output(
        sampling,
        manager=manager,
        request_id="r1",
        speculative_mode="ngram",
        logger=logger,
    )
    assert skipped.structured_output is None
    assert manager.get("r1") is None
    # Structured log fields go through log_event, not raw extra=kwargs.
    assert logger.events[0][1] == "constrained_decoding_skipped"
    assert logger.events[0][2] == {
        "_megalm_fields": {"request_id": "r1", "reason": "speculative_mode"}
    }

    kept = resolve_structured_output(
        sampling,
        manager=manager,
        request_id="r2",
        speculative_mode="off",
        logger=logger,
    )
    assert kept.structured_output is sampling.structured_output
    assert manager.get("r2") is not None


def test_duplicate_request_id_keeps_the_live_matcher():
    manager = _manager()
    request = StructuredOutput("json_object")
    first = manager.begin("same", request)
    assert first is not None
    # A duplicate submission must fail open instead of overwriting the live
    # request's matcher (which would silently disable its constraints).
    assert manager.begin("same", request) is None
    assert manager.get("same") is first


def test_xgrammar_exception_is_isolated_to_one_request():
    manager = _manager()
    grammar = manager.begin("r1", StructuredOutput("json_object"))
    assert grammar is not None

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated xgrammar fault")

    grammar.matcher.fill_next_token_bitmask = boom
    grammar.matcher.accept_token = boom
    bitmask = manager.allocate_bitmask(1)
    assert grammar.fill_bitmask(bitmask, 0) is False
    assert grammar.failed
    # Once failed, no further xgrammar calls happen and the request simply
    # degrades; the manager/replica stays healthy.
    assert grammar.fill_bitmask(bitmask, 0) is False
    assert grammar.accept_token(0) is False
    assert grammar.is_terminated() is False


def test_bitmask_buffer_is_reused_across_steps():
    manager = _manager()
    first = manager.bitmask_buffer(1)
    assert manager.bitmask_buffer(1) is first
    assert manager.bitmask_buffer(2) is not first
    assert first.shape[1] == (manager.vocab_size + 31) // 32


def test_warnings_deduplicate_per_reason_not_globally(caplog, monkeypatch):
    import logging

    # A fresh propagating logger avoids interference from configure_logging()
    # calls made elsewhere in the suite.
    monkeypatch.setattr(
        structured_module, "_LOGGER", logging.getLogger("test.structured")
    )
    manager = _manager()
    with caplog.at_level(logging.WARNING):
        manager._warn("compile failed for schema A")
        manager._warn("compile failed for schema A")
        manager._warn("compile failed for schema B")
    unavailable = [
        record
        for record in caplog.records
        if record.getMessage() == "constrained_decoding_unavailable"
    ]
    assert len(unavailable) == 2
    assert {record._megalm_fields["reason"] for record in unavailable} == {
        "compile failed for schema A",
        "compile failed for schema B",
    }


def test_concurrent_begin_and_release_are_state_safe():
    import threading

    manager = _manager()
    request = StructuredOutput("json_object")
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            key = f"r{index}"
            grammar = manager.begin(key, request)
            assert grammar is not None
            assert manager.get(key) is grammar
            manager.bitmask_buffer(1)
            manager.release(key)
            assert manager.get(key) is None
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert manager._matchers == {}


def test_advance_grammar_for_tokens_detects_structural_completion():
    from megalm.runtime.engine import Engine

    manager = _manager()
    mt = ModelTokenizer(Path("model"))
    backend = mt._tokenizer
    grammar = manager.begin("r1", StructuredOutput("json_object"))
    assert grammar is not None
    scheduled = SimpleNamespace(request_ids=("r1",), should_sample=(True,))
    stub = SimpleNamespace(
        _grammar=manager,
        metrics=SimpleNamespace(structured_rejected=0),
    )

    open_brace = backend.token_to_id("{")
    close_brace = backend.token_to_id("}")
    assert open_brace is not None and close_brace is not None
    for token in (open_brace, close_brace):
        done = Engine._advance_grammar_for_tokens(stub, scheduled, [token])
        assert done == frozenset()

    # After the closing brace, EOS is the only allowed token. Advancing with
    # it terminates the grammar and flags the request as structurally done.
    bitmask = manager.allocate_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)
    done = Engine._advance_grammar_for_tokens(
        stub, scheduled, [mt.eos_token_id]
    )
    assert done == frozenset({"r1"})

    # Non-sampled rows are ignored entirely.
    other = SimpleNamespace(
        request_ids=("r2",), should_sample=(False,)
    )
    assert Engine._advance_grammar_for_tokens(stub, other, [0]) == frozenset()
    assert manager.get("r2") is None


def test_grammar_stats_enrich_engine_metrics_snapshot():
    from megalm.runtime.engine import EngineMetrics

    manager = _manager()
    manager.begin("r1", StructuredOutput("json_object"))
    snapshot = EngineMetrics(_grammar=manager).snapshot()
    assert snapshot["structured_compile_failures"] == 0
    assert snapshot["structured_matcher_errors"] == 0
    assert snapshot["structured_active_matchers"] == 1
    assert snapshot["structured_bitmask_buffers"] == 0


def test_gateway_sampling_carries_structured_output():
    from megalm.gateway.chat_api import (
        ChatCompletionRequest,
        _normalized_response_format,
    )

    payload = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": {"schema": SCHEMA}},
    )
    mode, schema_json = _normalized_response_format(payload.response_format)
    structured = StructuredOutput(mode, schema_json)
    sampling = payload.sampling(structured)
    assert sampling.structured_output == structured
    assert json.loads(sampling.structured_output.schema_json) == SCHEMA


# ---------------------------------------------------------------------------
# Bitmask kernel
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import tensorplay

        return bool(tensorplay.cuda.is_available())
    except Exception:
        return False


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not _cuda_available(), reason="no CUDA"))])
def test_grammar_mask_kernel_masks_disallowed_logits(device):
    import tensorplay

    vocab = 96
    word_count = (vocab + 31) // 32
    logits = tensorplay.ones((2, vocab), dtype=tensorplay.float32, device=device)
    packed = tensorplay.zeros(
        (2, word_count), dtype=tensorplay.int32, device=device
    )
    # Row 0: only token 5 allowed; row 1: only token 70 allowed.
    packed[0, 0] = 1 << 5
    packed[1, 2] = 1 << (70 - 64)
    rows = tensorplay.tensor([0, 1], dtype=tensorplay.int32, device=device)
    apply_grammar_bitmask(logits, packed, rows, vocab_size=vocab)
    values = logits.tolist()
    assert values[0][5] == 1.0
    assert values[1][70] == 1.0
    assert all(value == float("-inf") for i, value in enumerate(values[0]) if i != 5)
    assert all(value == float("-inf") for i, value in enumerate(values[1]) if i != 70)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not _cuda_available(), reason="no CUDA"))])
def test_grammar_mask_suppresses_padded_logits(device):
    """lm_head padding beyond the grammar vocabulary is never sampleable."""

    import tensorplay

    logits_vocab, grammar_vocab = 128, 96
    logits = tensorplay.ones(
        (1, logits_vocab), dtype=tensorplay.float32, device=device
    )
    packed = tensorplay.zeros(
        (1, (grammar_vocab + 31) // 32), dtype=tensorplay.int32, device=device
    )
    packed[0, 0] = 1 << 5  # only token 5 is allowed inside the grammar
    rows = tensorplay.tensor([0], dtype=tensorplay.int32, device=device)
    apply_grammar_bitmask(
        logits,
        packed,
        rows,
        vocab_size=logits_vocab,
        grammar_vocab=grammar_vocab,
    )
    values = logits[0].tolist()
    assert values[5] == 1.0
    assert all(
        value == float("-inf") for i, value in enumerate(values) if i != 5
    )


def test_engine_mask_helper_applies_only_to_constrained_rows():
    import tensorplay

    from megalm.runtime.engine import Engine

    manager = _manager()
    grammar = manager.begin(
        "r1", StructuredOutput("json_schema", json.dumps(SCHEMA))
    )
    assert grammar is not None
    logits = tensorplay.ones(
        (2, manager.vocab_size), dtype=tensorplay.float32, device="cpu"
    )
    from megalm.runtime.engine import EngineMetrics

    stub = SimpleNamespace(_grammar=manager, metrics=EngineMetrics())
    Engine._apply_grammar_mask(stub, logits, (grammar, None))
    values = logits.tolist()
    # Row 0 was constrained: most tokens must be suppressed. Row 1 untouched.
    assert any(value == float("-inf") for value in values[0])
    assert all(value == 1.0 for value in values[1])
    assert stub.metrics.structured_mask_steps == 1
    assert stub.metrics.structured_masked_rows == 1


def test_engine_accepts_emitted_tokens_into_matcher():
    from megalm.runtime.engine import Engine
    from megalm.runtime.scheduler import Completion

    manager = _manager()
    backend = ModelTokenizer(Path("model"))._tokenizer
    grammar = manager.begin("r1", StructuredOutput("json_object"))
    assert grammar is not None
    stub = SimpleNamespace(_grammar=manager)
    completion = Completion(
        events=(),
        finished_request_ids=(),
        output_request_ids=("r1", "other"),
        output_token_ids=(backend.token_to_id("{"), 42),
        finish_reasons=(None, None),
    )
    Engine._advance_grammar(stub, completion)
    # A valid first token was accepted; an unrelated request stayed untouched.
    assert manager.get("r1") is grammar
    assert manager.get("other") is None


# ---------------------------------------------------------------------------
# End-to-end grammar-guided generation
# ---------------------------------------------------------------------------


def test_guided_generation_produces_schema_valid_json():
    """Drive masked argmax through the matcher and verify the decoded text."""

    import random

    import tensorplay

    manager = _manager()
    tokenizer = ModelTokenizer(Path("model"))
    backend = tokenizer._tokenizer
    grammar = manager.begin(
        "r1", StructuredOutput("json_schema", json.dumps(SCHEMA))
    )
    assert grammar is not None
    rng = random.Random(0)
    target = '{"city":"SF","days":3}'
    generated: list[int] = []
    position = 0
    for _ in range(256):
        bitmask = manager.allocate_bitmask(1)
        grammar.fill_bitmask(bitmask, 0)
        desired = (
            tokenizer.eos_token_id
            if position >= len(target)
            else backend.token_to_id(target[position])
        )
        assert desired is not None
        # Bias the target token strongly; the grammar mask must still be the
        # thing that keeps the generation on-schema.
        scores = [rng.random() - 1.0 for _ in range(manager.vocab_size)]
        scores[desired] = 10.0
        logits = tensorplay.tensor(
            [scores], dtype=tensorplay.float32, device="cpu"
        )
        packed = tensorplay.tensor(
            [[int(value) for value in bitmask[0]]],
            dtype=tensorplay.int32,
            device="cpu",
        )
        rows = tensorplay.tensor([0], dtype=tensorplay.int32, device="cpu")
        apply_grammar_bitmask(logits, packed, rows, vocab_size=manager.vocab_size)
        scores = logits[0].tolist()
        token_id = max(range(manager.vocab_size), key=lambda i: scores[i])
        assert token_id == desired
        if token_id == tokenizer.eos_token_id:
            break
        assert grammar.accept_token(token_id)
        generated.append(token_id)
        position += 1
    text = backend.decode(generated, skip_special_tokens=True)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    assert SCHEMA["required"][0] in parsed
