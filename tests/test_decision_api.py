"""Frontend tests for Jev-style decision endpoints."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from megalm.contracts.request import FinishReason, SamplingParams
from megalm.gateway.decision_api import install_decision_routes
from megalm.gateway.protocol import sampling_from_wire, sampling_to_wire
from megalm.runtime.tokenizer import ModelTokenizer


class _FakeReplica:
    async def describe(self):
        return {"max_model_len": 4096}

    async def cancel(self, request_id):
        return None


class _TokenizerBatcher:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    async def encode(self, text):
        return tuple(self.tokenizer.encode(text))


class _AdmissionBatcher:
    """Return deterministic label log-probs taken from the request metadata."""

    def __init__(self):
        self.submissions = []

    async def submit(self, token_ids, *, max_new_tokens, sampling, **kwargs):
        self.submissions.append((token_ids, max_new_tokens, sampling))
        stream = SimpleNamespace(
            request_id=kwargs.get("request_id", "decision"),
            events=asyncio.Queue(),
        )
        labels = sampling.score_labels
        # Bias the first label; all values are <= 0 like log_softmax output.
        logprobs = tuple(
            -0.1 * index if index == 0 else -2.0 - 0.1 * index
            for index in range(len(labels))
        )
        stream.events.put_nowait(
            SimpleNamespace(
                token_ids=labels,
                audio=(),
                finish_reason=FinishReason.STOP,
                error=None,
                logprobs=logprobs,
            )
        )
        return stream


def _make_client():
    app = FastAPI()
    tokenizer = ModelTokenizer(Path("model"))
    admission = _AdmissionBatcher()
    install_decision_routes(
        app,
        replica=_FakeReplica(),
        tokenizer=tokenizer,
        tokenizer_batcher=_TokenizerBatcher(tokenizer),
        admission_batcher=admission,
        served_model_name="test-model",
    )
    app.state.admission = admission
    return TestClient(app)


@pytest.fixture
def client():
    with _make_client() as test_client:
        yield test_client


SYSTEMONE_BODY = {
    "model": "typesafe/jev-1.13",
    "state": {"ticket": "My card was charged twice for one order."},
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "payments and refunds",
                "shipping": "deliveries",
                "security": None,
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does the customer need a reply within the hour?",
        },
        "tone": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["calm", "annoyed", "furious"],
        },
    },
}


def test_systemone_returns_typed_answers(client):
    response = client.post("/v1/systemone", json=SYSTEMONE_BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "structured_decision"
    assert body["model"] == "test-model"
    assert set(body["answers"]) == {"team", "urgent", "tone"}
    team = body["answers"]["team"]
    assert team["type"] == "choice"
    assert team["choice"] == "billing"
    assert set(team["probabilities"]) == {"billing", "shipping", "security"}
    assert sum(team["probabilities"].values()) == pytest.approx(1.0)
    assert 0 <= team["confidence"] <= 1
    urgent = body["answers"]["urgent"]
    assert urgent["type"] == "noul"
    assert urgent["noul"] == pytest.approx(urgent["probabilities"]["yes"])
    tone = body["answers"]["tone"]
    assert tone["type"] == "score"
    assert 0 <= tone["score"] <= 2
    assert tone["legend"]["2"] == "furious"
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["input_tokens"] > 0
    assert set(body["diagnostics"]) == {"team", "urgent", "tone"}

    submissions = client.app.state.admission.submissions
    assert len(submissions) == 3
    for token_ids, max_new_tokens, sampling in submissions:
        assert max_new_tokens == 1
        assert isinstance(sampling, SamplingParams)
        assert len(sampling.score_labels) >= 2


def test_systemone_rejects_bad_requests(client):
    body = {
        "state": "x",
        "questions": {
            "q": {"type": "choice", "criteria": {"a": None}}
        },
    }
    body["questions"]["q"]["instructions"] = None
    assert client.post("/v1/systemone", json=body).status_code == 200

    too_many = {
        "state": "x",
        "questions": {
            str(index): {
                "type": "noul",
                "instructions": "is it true?",
            }
            for index in range(65)
        },
    }
    response = client.post("/v1/systemone", json=too_many)
    assert response.status_code == 400

    blank_level = {
        "state": "x",
        "questions": {
            "q": {"type": "score", "criteria": ["a", ""]}
        },
    }
    assert client.post("/v1/systemone", json=blank_level).status_code == 400

    thinking = dict(SYSTEMONE_BODY, chat_template_kwargs={"enable_thinking": True})
    assert client.post("/v1/systemone", json=thinking).status_code == 400


def test_decisions_endpoint(client):
    body = {
        "input": "The package arrived with a broken screen.",
        "questions": [
            {
                "id": "damaged",
                "type": "yes_no",
                "question": "Does the customer report a damaged item?",
            },
            {
                "id": "department",
                "type": "choice",
                "question": "Which department?",
                "choices": [
                    {"value": "returns"},
                    {"value": "billing"},
                ],
            },
            {
                "id": "severity",
                "type": "score",
                "question": "How severe?",
                "levels": [{"label": "cosmetic"}, {"label": "broken"}],
            },
        ],
    }
    response = client.post("/v1/decisions", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["object"] == "decisions"
    assert result["prompt_format_version"] == 1
    assert set(result["answers"]) == {"damaged", "department", "severity"}
    assert result["answers"]["department"]["choice"] == "returns"
    assert 0 <= result["answers"]["severity"]["score"] <= 1
    assert result["usage"]["completion_tokens"] == 0

    wrong_version = dict(body, prompt_format_version=99)
    assert client.post("/v1/decisions", json=wrong_version).status_code == 400
    duplicate = dict(body, questions=body["questions"] * 2)
    assert client.post("/v1/decisions", json=duplicate).status_code == 400


def test_score_read_round_trip_through_scheduler():
    """A score read surfaces its labels/log-probs in the completion slot."""

    from megalm.contracts.request import InferenceRequest
    from megalm.runtime.cache import KVCacheManager
    from megalm.runtime.scheduler import ContinuousBatchScheduler

    manager = KVCacheManager(
        num_pages=8,
        page_size=4,
        max_sequences=2,
        max_model_len=32,
        device="cpu",
        enable_prefix_cache=False,
    )
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=2,
        max_num_batched_tokens=32,
        prefill_chunk_size=16,
        emit_events=False,
    )
    scheduler.submit(
        InferenceRequest(
            request_id="score-1",
            model="test-model",
            input_token_ids=(1, 2, 3, 4),
            max_new_tokens=1,
            sampling=SamplingParams(score_labels=(7, 9, 11)),
        )
    )
    batch = scheduler.schedule()
    assert batch is not None
    completion = scheduler.complete_host(
        batch,
        [7],
        score_logprobs=((-0.1, -2.0, -3.0),),
    )
    assert completion.output_label_ids == ((7, 9, 11),)
    assert completion.output_label_logprobs == ((-0.1, -2.0, -3.0),)
    assert completion.output_request_ids == ("score-1",)


def test_engine_reads_raw_label_logprobs():
    """The engine read is a pure function of raw logits, with no generation."""

    import tensorplay

    from megalm.runtime.engine import Engine

    logits = tensorplay.tensor(
        [
            [1.0, 3.0, 2.0, 0.0],
            [0.5, 0.2, 4.0, 1.0],
        ],
        dtype=tensorplay.float32,
        device="cpu",
    )
    sampling = (
        SamplingParams(score_labels=(1, 2)),
        None,
    )
    rows, values = Engine._read_score_logprobs(None, logits, sampling)
    assert rows == (0,)
    host = values.tolist()[0]
    assert len(host) == 2
    assert host[0] > host[1]  # label 1 has the larger raw logit
    assert all(value <= 0 for value in host)


def test_score_labels_wire_round_trip():
    sampling = SamplingParams(temperature=0.7, logprobs=True, score_labels=(7, 9, 11))
    assert sampling_from_wire(sampling_to_wire(sampling)) == sampling


def test_label_check_rejects_multi_token_label():
    from megalm.gateway.decision_api import _label_token_ids

    class _Tokenizer:
        def encode(self, text):
            # Pretend every suffix merges into the previous token.
            return (1, 2, 3) if text.endswith("A") else (1, 2)

    with pytest.raises(ValueError, match="single token"):
        _label_token_ids(_Tokenizer(), "prompt", ("A", "B"), "q")
