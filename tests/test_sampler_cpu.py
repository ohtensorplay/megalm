"""CPU-verified semantics for the TensorPlay fallback sampling path.

The Triton fast path requires CUDA; these tests pin the provider-independent
semantics (penalties, filters, seeding, logprobs) on the pure-tensorplay
fallback so contract drift between the engine and the sampler is caught on
machines without GPUs.
"""

import pytest
import tensorplay

from megalm.contracts.request import SamplingParams
from megalm.runtime.sampler import Sampler, SamplingError


@pytest.fixture
def sampler() -> Sampler:
    return Sampler(use_triton=False)


def test_greedy_returns_argmax_and_inactive_rows_stay_negative(
    sampler: Sampler,
) -> None:
    logits = tensorplay.tensor(
        [[1.0, 5.0, 2.0], [9.0, 1.0, 2.0]],
        dtype=tensorplay.float32,
    )
    tokens = sampler.sample(
        logits,
        (SamplingParams(), None),
        ("a", "b"),
        ((), ()),
        history_counts=((), ()),
    )
    assert tokens.tolist() == [1, -1]


def test_repetition_penalty_matches_defined_semantics(
    sampler: Sampler,
) -> None:
    logits = tensorplay.tensor([[10.0, 9.0, 0.0]], dtype=tensorplay.float32)
    tokens = sampler.sample(
        logits,
        (SamplingParams(temperature=0, repetition_penalty=2.0),),
        ("greedy",),
        ((0,),),
        history_counts=((1,),),
    )
    # 10/2 = 5 < 9, so the penalized argmax moves to token 1.
    assert tokens.tolist() == [1]


def test_frequency_presence_penalty_uses_history_counts(
    sampler: Sampler,
) -> None:
    logits = tensorplay.tensor([[5.0, 4.0, 0.0]], dtype=tensorplay.float32)
    tokens = sampler.sample(
        logits,
        (
            SamplingParams(
                temperature=0,
                frequency_penalty=2.0,
                presence_penalty=1.0,
            ),
        ),
        ("freq",),
        ((0,),),
        history_counts=((2,),),
    )
    # 5 - (1 + 2*2) = 0 < 4, so token 1 wins.
    assert tokens.tolist() == [1]


def test_mismatched_history_counts_are_rejected(sampler: Sampler) -> None:
    logits = tensorplay.tensor([[1.0, 2.0]], dtype=tensorplay.float32)
    with pytest.raises(SamplingError):
        sampler.sample(
            logits,
            (SamplingParams(temperature=0, frequency_penalty=1.0),),
            ("a",),
            ((0, 1),),
            history_counts=((1,),),
        )


def test_seeded_sampling_is_deterministic(sampler: Sampler) -> None:
    logits = tensorplay.rand(2, 64, dtype=tensorplay.float32)
    params = SamplingParams(temperature=1.0, seed=42)
    first = sampler.sample(
        logits,
        (params, params),
        ("x", "y"),
        ((), ()),
        history_counts=((), ()),
    ).tolist()
    second = sampler.sample(
        logits,
        (params, params),
        ("x", "y"),
        ((), ()),
        history_counts=((), ()),
    ).tolist()
    assert first == second
    assert all(token >= 0 for token in first)


def test_top_k_one_matches_greedy(sampler: Sampler) -> None:
    tensorplay.manual_seed(7)
    logits = tensorplay.rand(1, 32, dtype=tensorplay.float32)
    greedy = sampler.sample(
        logits,
        (SamplingParams(),),
        ("a",),
        ((),),
        history_counts=((),),
    ).tolist()
    topk = sampler.sample(
        logits,
        (SamplingParams(temperature=1.0, top_k=1),),
        ("a",),
        ((),),
        history_counts=((),),
    ).tolist()
    assert topk == greedy


def test_sample_indices_are_accepted_on_fallback(sampler: Sampler) -> None:
    logits = tensorplay.tensor([[3.0, 1.0]], dtype=tensorplay.float32)
    indices = tensorplay.tensor([1], dtype=tensorplay.int64)
    tokens = sampler.sample(
        logits,
        (SamplingParams(),),
        ("a",),
        ((),),
        history_counts=((),),
        sample_indices=indices,
    )
    assert tokens.tolist() == [0]


def test_selected_logprob_matches_greedy_distribution(
    sampler: Sampler,
) -> None:
    logits = tensorplay.tensor(
        [[1.0, 5.0, 2.0], [9.0, 1.0, 2.0]],
        dtype=tensorplay.float32,
    )
    tokens, logprobs = sampler.sample(
        logits,
        (SamplingParams(), None),
        ("a", "b"),
        ((), ()),
        history_counts=((), ()),
        return_logprobs=True,
    )
    expected = tensorplay.log_softmax(logits[0], dim=-1)[int(tokens[0])]
    assert int(tokens[0]) == 1
    assert abs(float(logprobs[0]) - float(expected)) < 1e-6
    assert float(logprobs[1]) == 0.0


def test_selected_logprob_applies_temperature(sampler: Sampler) -> None:
    tensorplay.manual_seed(3)
    logits = tensorplay.rand(1, 48, dtype=tensorplay.float32)
    tokens, logprobs = sampler.sample(
        logits,
        (SamplingParams(temperature=0.5, seed=9),),
        ("x",),
        ((),),
        history_counts=((),),
        return_logprobs=True,
    )
    expected = tensorplay.log_softmax(logits[0] / 0.5, dim=-1)[int(tokens[0])]
    assert abs(float(logprobs[0]) - float(expected)) < 1e-6


def test_selected_logprob_applies_penalties(sampler: Sampler) -> None:
    logits = tensorplay.tensor([[10.0, 9.0, 0.0]], dtype=tensorplay.float32)
    tokens, logprobs = sampler.sample(
        logits,
        (SamplingParams(temperature=0, repetition_penalty=2.0),),
        ("g",),
        ((0,),),
        history_counts=((1,),),
        return_logprobs=True,
    )
    penalized = logits[0].clone()
    penalized[0] = 10.0 / 2.0
    expected = tensorplay.log_softmax(penalized, dim=-1)[int(tokens[0])]
    assert int(tokens[0]) == 1
    assert abs(float(logprobs[0]) - float(expected)) < 1e-6


def test_selected_logprob_applies_frequency_counts(
    sampler: Sampler,
) -> None:
    logits = tensorplay.tensor([[5.0, 4.0, 0.0]], dtype=tensorplay.float32)
    tokens, logprobs = sampler.sample(
        logits,
        (
            SamplingParams(
                temperature=0,
                frequency_penalty=2.0,
                presence_penalty=1.0,
            ),
        ),
        ("f",),
        ((0,),),
        history_counts=((2,),),
        return_logprobs=True,
    )
    adjusted = logits[0].clone()
    adjusted[0] = 5.0 - (1.0 + 2.0 * 2.0)
    expected = tensorplay.log_softmax(adjusted, dim=-1)[int(tokens[0])]
    assert int(tokens[0]) == 1
    assert abs(float(logprobs[0]) - float(expected)) < 1e-6


def test_no_active_rows_returns_negative_tokens_and_zero_logprobs(
    sampler: Sampler,
) -> None:
    logits = tensorplay.tensor([[1.0, 2.0]], dtype=tensorplay.float32)
    tokens, logprobs = sampler.sample(
        logits,
        (None,),
        ("a",),
        ((),),
        history_counts=((),),
        return_logprobs=True,
    )
    assert tokens.tolist() == [-1]
    assert logprobs.tolist() == [0.0]


def test_top_logprobs_are_sorted_and_use_adjusted_distribution(
    sampler: Sampler,
) -> None:
    logits = tensorplay.tensor([[1.0, 5.0, 2.0]], dtype=tensorplay.float32)
    params = SamplingParams(temperature=1.0, logprobs=True, top_logprobs=3)
    rows = sampler.top_logprobs(logits, (params,), ("a",), ((),), 3)
    row = rows[0]
    assert row is not None
    assert len(row) == 3
    assert row[0][0] == 1
    values = [value for _, value in row]
    assert values == sorted(values, reverse=True)
    sampled, logprobs = sampler.sample(
        logits,
        (params,),
        ("a",),
        ((),),
        history_counts=((),),
        return_logprobs=True,
    )
    assert abs(float(logprobs[0]) - dict(row)[int(sampled[0])]) < 1e-6


def test_top_logprobs_require_logprobs() -> None:
    from megalm.contracts.request import RequestContractError

    with pytest.raises(RequestContractError, match="requires logprobs"):
        SamplingParams(top_logprobs=2)
