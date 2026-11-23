import pytest
import tensorplay

from megalm.contracts.request import SamplingParams
from megalm.kernels.topk_radix import topk_cutoffs
from megalm.runtime.sampler import Sampler


pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(),
    reason="sampling fast-path tests require CUDA",
)


def test_greedy_graph_guard_matches_sampler_semantics() -> None:
    assert Sampler.can_use_greedy_graph((SamplingParams(), SamplingParams()))
    assert not Sampler.can_use_greedy_graph(())
    assert not Sampler.can_use_greedy_graph((SamplingParams(), None))
    assert not Sampler.can_use_greedy_graph(
        (SamplingParams(temperature=0.7),)
    )
    assert not Sampler.can_use_greedy_graph(
        (SamplingParams(repetition_penalty=1.1),)
    )


def test_masked_greedy_reuses_output_and_skips_inactive_rows() -> None:
    logits = tensorplay.tensor(
        [
            [1.0, 5.0, 2.0],
            [9.0, 1.0, 2.0],
            [1.0, 2.0, 8.0],
        ],
        device="cuda",
        dtype=tensorplay.float16,
    )
    sample_indices = tensorplay.tensor([2, -1, 7], device="cuda")
    sampler = Sampler(use_triton=True)
    params = (SamplingParams(), None, SamplingParams())

    first = sampler.sample(
        logits,
        params,
        ("a", "b", "c"),
        ((), (), ()),
        sample_indices=sample_indices,
    )
    second = sampler.sample(
        logits,
        params,
        ("a", "b", "c"),
        ((), (), ()),
        sample_indices=sample_indices,
    )

    assert first.data_ptr() == second.data_ptr()
    assert second.tolist() == [1, -1, 2]


def test_radix_topk_cutoff_is_exact_for_small_and_large_k() -> None:
    tensorplay.manual_seed(11)
    scores = tensorplay.randn(4, 151936, device="cuda", dtype=tensorplay.float32)
    rows = tensorplay.arange(4, device="cuda")
    top_k = tensorplay.tensor(
        [1, 50, 2048, 151935],
        dtype=tensorplay.int32,
        device="cuda",
    )
    actual = topk_cutoffs(
        scores,
        rows,
        top_k,
        max_top_k=151935,
    )
    expected = tensorplay.stack(
        [
            tensorplay.topk(scores[row], int(top_k[row])).values[-1]
            for row in range(4)
        ]
    )
    assert tensorplay.equal(actual, expected)


def test_full_filtered_sampler_is_seeded_batch_invariant_and_exact() -> None:
    tensorplay.manual_seed(12)
    target_logits = tensorplay.randn(
        151936,
        device="cuda",
        dtype=tensorplay.float16,
    )
    sampling = SamplingParams(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        min_p=0.01,
        repetition_penalty=1.1,
        seed=123456,
    )
    history = tuple(range(256))

    single = Sampler(use_triton=True).sample(
        target_logits[None, :],
        (sampling,),
        ("target",),
        (history,),
    )
    batched_logits = tensorplay.stack(
        (tensorplay.randn_like(target_logits), target_logits)
    )
    batched = Sampler(use_triton=True).sample(
        batched_logits,
        (
            SamplingParams(temperature=0.8, top_p=0.95, seed=7),
            sampling,
        ),
        ("other", "target"),
        ((), history),
    )
    assert int(single[0]) == int(batched[1])

    scores = target_logits.float()
    seen = tensorplay.tensor(history, device="cuda")
    seen_values = scores.index_select(0, seen)
    scores.index_copy_(
        0,
        seen,
        tensorplay.where(
            seen_values < 0,
            seen_values * sampling.repetition_penalty,
            seen_values / sampling.repetition_penalty,
        ),
    )
    scores /= sampling.temperature
    weights = tensorplay.exp(scores - scores.max())
    token = int(single[0])
    cutoff = tensorplay.topk(scores, sampling.top_k).values[-1]
    assert scores[token] >= cutoff
    assert weights[token] >= sampling.min_p
    assert weights[weights > weights[token]].sum() < (
        sampling.top_p * weights.sum()
    )


def test_custom_stochastic_path_does_not_call_tensorplay_filtering_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = tensorplay.randn(
        2,
        6400,
        device="cuda",
        dtype=tensorplay.float16,
    )

    def forbidden(*args: object, **kwargs: object) -> tensorplay.Tensor:
        del args, kwargs
        raise AssertionError("TensorPlay sampling fallback was called")

    monkeypatch.setattr(tensorplay, "topk", forbidden)
    monkeypatch.setattr(tensorplay, "sort", forbidden)
    monkeypatch.setattr(tensorplay, "multinomial", forbidden)
    output = Sampler(use_triton=True).sample(
        logits,
        (
            SamplingParams(
                temperature=0.8,
                top_k=128,
                top_p=0.9,
                min_p=0.02,
                seed=1,
            ),
            SamplingParams(temperature=1.0, top_p=0.95, seed=2),
        ),
        ("a", "b"),
        (tuple(range(32)), tuple(range(16))),
    )
    assert bool(tensorplay.all(output >= 0))


def test_greedy_repetition_penalty_matches_defined_semantics() -> None:
    logits = tensorplay.tensor(
        [[10.0, 9.0, 0.0]],
        device="cuda",
        dtype=tensorplay.float16,
    )
    output = Sampler(use_triton=True).sample(
        logits,
        (
            SamplingParams(
                temperature=0,
                repetition_penalty=2.0,
            ),
        ),
        ("greedy",),
        ((0,),),
    )
    assert output.tolist() == [1]
