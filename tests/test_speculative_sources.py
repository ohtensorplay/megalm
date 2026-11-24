"""G7 gold tests: unified draft-source ABI and suffix decoding mode.

The default ``off`` path and the original ``ngram`` path must stay unchanged;
the plugin source is only used when a new mode explicitly selects one.
"""

from __future__ import annotations

import pytest

from megalm.contracts.request import InferenceRequest, SamplingParams
from megalm.runtime.cache import KVCacheManager
from megalm.runtime.engine import EngineConfig, EngineError, EngineMetrics
from megalm.runtime.scheduler import ContinuousBatchScheduler
from megalm.runtime.speculative import (
    PromptLookupDraftSource,
    SharedSuffixDraftSource,
    build_draft_source,
    prompt_lookup_draft,
)


def test_ngram_adapter_delegates_to_the_original_index() -> None:
    tokens = (1, 2, 3, 1, 2, 3, 9)
    kwargs = dict(
        max_draft_tokens=3,
        min_ngram_size=1,
        max_ngram_size=4,
        min_candidate_support=1,
    )
    expected = prompt_lookup_draft(tokens, **kwargs)
    source = PromptLookupDraftSource(max_ngram_size=4)
    assert source.propose("request", tokens, **kwargs) == expected


def test_shared_suffix_reuses_a_pattern_from_another_request() -> None:
    source = SharedSuffixDraftSource(max_ngram_size=4)
    source.observe("a", (9, 7, 5, 3, 1, 2, 3, 4, 6, 8))
    draft = source.propose(
        "b",
        (8, 5, 3, 1, 2, 3),
        max_draft_tokens=3,
        min_ngram_size=2,
        max_ngram_size=4,
        min_candidate_support=1,
    )
    # (3, 1, 2, 3) appears in request a and continues with (4, 6, 8).
    assert draft == (4, 6, 8)
    source.forget("a")
    assert (
        source.propose(
            "b",
            (8, 5, 3, 1, 2, 3),
            max_draft_tokens=3,
            min_ngram_size=2,
            max_ngram_size=4,
            min_candidate_support=1,
        )
        == ()
    )


def test_shared_suffix_respects_excluded_drafts() -> None:
    source = SharedSuffixDraftSource(max_ngram_size=4)
    source.observe("a", (3, 1, 2, 3, 4, 6, 8))
    kwargs = dict(
        max_draft_tokens=3,
        min_ngram_size=2,
        max_ngram_size=4,
        min_candidate_support=1,
    )
    assert source.propose("b", (5, 3, 1, 2, 3), **kwargs) == (4, 6, 8)
    assert (
        source.propose(
            "b", (5, 3, 1, 2, 3), excluded_drafts=((4, 6),), **kwargs
        )
        == ()
    )


def test_build_draft_source_selects_only_prompt_lookup_modes() -> None:
    assert isinstance(
        build_draft_source("ngram", max_ngram_size=4),
        PromptLookupDraftSource,
    )
    assert isinstance(
        build_draft_source("suffix", max_ngram_size=4),
        SharedSuffixDraftSource,
    )
    assert build_draft_source("off", max_ngram_size=4) is None
    assert build_draft_source("mtp", max_ngram_size=4) is None


def test_acceptance_rate_is_exposed_for_ablations() -> None:
    metrics = EngineMetrics(
        speculative_draft_tokens=8,
        speculative_accepted_draft_tokens=6,
    )
    assert metrics.snapshot()["speculative_acceptance_rate"] == pytest.approx(0.75)
    empty = EngineMetrics().snapshot()
    assert empty["speculative_acceptance_rate"] == 0.0


def _config(**overrides: object) -> EngineConfig:
    payload = dict(
        model_path="/nonexistent",
        max_num_batched_tokens=16,
        max_num_sequences=2,
    )
    payload.update(overrides)
    return EngineConfig(**payload)  # type: ignore[arg-type]


def test_suffix_mode_is_a_valid_engine_config() -> None:
    assert _config(speculative_mode="suffix").speculative_mode == "suffix"
    with pytest.raises(EngineError, match="speculative_mode"):
        _config(speculative_mode="draft_model")


def test_scheduler_accepts_a_plugin_draft_source() -> None:
    manager = KVCacheManager(
        num_pages=8,
        page_size=4,
        max_sequences=1,
        max_model_len=32,
        device="cpu",
        enable_prefix_cache=False,
    )
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=8,
        prefill_chunk_size=8,
        emit_events=False,
    )
    scheduler.submit(
        InferenceRequest(
            request_id="x",
            model="qwen3",
            input_token_ids=(1, 2, 3, 1, 2),
            max_new_tokens=6,
            sampling=SamplingParams(),
        )
    )
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete_host(prefill, [3])
    decode = scheduler.schedule()
    assert decode is not None

    source = SharedSuffixDraftSource(max_ngram_size=4)
    speculative = scheduler.prepare_ngram_speculative(
        decode,
        num_speculative_tokens=3,
        min_ngram_size=2,
        max_ngram_size=3,
        min_candidate_support=1,
        probe_interval=1,
        draft_source=source,
    )
    # The plugin source proposes from the same repeated tail as ngram.
    assert speculative.is_speculative
    assert speculative.draft_token_ids == ((1, 2),)
    assert source.name == "suffix"
