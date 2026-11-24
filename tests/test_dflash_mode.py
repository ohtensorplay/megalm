"""G7 gold tests: DFlash mode draft source + greedy chain verification."""

from __future__ import annotations

import subprocess
import sys

import pytest

from megalm.runtime.dflash import (
    DFlashDraftSource,
    verify_chain,
    verify_chain_batch,
)
from megalm.runtime.speculative import build_draft_source


def test_verify_chain_accepts_full_prefix_and_emits_bonus_target() -> None:
    result = verify_chain([1, 2, 3, 4], [1, 2, 3, 9])
    assert result.accepted_count == 3
    assert result.output_token_ids == (1, 2, 3, 9)
    assert result.fully_accepted is False


def test_verify_chain_replaces_first_divergence_and_masks_tail() -> None:
    result = verify_chain([1, 2, 3, 4], [1, 7, 8, 9])
    assert result.accepted_count == 1
    assert result.output_token_ids == (1, 7, -1, -1)
    assert result.fully_accepted is False


def test_verify_chain_full_match_emits_last_target() -> None:
    result = verify_chain([1, 2, 3], [1, 2, 3])
    assert result.accepted_count == 3
    assert result.output_token_ids == (1, 2, 3)
    assert result.fully_accepted is True


def test_verify_chain_divergence_at_first_token() -> None:
    result = verify_chain([1, 2, 3], [9, 8, 7])
    assert result.accepted_count == 0
    assert result.output_token_ids == (9, -1, -1)


def test_verify_chain_empty_and_validation() -> None:
    assert verify_chain([], []).output_token_ids == ()
    with pytest.raises(ValueError, match="equal length"):
        verify_chain([1], [1, 2])


def test_verify_chain_batch() -> None:
    results = verify_chain_batch([[1, 2], [3, 4]], [[1, 9], [3, 4]])
    assert [r.accepted_count for r in results] == [1, 2]
    assert [r.output_token_ids for r in results] == [(1, 9), (3, 4)]
    with pytest.raises(ValueError, match="equal row counts"):
        verify_chain_batch([[1]], [[1], [2]])


def test_dflash_draft_source_implements_abi() -> None:
    source = DFlashDraftSource(max_ngram_size=4, min_candidate_agreement=0.0)
    # DraftSource is a structural Protocol (not runtime_checkable); assert the
    # ABI methods exist directly instead.
    for method in ("propose", "observe", "forget"):
        assert callable(getattr(source, method))
    assert source.name == "dflash"
    # A continuation observed once is proposed when no consensus is required.
    source.observe("req-a", (1, 2, 3, 4, 5, 6))
    draft = source.propose(
        "req-b",
        (1, 2, 3),
        max_draft_tokens=3,
        min_ngram_size=2,
        max_ngram_size=4,
    )
    assert draft == (4, 5, 6)
    source.forget("req-a")
    assert source.propose(
        "req-b",
        (1, 2, 3),
        max_draft_tokens=3,
        min_ngram_size=2,
        max_ngram_size=4,
    ) == ()


def test_dflash_agreement_threshold_suppresses_single_continuation() -> None:
    # With full agreement required, a single observed continuation cannot form
    # a supported candidate (support ratio 1/1 == 1.0, so it is allowed); using
    # two distinct continuations makes the ratio 0.5 and the stricter inner
    # threshold of 1.0 suppresses the proposal.
    source = DFlashDraftSource(max_ngram_size=3, min_candidate_agreement=1.0)
    source.observe("a", (1, 2, 3, 7, 7, 7))
    source.observe("b", (1, 2, 3, 8, 8, 8))
    draft = source.propose(
        "c",
        (1, 2, 3),
        max_draft_tokens=3,
        min_ngram_size=2,
        max_ngram_size=3,
    )
    assert draft == ()


def test_dflash_validates_agreement_range() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        DFlashDraftSource(min_candidate_agreement=1.5)


def test_build_draft_source_returns_dflash_without_importing_on_default() -> None:
    source = build_draft_source("dflash", max_ngram_size=4)
    assert isinstance(source, DFlashDraftSource)
    assert source.name == "dflash"
    # Default modes never touch the DFlash module.
    code = (
        "from megalm.runtime.speculative import build_draft_source;"
        "build_draft_source('off', max_ngram_size=4);"
        "build_draft_source('ngram', max_ngram_size=4);"
        "import sys;"
        "assert 'megalm.runtime.dflash' not in sys.modules;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_engine_config_accepts_dflash_mode() -> None:
    from megalm.runtime.engine import EngineConfig

    config = EngineConfig(model_path="model", speculative_mode="dflash")
    assert config.speculative_mode == "dflash"

    with pytest.raises(Exception, match="speculative_mode"):
        EngineConfig(model_path="model", speculative_mode="magic")


