"""Unit tests for CandidateGenerator interface, LengthEnsembleCharMLMGenerator, and PartialLetterFilter."""

import pytest
from tuning.candidate_generator import (
    Candidate,
    PartialLetterFilter,
    MockCandidateGenerator,
    LengthEnsembleCharMLMGenerator,
)


def test_partial_letter_filter():
    # Compatible cases
    assert PartialLetterFilter.is_compatible("סרכיך", "סר⬚⬚ך") is True
    assert PartialLetterFilter.is_compatible("אמר", "⬚מ⬚") is True
    assert PartialLetterFilter.is_compatible("שלום", "שלום") is True

    # Incompatible cases: length mismatch
    assert PartialLetterFilter.is_compatible("שלום", "ש⬚ם") is False

    # Incompatible cases: character mismatch
    assert PartialLetterFilter.is_compatible("סרכיך", "סר⬚⬚ם") is False


def test_mock_candidate_generator_length_filter():
    gen = MockCandidateGenerator(mock_candidates=["אמר", "דבר", "צוה", "ישראל"])

    # Target length 3 -> "אמר", "דבר", "צוה"
    cands_3 = gen.generate_candidates("context", "right", target_len=3)
    assert len(cands_3) == 3
    assert [c.text for c in cands_3] == ["אמר", "דבר", "צוה"]

    # Target length 5 -> "ישראל"
    cands_5 = gen.generate_candidates("context", "right", target_len=5)
    assert len(cands_5) == 1
    assert cands_5[0].text == "ישראל"


def test_mock_candidate_generator_partial_letters():
    gen = MockCandidateGenerator(mock_candidates=["סרכיך", "סרמםך", "אמרתי"])
    cands = gen.generate_candidates("context", "right", partial_pattern="סר⬚⬚ך")
    assert len(cands) == 2
    assert [c.text for c in cands] == ["סרכיך", "סרמםך"]


def test_length_ensemble_generator_unknown_length():
    base_gen = MockCandidateGenerator(
        mock_candidates=["אמר", "אל משה", "כי בלב", "מורה צדק"]
    )
    ensemble_gen = LengthEnsembleCharMLMGenerator(
        base_gen, min_len=2, max_len=10, length_penalty_power=0.5
    )

    # Unknown length query (target_len=None)
    cands = ensemble_gen.generate_candidates("context_left", "context_right", target_len=None, top_k=5)
    assert len(cands) > 0
    # Candidate texts from different length hypotheses (3, 6, 9 chars)
    texts = [c.text for c in cands]
    assert "אמר" in texts
    assert "אל משה" in texts
    assert "כי בלב" in texts
    assert "מורה צדק" in texts
