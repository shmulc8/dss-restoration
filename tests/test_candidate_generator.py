"""Unit tests for CandidateGenerator interface and PartialLetterFilter."""

import pytest
from eval.candidate_generator import (
    Candidate,
    PartialLetterFilter,
    MockCandidateGenerator,
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
