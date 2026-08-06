"""Unit tests for native unified masking and metrics runner modules."""

from eval.metrics_runner import (
    normalize_he,
    normalize_hebrew_lemma,
    char_sim,
    split_candidate_words,
)
from utils.tokenizer_compat import word_char_spans


def test_hebrew_normalization():
    # Normalizes spaces and diacritics
    assert normalize_he("  אמר  ") == "אמר"
    assert normalize_he("אמר") == "אמר"


def test_normalize_hebrew_lemma():
    # Normalizes plene/defective Qumran spellings and final letter forms
    assert normalize_hebrew_lemma("לוא") == "לא"
    assert normalize_hebrew_lemma("כיא") == "כי"
    assert normalize_hebrew_lemma("זואת") == "זאת"
    assert normalize_hebrew_lemma("סרכיך") == "סרכיכ"


def test_char_sim():
    # Exact match = 1.0
    assert char_sim("אמר", "אמר") == 1.0
    # Completely disjoint letters = 0.0
    assert char_sim("אמר", "גדה") == 0.0
    # 1 shared letter out of 3 = 0.333...
    assert 0.3 < char_sim("אמר", "דבר") < 0.4


def test_split_candidate_words():
    # WordPiece token joining
    assert split_candidate_words(["ויאמר", "##ו", "משה"]) == ["ויאמרו", "משה"]
    assert split_candidate_words(["א", "ב", "ג"]) == ["א", "ב", "ג"]


def test_word_char_spans():
    text = "ויאמר משה אל"
    spans = word_char_spans(text)
    assert len(spans) == 3
    assert spans[0] == (0, 5)
    assert spans[1] == (6, 9)
    assert spans[2] == (10, 12)
