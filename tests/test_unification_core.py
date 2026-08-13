"""Unit tests for native unified masking and metrics runner modules."""

from tuning.metrics_runner import (
    normalize_he,
    normalize_hebrew_lemma,
    char_sim,
    split_candidate_words,
)
from tuning.tokenizer_compat import word_char_spans


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


def test_epigraphic_stroke_filter():
    from tuning.candidate_generator import EpigraphicStrokeFilter
    # Exact letter match = 1.0
    assert EpigraphicStrokeFilter.stroke_similarity("ר", "ר") == 1.0
    # Ambiguous stroke pair (ר vs ד) = 0.85
    assert EpigraphicStrokeFilter.stroke_similarity("ר", "ד") == 0.85
    # Wildcard = 1.0
    assert EpigraphicStrokeFilter.stroke_similarity("⬚", "ר") == 1.0
    # Compatible under stroke matrix (e.g. ארוני matches א⬚⬚ד⬚)
    assert EpigraphicStrokeFilter.is_stroke_compatible("ארוני", "א⬚⬚ד⬚") == True
    # Incompatible stroke (e.g. ארוני vs א⬚⬚ש⬚)
    assert EpigraphicStrokeFilter.is_stroke_compatible("ארוני", "א⬚⬚ש⬚") == False


def test_sectarian_idf_booster():
    from tuning.candidate_generator import SectarianIDFBooster
    # Qumran sectarian keyword boost
    assert SectarianIDFBooster.get_boost("הסרך") == 3.5
    assert SectarianIDFBooster.get_boost("בתמים") == 2.5
    # Non-sectarian common word = 0.0
    assert SectarianIDFBooster.get_boost("שלום") == 0.0
