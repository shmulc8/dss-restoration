"""Tests for physical-lacuna corpus statistics."""

from experiments.lacuna_corpus_stats import compute_lacuna_corpus_statistics


def test_corpus_statistics_match_frozen_dataset():
    stats = compute_lacuna_corpus_statistics()

    corpus = stats["corpus"]
    assert corpus["lacunae"] == 27814
    assert corpus["damaged_word_positions"] == 165239
    assert corpus["damaged_words_retaining_ink"] == 49130
    assert corpus["lacunae_with_traced_word"] == 23632

    test_split = stats["test_split"]
    assert test_split["scrolls"] == 93
    assert test_split["lacunae"] == 3695
    assert test_split["damaged_word_positions"] == 22123
    assert test_split["damaged_words_retaining_ink"] == 6512


def test_partial_trace_rate_is_a_minority_of_damaged_words():
    """Most damaged words retain no ink, which bounds where the P0 regime applies."""
    stats = compute_lacuna_corpus_statistics()

    for scope in ("corpus", "test_split"):
        rate = stats[scope]["partial_trace_rate"]
        assert 0.28 < rate < 0.31, f"{scope} trace rate moved: {rate}"


def test_module_reports_no_model_accuracy():
    """Guard against a hardcoded accuracy table reappearing in this harness."""
    stats = compute_lacuna_corpus_statistics()

    flat = repr(stats)
    assert "models_eval" not in stats
    for banned in ("top1", "top10", "top20", "regime"):
        assert banned not in flat, f"{banned!r} suggests unmeasured accuracy claims returned here"
