"""Unit test for large-scale physical lacuna evaluation harness."""

import pytest
from experiments.lacuna_corpus_stats import evaluate_test_lacunae_accuracy


def test_evaluate_test_lacunae_accuracy():
    stats = evaluate_test_lacunae_accuracy()
    assert stats["total_test_lacunae"] == 3695
    assert stats["test_split_scrolls"] == 93
    assert stats["test_partial_trace_words"] > 0
    assert "models_eval" in stats
