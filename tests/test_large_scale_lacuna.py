"""Unit test for large-scale physical lacuna evaluation harness."""

import pytest
from eval.large_scale_lacuna_eval import analyze_large_scale_lacunae


def test_analyze_large_scale_lacunae():
    stats = analyze_large_scale_lacunae()
    assert stats["total_physical_lacunae"] == 27814
    assert stats["split_breakdown"]["train"] > 0
    assert stats["split_breakdown"]["test"] > 0
    assert stats["partial_letter_trace_words"] > 0
