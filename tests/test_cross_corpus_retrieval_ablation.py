import numpy as np

from analysis.cross_corpus_connections import Passage
from eval.cross_corpus_retrieval_ablation import (
    candidate_support,
    contains_phrase,
    holm_adjust,
    paired_cluster_statistics,
    rerank_candidates,
    visible_query,
)
from eval.tf_embible_dss_benchmark import Item


def test_visible_query_excludes_hidden_gold():
    item = Item(
        item_id="heldout:4QX:0:8:1",
        scroll="4QX",
        left=("אמר", "הכהן"),
        gold=("סוד",),
        right=("אל", "העם"),
    )
    assert visible_query(item) == "אמר הכהן אל העם"
    assert "סוד" not in visible_query(item)


def test_contains_phrase_requires_contiguous_word_boundaries():
    tokens = ("א", "ב", "ג", "ד")
    assert contains_phrase(tokens, ("ב", "ג"))
    assert not contains_phrase(tokens, ("ב", "ד"))
    assert not contains_phrase(tokens, ("אב",))


def test_answer_removal_drops_documents_containing_complete_gold():
    documents = [
        Passage("a", "x", "A", "1", "א ב ג", ("א", "ב", "ג")),
        Passage("b", "x", "B", "1", "א ד ג", ("א", "ד", "ג")),
    ]
    ranking = [(0, 0.9), (1, 0.8)]
    assert candidate_support("ב", ranking, documents) > 0
    assert (
        candidate_support(
            "ב",
            ranking,
            documents,
            removed_phrase=("ב",),
        )
        == 0
    )


def test_alpha_zero_preserves_baseline_order():
    baseline = [("א", -1.0), ("ב", -2.0), ("ג", -3.0)]
    rows = rerank_candidates(
        baseline,
        {"ג": 1.0, "ב": 0.5},
        alpha=0.0,
    )
    assert [candidate for candidate, _ in rows] == ["א", "ב", "ג"]


def test_holm_adjust_is_monotone_in_significance_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0 <= value <= 1 for value in adjusted.values())


def test_paired_cluster_statistics_detect_identical_systems():
    baseline = [
        {"item_id": "a", "scroll": "S1", "hit_top10": True},
        {"item_id": "b", "scroll": "S2", "hit_top10": False},
    ]
    statistics = paired_cluster_statistics(
        baseline,
        list(baseline),
        iterations=100,
        seed=7,
    )
    assert statistics["delta_exact_top10_points"] == 0
    assert statistics["paired_cluster_bootstrap_95_ci"] == [0.0, 0.0]
    assert np.isclose(statistics["cluster_sign_flip_p"], 1.0)
