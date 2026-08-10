"""Ensure descriptive paper statistics remain derived and score-free."""

import json
from pathlib import Path

from experiments.build_paper_data_profile import build_profile


ROOT = Path(__file__).resolve().parents[1]


def test_profile_matches_frozen_corpus_shape() -> None:
    profile = build_profile()
    corpus = profile["corpus"]
    shape = corpus["all_lacunae_shape"]
    assert (corpus["scrolls"], corpus["chunks"], corpus["preserved_words"], corpus["lacunae"]) == (
        736, 1647, 95736, 27814
    )
    assert shape["gap_word_count"]["median"] == 4
    assert shape["gap_word_count"]["buckets"]["6+"]["n"] == 9101
    assert shape["material_evidence"]["lacunae_with_at_least_one_traced_word"] == 23632


def test_profile_matches_evaluation_manifests() -> None:
    sets = build_profile()["evaluation_sets"]
    assert sets["unknown_length_synthetic"]["targets"] == 300
    assert sets["unknown_length_synthetic"]["scrolls"] == 6
    assert sets["qd_literature_agreement"]["targets"] == 74
    assert sets["qd_literature_agreement"]["unique_target_readings"] == 99


def test_generated_profile_is_current_and_contains_no_scores() -> None:
    expected = build_profile()
    generated = json.loads(
        (ROOT / "experiments/results/paper/paper_data_profile.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated == expected
    serialized = json.dumps(generated).lower()
    for forbidden in ("top1", "top5", "top10", "top20", "accuracy", "hit@"):
        assert forbidden not in serialized
