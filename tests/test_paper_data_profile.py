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
    eligibility = corpus["scroll_eligibility"]
    assert (
        eligibility["primary_mlm_scrolls"],
        eligibility["scrolls_without_primary_mlm_chunks"],
        eligibility["zero_preserved_word_scrolls"],
        eligibility["extreme_fragmentation_scrolls"],
    ) == (377, 359, 25, 71)
    assert eligibility["extreme_fragmentation_in_unknown_length_targets"] == 0
    assert eligibility["extreme_fragmentation_in_qd_targets"] == 0


def test_external_pipeline_audit_is_frozen_and_descriptive() -> None:
    audit = build_profile()["corpus"]["external_pipeline_audit"]
    assert audit["status"] == "descriptive_external_pipeline_audit_no_model_scores"
    assert audit["strict_reconstruction_free_pipeline"]["model_contributing_scrolls"] == 377
    assert audit["contiguous_run_pipeline"]["model_contributing_scrolls"] == 254
    assert audit["contiguous_run_pipeline"]["partially_reconstructed_words"] == 3655
    assert audit["overlap"]["model_scrolls_in_both"] == 223
    assert audit["overlap"]["extreme_fragmentation_scroll_names_in_contiguous_run_model"] == [
        "4Q457a",
        "4Q472a",
    ]


def test_profile_matches_evaluation_manifests() -> None:
    sets = build_profile()["evaluation_sets"]
    assert sets["unknown_length_synthetic"]["targets"] == 300
    assert sets["unknown_length_synthetic"]["scrolls"] == 79
    assert sets["qd_literature_agreement"]["targets"] == 93
    assert sets["qd_literature_agreement"]["unique_target_readings"] == 184


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
