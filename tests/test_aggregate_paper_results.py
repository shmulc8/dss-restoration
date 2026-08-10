"""Regression tests for the frozen paper-results aggregator."""

import json
from pathlib import Path

from experiments.aggregate_paper_results import topk_hit


ROOT = Path(__file__).resolve().parents[1]


def test_topk_hit_normalizes_token_list_gold() -> None:
    case = {"gold": ["two", "words"], "top10": {"system": ["two words"]}}
    assert topk_hit(case, "system", 1) == 1


def test_snapshot_cluster_estimates_equal_reported_top10() -> None:
    snapshot = json.loads(
        (ROOT / "experiments/results/paper/paper_results_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    for result in snapshot["span_systems"].values():
        assert result["top10_scroll_cluster_ci"]["estimate"] == result["top10"]
        assert result["top10_scroll_cluster_ci"]["scroll_clusters"] == 6


def test_checkpoint_rows_are_not_misreported_as_controlled_seeds() -> None:
    snapshot = json.loads(
        (ROOT / "experiments/results/paper/paper_results_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["byt5_checkpoint_replications"]["controlled_seed_set"] is False


def test_human_participant_study_is_not_in_the_active_protocol() -> None:
    protocol = json.loads(
        (ROOT / "experiments/paper_protocol_v1.json").read_text(encoding="utf-8")
    )
    track_names = {track["name"] for track in protocol["research_tracks"].values()}
    assert "blinded_scholar_assistance" not in track_names

    snapshot = json.loads(
        (ROOT / "experiments/results/paper/paper_results_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        "scholar-assistance" not in gate
        for gate in snapshot["remaining_promotion_gates"]
    )

    paper = (ROOT / "docs/paper.tex").read_text(encoding="utf-8")
    assert r"\section{Scholar-Assistance Study}" not in paper
