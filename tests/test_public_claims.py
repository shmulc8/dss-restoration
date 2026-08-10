"""Guard public research claims against stale or unsupported numbers."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SURFACES = (
    ROOT / "README.md",
    ROOT / "docs" / "RESULTS.md",
    ROOT / "docs" / "paper.tex",
)

FORBIDDEN_FRAGMENTS = (
    "83.78%",
    "83.78\\%",
    "82.90%",
    "82.90\\%",
    "14.5%",
    "14.5\\%",
    "physical-constraint ablation",
    "same 74",
    "74 Qumran",
)


def public_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_SURFACES)


def test_removed_claims_do_not_reappear() -> None:
    text = public_text()
    for fragment in FORBIDDEN_FRAGMENTS:
        assert fragment not in text


def test_qd_headline_matches_generated_report() -> None:
    report = json.loads(
        (ROOT / "experiments/results/paper/qd_evidence_conditions_20260811.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["targets"]) == 93
    conditions = report["condition_results"]
    text = public_text()
    for condition in ("context_only", "visible_only", "frequency_visible_only"):
        assert f'{conditions[condition]["top10"]:.1f}%' in text
    assert "oracle-assisted" in text
    assert "historical truth" in text


def test_synthetic_headline_matches_generated_report() -> None:
    report = json.loads(
        (ROOT / "experiments/results/paper/span_balanced_300_20260811.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["cases"]) == 300
    diagnostics = report["protocol"]["heldout_sample_diagnostics"]
    assert diagnostics["unique_scrolls"] == 79
    assert "non-overlapping" in diagnostics["sampling"]
    text = public_text()
    assert f'{report["results"]["uwc_word"]["top10"]:.1f}%' in text
    assert "79 held-out scrolls" in text


def test_matched_byt5_seeds_are_traceable() -> None:
    configurations = []
    scores = []
    for seed in (41, 42, 43):
        report = json.loads(
            (
                ROOT
                / f"experiments/results/paper/byt5_balanced_seed{seed}_20260811.json"
            ).read_text(encoding="utf-8")
        )
        training = report["protocol"]["training"]
        configurations.append(
            (training["epochs"], training["batch_size"], training["learning_rate"])
        )
        scores.append(report["results"]["exact_top10"])
        assert training["seed"] == seed
    assert len(set(configurations)) == 1
    assert scores == [4 / 3, 4 / 3, 1.0]


def test_split_audit_is_current() -> None:
    audit = json.loads(
        (ROOT / "experiments/results/paper/split_similarity_audit.json").read_text(
            encoding="utf-8"
        )
    )
    similarity = audit["chunk_similarity"]
    assert similarity["exact_normalized_duplicates"] == 0
    assert similarity["maximum_5gram_jaccard"]["at_least_0.5"] == 0
    assert audit["composition"]["compositions_crossing_splits"] == 26


def test_bootstrap_and_mcnemar_are_callable() -> None:
    import pandas as pd

    from tuning.metrics_runner import _cluster_bootstrap, mcnemar_test

    frame = pd.DataFrame(
        {
            "sentence_uid": ["a", "a", "b", "b", "c", "c"],
            "hit@10": [1, 0, 1, 1, 0, 0],
        }
    )
    boot = _cluster_bootstrap(frame, ["hit@10"])
    assert boot["hit@10"]["n"] == 6
    z_value, p_value = mcnemar_test([1, 1, 0, 0, 1], [0, 1, 1, 0, 0])
    assert isinstance(z_value, float)
    assert 0.0 <= p_value <= 1.0


def test_public_surfaces_link_evidence_and_methodology() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/METHODOLOGY.md" in readme
    assert "docs/RESULTS.md" in readme
