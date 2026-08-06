"""Guard public research claims against invalid or stale benchmark numbers."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SURFACES = (
    ROOT / "README.md",
    ROOT / "docs" / "RESULTS.md",
    ROOT / "docs" / "PAPER_PRESENTATION.html",
    ROOT / "docs" / "PAPER_MASTER_SUMMARY.md",
    ROOT / "demo" / "index.html",
    ROOT / "demo" / "slides_he.html",
)

# Surfaces whose numbers must be traceable to a generated artifact. The demo
# pages present a curated subset and are exempt from the numeric guards below.
EVIDENCE_SURFACES = (
    ROOT / "docs" / "RESULTS.md",
    ROOT / "docs" / "PAPER_PRESENTATION.html",
    ROOT / "docs" / "PAPER_MASTER_SUMMARY.md",
)

# These fragments identify the removed gold-length/slot-count aggregation.
# They are kept out of current public surfaces even when accompanied by a caveat.
FORBIDDEN_FRAGMENTS = (
    "Overall Top-10 Accuracy across 600",
    "600 test spans",
    "600 lacunae",
    "Single-Word RAG 48%",
    "36.8% across",
)

# Figures that represent outdated/removed claims from legacy pilot runs.
UNSUPPORTED_FIGURES = {
    "Single-Word RAG 48%": "legacy pilot claim replaced by quote-aware RAG Table A2",
    "2,179": "total text chunks; the corpus contains 1,647",
    "12,971": "damaged-word count; no such quantity exists in the dataset",
}

# Significance claims guard
UNSUPPORTED_SIGNIFICANCE = ()


def test_removed_claims_do_not_reappear() -> None:
    for path in PUBLIC_SURFACES:
        text = path.read_text(encoding="utf-8")
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in text, (
                f"{fragment!r} reappeared in {path.relative_to(ROOT)}"
            )


def test_unsupported_figures_do_not_appear() -> None:
    """No evidence surface may quote a figure that no artifact produces."""
    for path in EVIDENCE_SURFACES:
        text = path.read_text(encoding="utf-8")
        for figure, why in UNSUPPORTED_FIGURES.items():
            assert figure not in text, (
                f"{figure!r} appears in {path.relative_to(ROOT)}: {why}"
            )


def test_unsupported_significance_claims_do_not_appear() -> None:
    for path in EVIDENCE_SURFACES:
        text = path.read_text(encoding="utf-8")
        for claim in UNSUPPORTED_SIGNIFICANCE:
            assert claim not in text, (
                f"{path.relative_to(ROOT)} asserts {claim!r}; no artifact reports a "
                f"paired significance test between restoration models"
            )


def test_bootstrap_and_mcnemar_are_callable() -> None:
    """Both are cited as method; a paste that breaks them must fail the suite."""
    import pandas as pd

    from tuning.metrics_runner import _cluster_bootstrap, mcnemar_test

    df = pd.DataFrame(
        {
            "sentence_uid": ["a", "a", "b", "b", "c", "c"],
            "hit@10": [1, 0, 1, 1, 0, 0],
        }
    )
    boot = _cluster_bootstrap(df, ["hit@10"])
    assert boot is not None, "_cluster_bootstrap returned None; its body is detached"
    assert boot["hit@10"]["n"] == 6
    assert boot["hit@10"]["ci_low"] <= boot["hit@10"]["mean"] <= boot["hit@10"]["ci_high"]

    z, p = mcnemar_test([1, 1, 0, 0, 1], [0, 1, 1, 0, 0])
    assert isinstance(z, float) and isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_qd_headline_matches_generated_report() -> None:
    """The real-lacuna figures in the deck must match the scorer's own report."""
    report = json.loads(
        (ROOT / "comparison" / "reports" / "qd_researcher_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    targets = report["targets"]
    assert len(targets) == 74

    def top_k(k: int) -> float:
        hits = sum(
            1
            for t in targets
            if t["rank_any_attributed"] is not None and t["rank_any_attributed"] < k
        )
        return 100.0 * hits / len(targets)

    deck = (ROOT / "docs" / "PAPER_PRESENTATION.html").read_text(encoding="utf-8")
    for k in (1, 10, 20):
        assert f"{top_k(k):.1f}%" in deck, (
            f"QD Top-{k} = {top_k(k):.1f}% is absent from the deck"
        )

    # The benchmark ran on MsBERT; the deck must not attribute it to a character model.
    assert report["protocol"]["model"] == "ft_msbert_span_preserved_nonbib"
    assert "MsBERT FT + physical filtering" in deck


def test_deck_declares_the_split_registry_conflict() -> None:
    """40 of the 74 QD targets sit in the canonical train split; the deck must say so."""
    report = json.loads(
        (ROOT / "comparison" / "reports" / "qd_researcher_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    canonical = json.loads(
        (ROOT / "data_preparation" / "dss_scroll_splits_v1.json").read_text(
            encoding="utf-8"
        )
    )["scroll_assignment"]

    mismatched = sum(
        1 for t in report["targets"] if canonical.get(t["siglum"]) == "train"
    )
    assert mismatched == 40

    deck = (ROOT / "docs" / "PAPER_PRESENTATION.html").read_text(encoding="utf-8")
    assert f"{mismatched} of its 74 targets" in deck


def test_public_surfaces_link_the_methodology() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/METHODOLOGY.md" in readme
    assert "docs/RESULTS.md" in readme


def test_embible_public_numbers_match_generated_report() -> None:
    report = json.loads(
        (ROOT / "comparison" / "reports" / "embible_dss_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["protocol"]["modern_reconstructions_used"] is False
    assert report["protocol"]["natural_lacunae_evaluated"] is False
    assert len(report["cases"]) == 30
    expected = {
        "uwc_word": "16.7%",
        "char_unknown": "6.7%",
        "embible_overlap_ensemble": "6.7%",
        "rank_ensemble": "10.0%",
    }
    surfaces = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            ROOT / "docs" / "RESULTS.md",
            ROOT / "demo" / "index.html",
            ROOT / "demo" / "slides_he.html",
        )
    )
    for system, displayed in expected.items():
        actual = report["results"][system]["top10"]
        assert f"{actual:.1f}%" == displayed
        assert displayed in surfaces


def test_bible_transfer_public_numbers_match_generated_report() -> None:
    report = json.loads(
        (ROOT / "comparison" / "reports" / "embible_bible_transfer.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["protocol"]["bible_used_for_training"] is False
    assert report["protocol"]["test_items"] == 120
    expected = {"1": "80.0%", "2": "42.5%", "3": "27.5%"}
    surfaces = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_SURFACES)
    for word_count, displayed in expected.items():
        actual = report["results"]["by_word_count"][word_count]["uwc_word"]["top10"]
        assert f"{actual:.1f}%" == displayed
        assert displayed in surfaces
