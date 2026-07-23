"""Guard public research claims against invalid or stale benchmark numbers."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SURFACES = (
    ROOT / "README.md",
    ROOT / "docs" / "RESULTS.md",
    ROOT / "demo" / "index.html",
    ROOT / "demo" / "slides_he.html",
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


def test_removed_claims_do_not_reappear() -> None:
    for path in PUBLIC_SURFACES:
        text = path.read_text(encoding="utf-8")
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in text, f"{fragment!r} reappeared in {path.relative_to(ROOT)}"


def test_public_surfaces_link_the_methodology() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/METHODOLOGY.md" in readme
    assert "docs/RESULTS.md" in readme


def test_embible_public_numbers_match_generated_report() -> None:
    report = json.loads(
        (ROOT / "analysis" / "reports" / "embible_dss_benchmark.json").read_text(
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
