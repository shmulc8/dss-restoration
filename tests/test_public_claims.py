"""Guard the public research surfaces against reintroducing invalid headlines."""

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
