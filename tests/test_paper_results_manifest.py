"""Keep the paper's generated tables tied to checked result artifacts."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _table(manifest: dict, command: str) -> dict:
    return next(t for t in manifest["tables"] if t["command"] == command)


def test_paper_manifest_is_the_team_review_snapshot() -> None:
    manifest = json.loads(
        (ROOT / "docs" / "paper_results_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "team_review_evidence_snapshot"
    assert all(table["status"] != "paper_result" for table in manifest["tables"])


def test_shared_alignment_claims_match_artifact() -> None:
    artifact = json.loads(
        (ROOT / "experiments/results/runs/summary_with_subsets.json").read_text(
            encoding="utf-8"
        )
    )["full"]
    paper = (ROOT / "docs" / "paper.tex").read_text(encoding="utf-8")
    tavbert = artifact["tavbert-base"]
    assert f'{100 * tavbert["hit@10"]:.1f}\\%' in paper
    msbert = artifact["msbert-finetuned"]
    coverage = (msbert["masked_words"] - msbert["unaligned"]) / msbert["masked_words"]
    assert f'{100 * msbert["hit@10"]:.1f}\\%' in paper
    assert f'{100 * msbert["hit@10"] * coverage:.1f}\\%' in paper
    assert str(msbert["unaligned"]) in paper


def test_span_pilot_headline_matches_fresh_snapshot() -> None:
    artifact = json.loads(
        (ROOT / "experiments/results/paper/span_balanced_300_20260811.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (ROOT / "docs" / "paper_results_manifest.json").read_text(encoding="utf-8")
    )
    rows = {row[0]: row for row in _table(manifest, "SpanPilotTable")["rows"]}
    assert rows["Preserved-only word span"][3] == f'{artifact["results"]["uwc_word"]["top10"]:.1f}\\%'
    assert artifact["results"]["by_word_count"]["3"]["uwc_word"]["top10"] == 0.0


def test_generated_tables_are_current() -> None:
    generated = ROOT / "docs" / "paper_tables.tex"
    before = generated.read_bytes()
    subprocess.run(
        [sys.executable, str(ROOT / "docs/generate_paper_tables.py")],
        check=True,
        cwd=ROOT,
    )
    assert generated.read_bytes() == before
