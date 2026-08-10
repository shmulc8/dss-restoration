"""Keep the paper's generated tables tied to checked result artifacts."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _table(manifest: dict, command: str) -> dict:
    return next(t for t in manifest["tables"] if t["command"] == command)


def test_paper_manifest_is_explicitly_below_final_promotion() -> None:
    manifest = json.loads(
        (ROOT / "docs" / "paper_results_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "shareable_reproduced_snapshot_not_final_promotion"
    assert all(table["status"] != "paper_result" for table in manifest["tables"])


def test_shared_alignment_rows_match_artifact() -> None:
    artifact = json.loads(
        (ROOT / "experiments/results/runs/summary_with_subsets.json").read_text(
            encoding="utf-8"
        )
    )["full"]
    manifest = json.loads(
        (ROOT / "docs" / "paper_results_manifest.json").read_text(encoding="utf-8")
    )
    rows = {row[0]: row for row in _table(manifest, "SyntheticAlignmentTable")["rows"]}

    tavbert = artifact["tavbert-finetuned"]
    assert rows["TavBERT FT (character)"][2] == f'{100 * tavbert["hit@10"]:.2f}\\%'

    msbert = artifact["msbert-finetuned"]
    coverage = (msbert["masked_words"] - msbert["unaligned"]) / msbert["masked_words"]
    assert rows["MsBERT FT (WordPiece)"][1] == f'{100 * msbert["hit@1"] * coverage:.2f}\\%'
    assert rows["MsBERT FT (WordPiece)"][2] == f'{100 * msbert["hit@10"] * coverage:.2f}\\%'
    assert rows["MsBERT FT (WordPiece)"][3] == f'{100 * (1 - coverage):.2f}\\%'


def test_span_pilot_headline_matches_fresh_snapshot() -> None:
    artifact = json.loads(
        (ROOT / "experiments/results/paper/span_baselines_rerun_20260810.json").read_text(
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
