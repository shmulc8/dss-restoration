#!/usr/bin/env python3
"""Freeze a descriptive eligibility audit against a second DSS pipeline.

The external checkout is intentionally supplied at run time. The checked-in
output records its Git revision and input hashes without embedding an
identity-bearing repository URL in the anonymous paper artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "curation/derived/preserved_nonbib_manifest.json"
CHUNKS = ROOT / "curation/derived/preserved_nonbib_chunks.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments/results/paper/corpus_pipeline_comparison_20260810.json"
EXTERNAL_DATA = Path("data_preparation/dss_sentences_min10_splits_ppp_count_1_nonbib.xlsx")
EXTERNAL_PIPELINE = Path("amir_style_data_curation/dss_marker_pipeline.ipynb")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data_path = args.external_repo / EXTERNAL_DATA
    pipeline_path = args.external_repo / EXTERNAL_PIPELINE
    frame = pd.read_excel(data_path)
    required = {"scroll", "num_words", "num_ppp", "split"}
    if not required <= set(frame.columns):
        raise ValueError(f"External dataset is missing columns: {sorted(required - set(frame.columns))}")

    external_scrolls = set(frame["scroll"].dropna().astype(str))
    primary_scrolls = {
        json.loads(line)["scroll"]
        for line in CHUNKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    flagged = manifest["extreme_fragmentation_scrolls"]
    zero_scrolls = {
        scroll
        for scroll, row in flagged.items()
        if row["zero_preserved_words"]
    }
    extreme_scrolls = set(flagged)
    revision = subprocess.run(
        ["git", "-C", str(args.external_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = {
        "status": "descriptive_external_pipeline_audit_no_model_scores",
        "external_source": {
            "anonymous_label": "independently_implemented_contiguous_run_pipeline",
            "git_revision": revision,
            "dataset_path": str(EXTERNAL_DATA),
            "dataset_sha256": sha256(data_path),
            "pipeline_path": str(EXTERNAL_PIPELINE),
            "pipeline_sha256": sha256(pipeline_path),
            "configuration": {
                "non_biblical_only": True,
                "minimum_contiguous_written_words": 10,
                "partially_reconstructed_words_included": True,
                "minimum_authentic_consonants_for_partial_word": 1,
            },
        },
        "strict_reconstruction_free_pipeline": {
            "archival_scrolls": manifest["counts"]["eligibility"]["archival_registry_scrolls"],
            "model_contributing_scrolls": len(primary_scrolls),
            "retained_words": sum(manifest["counts"]["preserved_words_by_split"].values()),
            "reconstructed_words_admitted": 0,
        },
        "contiguous_run_pipeline": {
            "model_contributing_scrolls": len(external_scrolls),
            "retained_words": int(frame["num_words"].sum()),
            "partially_reconstructed_words": int(frame["num_ppp"].sum()),
            "partially_reconstructed_word_percent": round(
                100 * frame["num_ppp"].sum() / frame["num_words"].sum(), 6
            ),
            "examples": len(frame),
        },
        "overlap": {
            "model_scrolls_in_both": len(primary_scrolls & external_scrolls),
            "strict_only_model_scrolls": len(primary_scrolls - external_scrolls),
            "contiguous_run_only_model_scrolls": len(external_scrolls - primary_scrolls),
            "zero_preserved_scrolls_in_contiguous_run_model": len(zero_scrolls & external_scrolls),
            "extreme_fragmentation_scrolls_in_contiguous_run_model": len(extreme_scrolls & external_scrolls),
            "extreme_fragmentation_scroll_names_in_contiguous_run_model": sorted(
                extreme_scrolls & external_scrolls
            ),
        },
        "interpretation": (
            "Neither pipeline deletes scrolls by a global intactness ratio. Eligibility "
            "emerges from local example construction; the strict pipeline preserves all "
            "identifiers for provenance while admitting only reconstruction-free labels."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
