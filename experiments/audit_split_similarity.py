#!/usr/bin/env python3
"""Audit composition and formulaic overlap in the frozen preserved-DSS split.

This is a descriptive leakage audit.  It does not alter the split or score a
model.  Chunk similarity is exact Jaccard overlap over preserved five-word
shingles; anonymous gap markers are discarded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curation.preserved_corpus import GAP_TOKEN, load_chunks

MANIFEST = ROOT / "curation/derived/preserved_nonbib_manifest.json"
SOURCE_CSV = ROOT / "dss_chunks.csv"
DEFAULT_OUTPUT = ROOT / "experiments/results/paper/split_similarity_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def composition_map() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["bib"] == "nonbib" and row["composition"].strip():
                mapping[row["book"]].add(row["composition"].strip())
    return mapping


def shingles(text: str, width: int = 5) -> set[tuple[str, ...]]:
    words = [word for word in text.split() if word != GAP_TOKEN]
    return {tuple(words[index : index + width]) for index in range(len(words) - width + 1)}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * fraction)]


def audit() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assignment = {
        scroll: split
        for split, scrolls in manifest["scroll_splits"].items()
        for scroll in scrolls
    }
    compositions = composition_map()
    development_compositions = {
        composition
        for scroll, split in assignment.items()
        if split in {"train", "dev"}
        for composition in compositions.get(scroll, set())
    }
    heldout_scrolls = set(manifest["scroll_splits"]["heldout"])
    composition_unseen_scrolls = {
        scroll
        for scroll in heldout_scrolls
        if compositions.get(scroll)
        and compositions[scroll].isdisjoint(development_compositions)
    }

    reference_rows = [*load_chunks("train"), *load_chunks("dev")]
    heldout_rows = load_chunks("heldout")
    reference = [(row, shingles(row["text"])) for row in reference_rows]
    heldout = [(row, shingles(row["text"])) for row in heldout_rows]
    exact_reference_texts = {
        " ".join(word for word in row["text"].split() if word != GAP_TOKEN)
        for row in reference_rows
    }
    maximum_jaccards: list[float] = []
    closest: list[dict[str, Any]] = []
    exact_duplicates = 0
    for row, row_shingles in heldout:
        normalized = " ".join(
            word for word in row["text"].split() if word != GAP_TOKEN
        )
        exact_duplicates += normalized in exact_reference_texts
        best_score = 0.0
        best_row: dict[str, Any] | None = None
        for candidate, candidate_shingles in reference:
            union = row_shingles | candidate_shingles
            score = len(row_shingles & candidate_shingles) / len(union) if union else 0.0
            if score > best_score:
                best_score = score
                best_row = candidate
        maximum_jaccards.append(best_score)
        closest.append(
            {
                "heldout_scroll": row["scroll"],
                "heldout_chunk": row["chunk_index"],
                "reference_scroll": best_row["scroll"] if best_row else None,
                "reference_split": best_row["split"] if best_row else None,
                "jaccard_5gram": best_score,
            }
        )

    composition_to_splits: dict[str, set[str]] = defaultdict(set)
    for scroll, split in assignment.items():
        for composition in compositions.get(scroll, set()):
            composition_to_splits[composition].add(split)
    return {
        "status": "descriptive_split_audit_no_model_scores",
        "definition": {
            "development_side": "train plus dev",
            "composition_unseen": "held-out scroll whose non-empty composition labels do not occur in train or dev",
            "near_duplicate": "maximum exact preserved-word 5-gram Jaccard against any train/dev chunk",
        },
        "inputs": {
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "manifest_sha256": sha256(MANIFEST),
            "composition_csv": str(SOURCE_CSV.relative_to(ROOT)),
            "composition_csv_sha256": sha256(SOURCE_CSV),
        },
        "composition": {
            "labeled_compositions": len(composition_to_splits),
            "compositions_crossing_splits": sum(
                len(splits) > 1 for splits in composition_to_splits.values()
            ),
            "heldout_scrolls": len(heldout_scrolls),
            "heldout_composition_unseen_scrolls": len(composition_unseen_scrolls),
            "heldout_composition_unseen_sigla": sorted(composition_unseen_scrolls),
        },
        "chunk_similarity": {
            "development_chunks": len(reference_rows),
            "heldout_chunks": len(heldout_rows),
            "exact_normalized_duplicates": exact_duplicates,
            "maximum_5gram_jaccard": {
                "median": statistics.median(maximum_jaccards),
                "p90": percentile(maximum_jaccards, 0.90),
                "p95": percentile(maximum_jaccards, 0.95),
                "maximum": max(maximum_jaccards, default=0.0),
                "at_least_0.5": sum(value >= 0.5 for value in maximum_jaccards),
                "at_least_0.8": sum(value >= 0.8 for value in maximum_jaccards),
            },
            "closest_pairs": sorted(
                closest, key=lambda row: -row["jaccard_5gram"]
            )[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
