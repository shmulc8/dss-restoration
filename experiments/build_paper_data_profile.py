#!/usr/bin/env python3
"""Build deterministic descriptive statistics used before model results.

This report contains corpus and evaluation-set shape only. It deliberately
contains no model scores, so descriptive evidence cannot be mistaken for a
restoration benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LACUNAE = ROOT / "curation/derived/nonbib_lacunae.jsonl"
MANIFEST = ROOT / "curation/derived/preserved_nonbib_manifest.json"
QD = ROOT / "experiments/results/paper/qd_msbert_rerun_20260810.json"
SPANS = ROOT / "experiments/results/paper/span_baselines_rerun_20260810.json"
DEFAULT_OUTPUT = ROOT / "experiments/results/paper/paper_data_profile.json"
BLANK_CHARS = {"?", "⬚"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_rank(values: list[int], probability: float) -> int:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def rate(count: int, total: int) -> float:
    return 100 * count / total if total else 0.0


def word_count_buckets(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    counts = Counter()
    for record in records:
        words = int(record["gap_word_count_estimate"])
        bucket = "1" if words == 1 else "2" if words == 2 else "3" if words == 3 else "4-5" if words <= 5 else "6+"
        counts[bucket] += 1
    return {
        bucket: {"n": counts[bucket], "percent": rate(counts[bucket], len(records))}
        for bucket in ("1", "2", "3", "4-5", "6+")
    }


def corpus_scope(records: list[dict[str, Any]]) -> dict[str, Any]:
    word_counts = [int(record["gap_word_count_estimate"]) for record in records]
    extents = [
        int(record["missing_char_count_estimate"])
        for record in records
        if record["missing_char_count_estimate"] is not None
    ]
    trace_letters = [
        sum(
            character not in BLANK_CHARS
            for pattern in record["visible_patterns"]
            for character in pattern
        )
        for record in records
    ]
    damaged_positions = sum(len(record["visible_patterns"]) for record in records)
    traced_positions = sum(
        any(character not in BLANK_CHARS for character in pattern)
        for record in records
        for pattern in record["visible_patterns"]
    )
    both_contexts = sum(
        bool(record["left_context"]) and bool(record["right_context"])
        for record in records
    )
    return {
        "scrolls": len({record["scroll"] for record in records}),
        "lacunae": len(records),
        "damaged_word_positions": damaged_positions,
        "gap_word_count": {
            "median": statistics.median(word_counts),
            "p90_nearest_rank": nearest_rank(word_counts, 0.90),
            "maximum": max(word_counts),
            "buckets": word_count_buckets(records),
        },
        "material_evidence": {
            "lacunae_with_at_least_one_traced_word": sum(value > 0 for value in trace_letters),
            "lacunae_with_at_least_one_traced_word_percent": rate(
                sum(value > 0 for value in trace_letters), len(records)
            ),
            "damaged_word_positions_retaining_ink": traced_positions,
            "damaged_word_positions_retaining_ink_percent": rate(
                traced_positions, damaged_positions
            ),
            "median_visible_letters_when_present": statistics.median(
                [value for value in trace_letters if value]
            ),
            "missing_character_estimate_available": len(extents),
            "missing_character_estimate_available_percent": rate(len(extents), len(records)),
            "missing_character_estimate_median": statistics.median(extents),
            "missing_character_estimate_iqr_nearest_rank": [
                nearest_rank(extents, 0.25),
                nearest_rank(extents, 0.75),
            ],
        },
        "context": {
            "both_sides_available": both_contexts,
            "both_sides_available_percent": rate(both_contexts, len(records)),
            "median_preserved_context_words": statistics.median(
                [len(record["left_context"]) + len(record["right_context"]) for record in records]
            ),
        },
    }


def build_profile() -> dict[str, Any]:
    manifest = load_json(MANIFEST)
    lacunae = load_jsonl(LACUNAE)
    qd = load_json(QD)
    spans = load_json(SPANS)
    heldout = [record for record in lacunae if record["split"] == "heldout"]

    unique_readings = Counter(
        (record["siglum"], int(record["word_id"]))
        for record in qd["unique_target_readings"]
    )
    qd_targets = qd["targets"]
    span_cases = spans["cases"]
    span_strata: dict[str, Any] = {}
    for words in (1, 2, 3):
        selected = [case for case in span_cases if len(case["gold"]) == words]
        character_lengths = [len("".join(case["gold"])) for case in selected]
        span_strata[str(words)] = {
            "n": len(selected),
            "scrolls": len({case["scroll"] for case in selected}),
            "character_length_median": statistics.median(character_lengths),
            "character_length_iqr_nearest_rank": [
                nearest_rank(character_lengths, 0.25),
                nearest_rank(character_lengths, 0.75),
            ],
            "character_length_range": [min(character_lengths), max(character_lengths)],
        }

    return {
        "status": "descriptive_statistics_only_no_model_scores",
        "corpus": {
            "source": manifest["source"],
            "scrolls": manifest["counts"]["scrolls"],
            "chunks": manifest["counts"]["chunks"],
            "preserved_words": sum(manifest["counts"]["preserved_words_by_split"].values()),
            "lacunae": manifest["counts"]["lacunae"],
            "splits": {
                split: {
                    "scrolls": len(manifest["scroll_splits"][split]),
                    "chunks": manifest["counts"]["chunks_by_split"][split],
                    "preserved_words": manifest["counts"]["preserved_words_by_split"][split],
                    "lacunae": manifest["counts"]["lacunae_by_split"][split],
                }
                for split in ("train", "dev", "heldout")
            },
            "all_lacunae_shape": corpus_scope(lacunae),
            "checkpoint_associated_heldout_shape": corpus_scope(heldout),
            "interpretation_caution": (
                "Gap word counts use consecutive source-word positions after editorial text "
                "is removed; they are transcription-derived structural estimates, not direct "
                "measurements of a single physical hole."
            ),
        },
        "evaluation_sets": {
            "unknown_length_synthetic": {
                "targets": len(span_cases),
                "scrolls": len({case["scroll"] for case in span_cases}),
                "sample_sha256": spans["protocol"]["heldout_sample_sha256"],
                "gold_length_word_count_and_boundaries_given": False,
                "strata": span_strata,
            },
            "qd_literature_agreement": {
                "targets": len(qd_targets),
                "scrolls": len({target["siglum"] for target in qd_targets}),
                "unique_target_readings": len(qd["unique_target_readings"]),
                "targets_with_multiple_readings": sum(value > 1 for value in unique_readings.values()),
                "targets_with_multiple_readings_percent": rate(
                    sum(value > 1 for value in unique_readings.values()), len(unique_readings)
                ),
                "estimated_length_median": statistics.median(
                    [target["constraint"]["estimated_length"] for target in qd_targets]
                ),
                "estimated_length_range": [
                    min(target["constraint"]["estimated_length"] for target in qd_targets),
                    max(target["constraint"]["estimated_length"] for target in qd_targets),
                ],
                "targets_anchored_at_one_or_both_edges": sum(
                    target["constraint"]["anchored_left"]
                    or target["constraint"]["anchored_right"]
                    for target in qd_targets
                ),
                "target_definition": "eligible non-biblical single-word natural lacunae",
            },
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (MANIFEST, LACUNAE, QD, SPANS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    profile = build_profile()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": profile["status"],
        "corpus": {
            key: profile["corpus"][key]
            for key in ("scrolls", "chunks", "preserved_words", "lacunae")
        },
        "lacuna_shape": profile["corpus"]["all_lacunae_shape"],
        "evaluation_sets": profile["evaluation_sets"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
