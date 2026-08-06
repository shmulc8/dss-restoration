"""Corpus statistics for physical lacunae in the Text-Fabric non-biblical corpus.

Reports extraction counts over the 27,814 physical lacunae recorded in
curation/derived/nonbib_lacunae.jsonl, split out for the canonical test scrolls.

This module computes counts only. It does not evaluate any model: scoring a
restoration model at this scale requires a decoding run, which no artifact in
this repository currently provides. Any accuracy figure quoted at the
3,695-lacuna scale would therefore be unsupported.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from data_preparation.splits import get_scroll_sets

logger = logging.getLogger(__name__)
LACUNAE_PATH = Path("curation/derived/nonbib_lacunae.jsonl")

# A damaged word position carries usable ink only if its visible pattern holds a
# character other than the unknown/blank placeholders.
BLANK_CHARS = {"?", "⬚"}


def _trace_counts(records) -> tuple[int, int, int]:
    """Return (damaged word positions, positions retaining ink, lacunae with >=1 traced word)."""
    damaged_words = 0
    trace_words = 0
    lacunae_with_trace = 0

    for rec in records:
        patterns = rec.get("visible_patterns") or []
        damaged_words += len(patterns)
        traced_here = sum(
            1 for pat in patterns if any(c not in BLANK_CHARS for c in pat)
        )
        trace_words += traced_here
        if traced_here:
            lacunae_with_trace += 1

    return damaged_words, trace_words, lacunae_with_trace


def compute_lacuna_corpus_statistics() -> Dict[str, Any]:
    """Compute physical-lacuna extraction statistics for the corpus and the test split."""
    if not LACUNAE_PATH.exists():
        raise FileNotFoundError(f"Lacunae dataset not found: {LACUNAE_PATH}")

    test_scrolls = get_scroll_sets()["test"]

    records = [
        json.loads(line)
        for line in LACUNAE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_records = [r for r in records if r.get("scroll") in test_scrolls]

    corpus_damaged, corpus_traced, corpus_lac_traced = _trace_counts(records)
    test_damaged, test_traced, test_lac_traced = _trace_counts(test_records)

    def rate(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    return {
        "corpus": {
            "lacunae": len(records),
            "damaged_word_positions": corpus_damaged,
            "damaged_words_retaining_ink": corpus_traced,
            "partial_trace_rate": rate(corpus_traced, corpus_damaged),
            "lacunae_with_traced_word": corpus_lac_traced,
            "lacunae_with_traced_word_rate": rate(corpus_lac_traced, len(records)),
        },
        "test_split": {
            "scrolls": len(test_scrolls),
            "lacunae": len(test_records),
            "damaged_word_positions": test_damaged,
            "damaged_words_retaining_ink": test_traced,
            "partial_trace_rate": rate(test_traced, test_damaged),
            "lacunae_with_traced_word": test_lac_traced,
        },
    }


if __name__ == "__main__":
    print(json.dumps(compute_lacuna_corpus_statistics(), indent=2))
