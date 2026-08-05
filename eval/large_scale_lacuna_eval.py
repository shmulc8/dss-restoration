"""Large-Scale Real Lacuna Evaluation Harness for Text-Fabric Corpus.

Evaluates DSS restoration models on the full 27,814 physical scroll lacunae
and 12,971 partial-letter damaged words recorded in data/derived/nonbib_lacunae.jsonl.
"""

import json
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

from utils.splits import get_scroll_sets

logger = logging.getLogger(__name__)
LACUNAE_PATH = Path("data/derived/nonbib_lacunae.jsonl")


def analyze_large_scale_lacunae(path: Path = LACUNAE_PATH) -> Dict[str, Any]:
    """Compute large-scale corpus statistics over all 27,814 physical scroll lacunae."""
    if not path.exists():
        raise FileNotFoundError(f"Lacunae dataset not found: {path}")

    scroll_sets = get_scroll_sets()
    train_scrolls = scroll_sets["train"]
    val_scrolls = scroll_sets["val"]
    test_scrolls = scroll_sets["test"]

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    total_lacunae = len(records)
    split_counts = Counter()
    word_len_dist = Counter()
    char_len_dist = Counter()
    partial_trace_count = 0
    total_damaged_word_tokens = 0

    for rec in records:
        scroll = rec.get("scroll")
        if scroll in train_scrolls:
            split_counts["train"] += 1
        elif scroll in val_scrolls:
            split_counts["val"] += 1
        elif scroll in test_scrolls:
            split_counts["test"] += 1
        else:
            split_counts["other"] += 1

        gap_words = rec.get("gap_word_count_estimate") or 1
        word_len_dist[gap_words] += 1

        patterns = rec.get("visible_patterns") or []
        total_damaged_word_tokens += len(patterns)
        for pat in patterns:
            has_letters = any(c not in {"?", "⬚"} for c in pat)
            if has_letters:
                partial_trace_count += 1

    partial_trace_rate = (partial_trace_count / total_damaged_word_tokens) if total_damaged_word_tokens else 0.0

    stats = {
        "total_physical_lacunae": total_lacunae,
        "total_damaged_words": total_damaged_word_tokens,
        "partial_letter_trace_words": partial_trace_count,
        "partial_letter_trace_rate": partial_trace_rate,
        "split_breakdown": dict(split_counts),
        "gap_word_length_distribution": {
            "1_word": word_len_dist[1],
            "2_words": word_len_dist[2],
            "3_words": word_len_dist[3],
            "4_plus_words": sum(v for k, v in word_len_dist.items() if k >= 4),
        },
    }

    return stats


if __name__ == "__main__":
    results = analyze_large_scale_lacunae()
    print(json.dumps(results, indent=2))
