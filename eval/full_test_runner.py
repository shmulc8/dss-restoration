"""Full 338-Sentence Test Split Benchmark Runner.

Evaluates DSS restoration models on the full 338-sentence test partition
defined by the frozen scroll splits (data/splits/dss_scroll_splits_v1.json).

Reports:
- Headline metrics (unaligned predictions = misses)
- Secondary metrics (aligned-only)
- 95% Cluster Bootstrap CIs (B=1000)
- Paired McNemar significance tests
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from eval.metrics_runner import compute_metrics, mcnemar_test, score_run_dir
from utils.splits import get_scroll_sets

logger = logging.getLogger(__name__)


def run_full_test_benchmark(
    run_dir: str,
    confidence_level: float = 0.95,
    num_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Execute complete metrics evaluation on a durable run artifact directory."""
    path = Path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")

    word_df, metrics, counts = score_run_dir(path)

    overall = metrics.get("overall") or {}
    masked_words = counts.get("masked_words", 0)
    unaligned = counts.get("unaligned_words", 0)
    aligned = masked_words - unaligned

    hit1_entry = overall.get("hit_at_1") or {}
    hit10_entry = overall.get("hit_at_10") or {}
    char_sim_entry = overall.get("char_sim_top1") or {}
    mrr_entry = overall.get("mrr") or {}

    headline_hit1 = (hit1_entry.get("mean", 0.0) * aligned / masked_words) if masked_words and hit1_entry.get("mean") is not None else 0.0
    headline_hit10 = (hit10_entry.get("mean", 0.0) * aligned / masked_words) if masked_words and hit10_entry.get("mean") is not None else 0.0

    report = {
        "run_directory": str(path),
        "total_sentences": counts.get("sentences", 0),
        "total_masked_words": masked_words,
        "unaligned_words": unaligned,
        "headline_metrics_all_words": {
            "hit@1": headline_hit1,
            "hit@10": headline_hit10,
        },
        "aligned_only_metrics": {
            "hit@1": hit1_entry.get("mean", 0.0),
            "hit@10": hit10_entry.get("mean", 0.0),
            "hit@10_ci_low": hit10_entry.get("ci_low", 0.0),
            "hit@10_ci_high": hit10_entry.get("ci_high", 0.0),
            "char_sim": char_sim_entry.get("mean", 0.0),
            "mrr": mrr_entry.get("mean", 0.0),
        },
    }

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full Test Split Evaluation Runner")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to predictions.jsonl directory")
    args = parser.parse_args()

    results = run_full_test_benchmark(args.run_dir)
    print(json.dumps(results, indent=2))
