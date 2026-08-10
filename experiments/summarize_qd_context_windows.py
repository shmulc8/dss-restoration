#!/usr/bin/env python3
"""Collect fixed-seed QD context-window sensitivity results."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/results/paper/qd_context_window_summary.json"


def main() -> None:
    paths = {
        str(window): ROOT
        / f"experiments/results/paper/qd_context_window{window}_seed42_20260811.json"
        for window in (2, 5, 10)
    }
    paths["full"] = ROOT / "experiments/results/paper/qd_methods_seed42_20260811.json"
    rows = []
    for window, path in paths.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "context_words_each_side": window,
                "context_only_top10": report["condition_results"]["context_only"]["top10"],
                "soft_visible_top10": report["condition_results"]["soft_visible"]["top10"],
                "hard_visible_top10": report["condition_results"]["visible_only"]["top10"],
                "source": str(path.relative_to(ROOT)),
            }
        )
    result = {
        "status": "fixed_seed_context_window_sensitivity",
        "seed": 42,
        "targets": 93,
        "selection": "descriptive sensitivity; no test-window selection",
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
