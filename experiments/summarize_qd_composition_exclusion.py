#!/usr/bin/env python3
"""Compare regular and QD-composition-excluded seed-42 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGULAR = ROOT / "experiments/results/paper/qd_methods_seed42_20260811.json"
EXCLUDED = ROOT / "experiments/results/paper/qd_composition_excluded_seed42_20260811.json"
MANIFEST = ROOT / "experiments/results/paper/qd_composition_exclusion.json"
OUTPUT = ROOT / "experiments/results/paper/qd_composition_exclusion_summary.json"
SOURCE_CSV = ROOT / "dss_chunks.csv"
CONDITIONS = ("context_only", "soft_visible", "visible_only")


def labeled_scrolls() -> set[str]:
    labels: dict[str, set[str]] = defaultdict(set)
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["bib"] == "nonbib" and row["composition"].strip():
                labels[row["book"]].add(row["composition"].strip())
    return {scroll for scroll, values in labels.items() if values}


def is_hit(row: dict, condition: str) -> int:
    rank = row[f"{condition}_rank"]
    return int(rank is not None and rank < 10)


def paired_cluster_ci(regular: list[dict], excluded: list[dict], condition: str) -> list[float]:
    by_scroll: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for left, right in zip(regular, excluded):
        if (left["siglum"], left["word_id"]) != (right["siglum"], right["word_id"]):
            raise ValueError("composition reports do not share ordered targets")
        by_scroll[str(left["siglum"])].append(
            (is_hit(left, condition), is_hit(right, condition))
        )
    scrolls = sorted(by_scroll)
    generator = random.Random(20260811)
    estimates = []
    for _ in range(10_000):
        sampled = [scrolls[generator.randrange(len(scrolls))] for _ in scrolls]
        pairs = [pair for scroll in sampled for pair in by_scroll[scroll]]
        estimates.append(100 * sum(right - left for left, right in pairs) / len(pairs))
    estimates.sort()
    return [estimates[250], estimates[9749]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regular", type=Path, default=REGULAR)
    parser.add_argument("--excluded", type=Path, default=EXCLUDED)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    regular = json.loads(args.regular.read_text(encoding="utf-8"))
    excluded = json.loads(args.excluded.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    labeled = labeled_scrolls()
    regular_rows = [row for row in regular["targets"] if str(row["siglum"]) in labeled]
    excluded_rows = [row for row in excluded["targets"] if str(row["siglum"]) in labeled]
    conditions = {}
    for condition in CONDITIONS:
        left = [is_hit(row, condition) for row in regular_rows]
        right = [is_hit(row, condition) for row in excluded_rows]
        conditions[condition] = {
            "regular_top10": 100 * sum(left) / len(left),
            "composition_excluded_top10": 100 * sum(right) / len(right),
            "delta_points": 100 * (sum(right) - sum(left)) / len(left),
            "paired_scroll_cluster_delta_95ci": paired_cluster_ci(
                regular_rows, excluded_rows, condition
            ),
        }
    result = {
        "status": "single_seed_qd_composition_exclusion_sensitivity",
        "targets_with_nonempty_composition_labels": len(regular_rows),
        "scrolls": len({str(row["siglum"]) for row in regular_rows}),
        "retained_training_chunks": manifest["retained_train_chunks"],
        "original_training_chunks": manifest["original_train_chunks"],
        "verified_composition_overlap": manifest["verified_composition_overlap"],
        "conditions": conditions,
        "limitation": "single matched seed; exclusion changes training volume as well as composition exposure",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
