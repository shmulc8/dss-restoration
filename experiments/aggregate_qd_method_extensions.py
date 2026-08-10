#!/usr/bin/env python3
"""Aggregate matched-seed and composition-exclusion QD analyses."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = tuple(
    ROOT / f"experiments/results/paper/qd_methods_seed{seed}_20260811.json"
    for seed in (41, 42, 43)
)
DEFAULT_OUTPUT = ROOT / "experiments/results/paper/qd_method_extensions_summary.json"
CONDITIONS = ("context_only", "soft_visible", "visible_only")
METRICS = ("top1", "top5", "top10", "top20")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hit(record: dict[str, Any], condition: str) -> int:
    rank = record[f"{condition}_rank"]
    return int(rank is not None and rank < 10)


def hierarchical_ci(
    reports: list[dict[str, Any]], condition: str, *, samples: int = 10_000
) -> list[float]:
    by_seed_scroll: list[dict[str, list[int]]] = []
    for report in reports:
        grouped: dict[str, list[int]] = defaultdict(list)
        for record in report["targets"]:
            grouped[str(record["siglum"])].append(hit(record, condition))
        by_seed_scroll.append(grouped)
    scrolls = sorted(by_seed_scroll[0])
    generator = random.Random(20260811)
    estimates = []
    for _ in range(samples):
        sampled_seeds = [generator.randrange(len(reports)) for _ in reports]
        sampled_scrolls = [scrolls[generator.randrange(len(scrolls))] for _ in scrolls]
        values = [
            value
            for seed in sampled_seeds
            for scroll in sampled_scrolls
            for value in by_seed_scroll[seed][scroll]
        ]
        estimates.append(100 * sum(values) / len(values))
    estimates.sort()
    return [
        estimates[int(0.025 * samples)],
        estimates[int(0.975 * samples) - 1],
    ]


def hierarchical_delta_ci(
    reports: list[dict[str, Any]], left: str, right: str, *, samples: int = 10_000
) -> list[float]:
    by_seed_scroll: list[dict[str, list[tuple[int, int]]]] = []
    for report in reports:
        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for record in report["targets"]:
            grouped[str(record["siglum"])].append(
                (hit(record, left), hit(record, right))
            )
        by_seed_scroll.append(grouped)
    scrolls = sorted(by_seed_scroll[0])
    generator = random.Random(20260812)
    estimates = []
    for _ in range(samples):
        sampled_seeds = [generator.randrange(len(reports)) for _ in reports]
        sampled_scrolls = [scrolls[generator.randrange(len(scrolls))] for _ in scrolls]
        pairs = [
            pair
            for seed in sampled_seeds
            for scroll in sampled_scrolls
            for pair in by_seed_scroll[seed][scroll]
        ]
        estimates.append(100 * sum(right_hit - left_hit for left_hit, right_hit in pairs) / len(pairs))
    estimates.sort()
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]]


def aggregate(paths: tuple[Path, ...]) -> dict[str, Any]:
    reports = [load(path) for path in paths]
    identities = [
        [(str(row["siglum"]), int(row["word_id"])) for row in report["targets"]]
        for report in reports
    ]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("matched seed reports do not share the same ordered targets")
    rows = []
    for path, report in zip(paths, reports):
        metadata = report["protocol"]["training"]
        rows.append(
            {
                "seed": metadata["seed"],
                "epochs": metadata["epochs"],
                "batch_size": metadata.get("batch_size", metadata.get("batch")),
                "learning_rate": metadata["learning_rate"],
                "epoch_losses": metadata["epoch_losses"],
                "result": str(path.relative_to(ROOT)),
                **{
                    f"{condition}_{metric}": report["condition_results"][condition][metric]
                    for condition in CONDITIONS
                    for metric in METRICS
                },
            }
        )
    conditions = {}
    for condition in CONDITIONS:
        values = [row[f"{condition}_top10"] for row in rows]
        conditions[condition] = {
            **{
                f"mean_{metric}": statistics.mean(
                    row[f"{condition}_{metric}"] for row in rows
                )
                for metric in METRICS
            },
            "sample_sd_across_seeds": statistics.stdev(values),
            "minimum_top10": min(values),
            "maximum_top10": max(values),
            "seed_and_scroll_hierarchical_bootstrap_95ci": hierarchical_ci(
                reports, condition
            ),
        }
    return {
        "status": "controlled_three_seed_qd_replication",
        "targets": len(identities[0]),
        "scrolls": len({siglum for siglum, _ in identities[0]}),
        "seeds": rows,
        "conditions": conditions,
        "paired_top10_deltas": {
            "soft_visible_minus_context": {
                "mean_points": conditions["soft_visible"]["mean_top10"]
                - conditions["context_only"]["mean_top10"],
                "seed_and_scroll_hierarchical_bootstrap_95ci": hierarchical_delta_ci(
                    reports, "context_only", "soft_visible"
                ),
            },
            "soft_visible_minus_frequency_visible": {
                "mean_points": conditions["soft_visible"]["mean_top10"]
                - reports[0]["condition_results"]["frequency_visible_only"]["top10"],
                "seed_and_scroll_hierarchical_bootstrap_95ci": hierarchical_delta_ci(
                    reports, "frequency_visible_only", "soft_visible"
                ),
            },
        },
        "uncertainty": (
            "hierarchical bootstrap resamples three optimization seeds and manuscript "
            "clusters with replacement"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-reports", nargs=3, type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = aggregate(tuple(args.seed_reports))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
