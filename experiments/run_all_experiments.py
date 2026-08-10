"""Run only the evaluation paths listed in the current evidence register.

This runner intentionally excludes superseded and exploratory experiments.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    key: str
    description: str
    command: tuple[str, ...]
    group: str


EXPERIMENTS = (
    Experiment(
        "paper-protocol",
        "Validate the locked paper evaluation and promotion contract",
        ("experiments/validate_paper_protocol.py",),
        "checks",
    ),
    Experiment(
        "corpus",
        "Validate the reconstruction-free derived corpus",
        ("curation/validate_preserved_nonbib_corpus.py", "--derived-only"),
        "checks",
    ),
    Experiment(
        "leakage",
        "Validate the canonical registry boundaries and held-out exclusions",
        ("experiments/validate_leakage.py",),
        "checks",
    ),
    Experiment(
        "paper-snapshot",
        "Validate and aggregate frozen target-level paper artifacts",
        ("experiments/run_paper_benchmark.py",),
        "checks",
    ),
    Experiment(
        "data-profile",
        "Regenerate descriptive corpus and lacuna-shape statistics",
        ("experiments/build_paper_data_profile.py",),
        "checks",
    ),
    Experiment(
        "split-audit",
        "Audit composition and near-duplicate overlap in the frozen split",
        ("experiments/audit_split_similarity.py",),
        "checks",
    ),
    Experiment(
        "qd-composition-manifest",
        "Validate the QD-target composition-exclusion manifest",
        ("experiments/build_qd_composition_exclusion.py", "--validate-existing"),
        "checks",
    ),
    Experiment(
        "qd-memorization-audit",
        "Audit exact preserved-training parallels for QD readings",
        ("experiments/audit_qd_memorization.py",),
        "checks",
    ),
    Experiment(
        "qd-method-summary",
        "Aggregate matched MLM seeds with seed-and-scroll uncertainty",
        ("experiments/aggregate_qd_method_extensions.py",),
        "checks",
    ),
    Experiment(
        "qd-context-summary",
        "Regenerate fixed-seed context-window sensitivity",
        ("experiments/summarize_qd_context_windows.py",),
        "checks",
    ),
    Experiment(
        "qd-composition-summary",
        "Regenerate the composition-excluded checkpoint sensitivity",
        ("experiments/summarize_qd_composition_exclusion.py",),
        "checks",
    ),
    Experiment(
        "qd",
        "Attributed Qumran Digital evidence-condition benchmark",
        (
            "experiments/run_qd_benchmark.py",
            "--report", "experiments/results/paper/qd_methods_seed42_20260811.json",
            "--markdown", "comparison/reports/QD_METHODS_SEED42_20260811.md",
        ),
        "pilots",
    ),
    Experiment(
        "embible",
        "Embible-style unknown-boundary character/word DSS baseline",
        (
            "experiments/tf_embible_dss_benchmark.py",
            "--dev-per-length",
            "20",
            "--test-per-length",
            "100",
            "--context-words", "2",
            "--word-model", "models/ft_msbert_span_preserved_nonbib",
            "--char-model", "models/ft_tavbert_span_preserved_nonbib_seed42",
            "--output-json", "experiments/results/paper/span_balanced_300_20260811.json",
            "--output-markdown", "comparison/reports/SPAN_BALANCED_300_20260811.md",
            "--local-files-only",
        ),
        "pilots",
    ),
    *(
        Experiment(
            f"byt5-{seed}",
            f"Matched ByT5 seed {seed} on the balanced span benchmark",
            (
                "experiments/tf_tokenization_free_benchmark.py",
                "--model-dir", model,
                "--split", "heldout",
                "--per-length", "100",
                "--seed", "73",
                "--context-words", "2",
                "--beam-width", "10",
                "--batch-size", "8",
                "--local-files-only",
                "--output-json",
                f"experiments/results/paper/byt5_balanced_seed{seed}_20260811.json",
            ),
            "pilots",
        )
        for seed, model in (
            (41, "models/ft_byt5_span_scroll_conditioned_seed41"),
            (42, "models/ft_byt5_span_preserved_nonbib_seed42"),
            (43, "models/ft_byt5_span_preserved_nonbib_seed43"),
        )
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list registered evaluations")
    mode.add_argument(
        "--checks", action="store_true", help="run validation checks only"
    )
    mode.add_argument(
        "--pilots", action="store_true", help="run retained pilot evaluations"
    )
    mode.add_argument(
        "--all", action="store_true", help="run checks followed by pilots"
    )
    parser.add_argument(
        "--only",
        choices=[experiment.key for experiment in EXPERIMENTS],
        action="append",
        help="restrict the selected mode to one or more experiment keys",
    )
    return parser.parse_args()


def select(args: argparse.Namespace) -> list[Experiment]:
    if args.list or args.all:
        selected = list(EXPERIMENTS)
    elif args.checks:
        selected = [item for item in EXPERIMENTS if item.group == "checks"]
    else:
        selected = [item for item in EXPERIMENTS if item.group == "pilots"]
    if args.only:
        allowed = set(args.only)
        selected = [item for item in selected if item.key in allowed]
    return selected


def main() -> int:
    args = parse_args()
    selected = select(args)
    if args.list:
        for experiment in selected:
            command = " ".join((sys.executable, *experiment.command))
            print(f"{experiment.key:10s} [{experiment.group}] {experiment.description}")
            print(f"{'':10s} {command}")
        return 0

    failures: list[str] = []
    for experiment in selected:
        print(f"\n=== {experiment.key}: {experiment.description} ===", flush=True)
        result = subprocess.run(
            [sys.executable, *experiment.command],
            cwd=ROOT,
            check=False,
        )
        if result.returncode:
            failures.append(experiment.key)
            print(
                f"FAILED: {experiment.key} exited {result.returncode}", file=sys.stderr
            )

    if failures:
        print(f"\nFailed evaluations: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll selected evaluations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
