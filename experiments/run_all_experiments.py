"""Run only the evaluation paths listed in the current evidence register.

This runner intentionally excludes superseded and exploratory experiments.
It does not turn the retained pilot numbers into paper results; see
``docs/METHODOLOGY.md`` for the promotion gate.
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
        ("curation/validate_preserved_nonbib_corpus.py",),
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
        "qd",
        "Attributed Qumran Digital literature-agreement pilot",
        ("experiments/run_qd_benchmark.py",),
        "pilots",
    ),
    Experiment(
        "rag",
        "Train-only RAG single- and multiword paired pilot",
        ("experiments/tf_preserved_rag_multiword_benchmark.py", "--per-bucket", "25"),
        "pilots",
    ),
    Experiment(
        "embible",
        "Embible-style unknown-boundary character/word DSS baseline",
        (
            "experiments/tf_embible_dss_benchmark.py",
            "--dev-per-length",
            "5",
            "--test-per-length",
            "10",
            "--local-files-only",
        ),
        "pilots",
    ),
    Experiment(
        "bible-transfer",
        "Fixed-decoder transfer diagnostic on Embible Biblical verses",
        (
            "experiments/embible_bible_transfer_benchmark.py",
            "--local-files-only",
        ),
        "pilots",
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
