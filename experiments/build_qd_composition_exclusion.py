#!/usr/bin/env python3
"""Build a training subset with QD target compositions excluded."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curation.preserved_corpus import load_chunks

QD_RESULT = ROOT / "experiments/results/paper/qd_evidence_conditions_20260811.json"
SOURCE_CSV = ROOT / "dss_chunks.csv"
DEFAULT_OUTPUT = ROOT / "experiments/results/paper/qd_composition_exclusion.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def composition_map(path: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["bib"] == "nonbib" and row["composition"].strip():
                mapping[row["book"]].add(row["composition"].strip())
    return mapping


def validate(result: dict) -> None:
    target = set(result["target_compositions"])
    retained = set(result["retained_compositions"])
    overlap = sorted(target & retained)
    if overlap or result["verified_composition_overlap"]:
        raise AssertionError(f"composition exclusion failed: {overlap}")
    if result["retained_train_chunks"] >= result["original_train_chunks"]:
        raise AssertionError("composition exclusion did not remove training chunks")


def build(qd_result: Path, source_csv: Path) -> dict:
    qd_result = qd_result.resolve()
    source_csv = source_csv.resolve()
    qd = json.loads(qd_result.read_text(encoding="utf-8"))
    compositions = composition_map(source_csv)
    target_scrolls = {str(row["siglum"]) for row in qd["targets"]}
    target_compositions = {
        label for scroll in target_scrolls for label in compositions.get(scroll, set())
    }
    train_rows = load_chunks("train")
    dev_rows = load_chunks("dev")

    def retained(rows: list[dict]) -> list[dict]:
        return [
            row
            for row in rows
            if compositions.get(row["scroll"], set()).isdisjoint(target_compositions)
        ]

    retained_train = retained(train_rows)
    retained_dev = retained(dev_rows)
    retained_train_scrolls = sorted({row["scroll"] for row in retained_train})
    retained_dev_scrolls = sorted({row["scroll"] for row in retained_dev})
    overlap = {
        label
        for scroll in retained_train_scrolls + retained_dev_scrolls
        for label in compositions.get(scroll, set())
    } & target_compositions
    if overlap:
        raise AssertionError(f"composition exclusion failed: {sorted(overlap)}")

    labeled_targets = [
        row for row in qd["targets"] if compositions.get(str(row["siglum"]), set())
    ]
    return {
        "status": "qd_target_composition_excluded_training_sensitivity",
        "definition": (
            "retain frozen train/dev scrolls only when every non-empty composition "
            "label is absent from the labeled QD target set"
        ),
        "inputs": {
            "qd_result": str(qd_result.relative_to(ROOT)),
            "qd_result_sha256": sha256(qd_result),
            "composition_csv": str(source_csv.relative_to(ROOT)),
            "composition_csv_sha256": sha256(source_csv),
        },
        "target_scrolls": sorted(target_scrolls),
        "target_compositions": sorted(target_compositions),
        "retained_compositions": sorted(
            {
                label
                for scroll in retained_train_scrolls + retained_dev_scrolls
                for label in compositions.get(scroll, set())
            }
        ),
        "labeled_targets": len(labeled_targets),
        "labeled_target_scrolls": len(
            {str(row["siglum"]) for row in labeled_targets}
        ),
        "original_train_chunks": len(train_rows),
        "retained_train_chunks": len(retained_train),
        "original_dev_chunks": len(dev_rows),
        "retained_dev_chunks": len(retained_dev),
        "train_scrolls": retained_train_scrolls,
        "dev_scrolls": retained_dev_scrolls,
        "verified_composition_overlap": sorted(overlap),
        "scope_note": (
            "This is a targeted composition-exclusion sensitivity analysis, not a "
            "new representative population split. Unlabeled targets are not used "
            "for the composition-disjoint claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qd-result", type=Path, default=QD_RESULT)
    parser.add_argument("--composition-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="validate the checked-in manifest without the licensed source CSV",
    )
    args = parser.parse_args()
    if args.validate_existing:
        validate(json.loads(args.output.read_text(encoding="utf-8")))
        print(f"composition exclusion manifest valid: {args.output}")
        return
    result = build(args.qd_result, args.composition_csv)
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
