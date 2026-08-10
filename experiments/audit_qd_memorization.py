#!/usr/bin/env python3
"""Audit exact preserved-training parallels for QD attributed readings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curation.preserved_corpus import GAP_TOKEN, load_chunks
from experiments.run_qd_benchmark import contiguous_context, join_clitics, rag_normalize

DEFAULT_INPUT = ROOT / "experiments/results/paper/qd_methods_seed42_20260811.json"
DEFAULT_OUTPUT = ROOT / "experiments/results/paper/qd_memorization_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventories() -> tuple[Counter[str], dict[int, set[tuple[str, ...]]]]:
    vocabulary: Counter[str] = Counter()
    ngrams = {3: set(), 5: set()}
    for row in load_chunks("train"):
        segment: list[str] = []
        for token in join_clitics(row["text"].split() + [GAP_TOKEN]):
            word = rag_normalize(token)
            if token == GAP_TOKEN or not word:
                for width in ngrams:
                    ngrams[width].update(
                        tuple(segment[index : index + width])
                        for index in range(len(segment) - width + 1)
                    )
                segment = []
            else:
                vocabulary[word] += 1
                segment.append(word)
    return vocabulary, ngrams


def audit(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    vocabulary, ngrams = inventories()
    records = []
    for target in report["targets"]:
        # Current scored artifacts retain predictions, not the repeated source context.
        # Recover it from the frozen input using the stable target key below.
        records.append(
            {
                "siglum": target["siglum"],
                "word_id": target["word_id"],
                "attributed_readings": target["attributed_readings"],
                "lexical_seen": [
                    reading for reading in target["attributed_readings"] if rag_normalize(reading) in vocabulary
                ],
            }
        )

    source_rows = {}
    qd_input = ROOT / "curation/derived/qd_researcher_variants.jsonl"
    with qd_input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_rows.setdefault((str(row["siglum"]), int(row["word_id"])), row)

    for record in records:
        source = source_rows[(str(record["siglum"]), int(record["word_id"]))]
        words = source["context_words"]
        index = int(source["target_index"])
        left, right = contiguous_context(words, index, max_side=2)
        left, right = list(left), list(right)
        exact = {"3": [], "5": []}
        for reading in record["attributed_readings"]:
            normalized = rag_normalize(reading)
            if left and right and tuple([left[-1], normalized, right[0]]) in ngrams[3]:
                exact["3"].append(reading)
            if len(left) == 2 and len(right) == 2 and tuple(left + [normalized] + right) in ngrams[5]:
                exact["5"].append(reading)
        record["exact_training_parallel"] = exact

    return {
        "status": "exact_preserved_training_parallel_audit",
        "input": str(path.resolve().relative_to(ROOT)),
        "input_sha256": sha256(path),
        "targets": len(records),
        "targets_with_lexically_seen_reading": sum(bool(row["lexical_seen"]) for row in records),
        "targets_with_exact_3gram_parallel": sum(
            bool(row["exact_training_parallel"]["3"]) for row in records
        ),
        "targets_with_exact_5gram_parallel": sum(
            bool(row["exact_training_parallel"]["5"]) for row in records
        ),
        "definition": {
            "3gram": "one preserved word on each side plus an attributed reading",
            "5gram": "two preserved words on each side plus an attributed reading",
            "gap_handling": "ngrams never cross anonymous natural gap markers",
        },
        "unresolved": (
            "Base-model pretraining corpora are not fully enumerated, so this audit cannot "
            "exclude pretraining exposure."
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
