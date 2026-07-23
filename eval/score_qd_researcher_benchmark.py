"""Score attributed Qumran Digital readings with the preserved-only model.

This measures whether a publication's proposed word appears among the model's
contextual candidates after the entire target word is masked.  It is a
literature-agreement metric, not proof that either the scholar or model has
reconstructed the physical manuscript correctly.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "derived" / "qd_researcher_variants.jsonl"
DEFAULT_MODEL = ROOT / "ft_msbert_span_preserved_nonbib"
DEFAULT_REPORT = (
    ROOT / "analysis" / "reports" / "qd_researcher_comparison.json"
)
HEBREW = set(chr(codepoint) for codepoint in range(0x05D0, 0x05EB))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
tlog.set_verbosity_error()


def hebrew_letters(value: str) -> str:
    return "".join(character for character in (value or "") if character in HEBREW)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_scorable(row: dict[str, Any], min_context_words: int) -> bool:
    """Require a complete candidate and enough genuinely visible context."""
    reading = row["reading"]
    if "○" in reading or "--" in reading or "." in reading:
        return False
    if len(row["reading_hebrew"]) < 2:
        return False
    visible_context = sum(
        word not in {"<GAP>", "<TARGET>"} for word in row["context_words"]
    )
    return visible_context >= min_context_words


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [record["rank"] for record in records]
    finite = [rank for rank in ranks if rank is not None]
    total = len(records)
    return {
        "n": total,
        "top1": sum(rank == 0 for rank in finite) / total * 100 if total else 0,
        "top5": sum(rank is not None and rank < 5 for rank in ranks)
        / total
        * 100
        if total
        else 0,
        "top10": sum(rank is not None and rank < 10 for rank in ranks)
        / total
        * 100
        if total
        else 0,
        "top20": sum(rank is not None and rank < 20 for rank in ranks)
        / total
        * 100
        if total
        else 0,
        "median_rank_when_in_vocabulary": (
            statistics.median(finite) if finite else None
        ),
        "in_vocabulary": len(finite),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-context-words", type=int, default=10)
    args = parser.parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not 1 <= args.batch_size <= 128:
        raise ValueError("--batch-size must be between 1 and 128")

    all_rows = read_jsonl(args.input)
    rows = [
        row for row in all_rows if is_scorable(row, args.min_context_words)
    ]
    if not rows:
        raise RuntimeError("No scorable rows remain after quality filtering")
    targets: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        targets[(row["siglum"], row["word_id"])] = row
    target_items = list(targets.values())
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(args.model))
    model = model.to(DEVICE).eval()
    normalized_token_by_id = [
        hebrew_letters(tokenizer.decode([token_id]).strip())
        for token_id in range(len(tokenizer))
    ]

    predictions_by_target: dict[tuple[str, int], list[str]] = {}
    for start in range(0, len(target_items), args.batch_size):
        batch = target_items[start : start + args.batch_size]
        model_words = [
            [
                tokenizer.mask_token if word in {"<GAP>", "<TARGET>"} else word
                for word in item["context_words"]
            ]
            for item in batch
        ]
        encoding = tokenizer(
            model_words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"].clone()
        positions: list[int] = []
        for batch_index, item in enumerate(batch):
            target_positions = [
                position
                for position, word_id in enumerate(
                    encoding.word_ids(batch_index=batch_index)
                )
                if word_id == item["target_index"]
            ]
            if not target_positions:
                raise RuntimeError(
                    f"Target vanished during tokenization: "
                    f"{item['siglum']}:{item['word_id']}"
                )
            positions.append(target_positions[0])
            for position in target_positions:
                input_ids[batch_index, position] = tokenizer.mask_token_id
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE),
            ).logits.cpu()
        for batch_index, item in enumerate(batch):
            ordered_ids = torch.argsort(
                logits[batch_index, positions[batch_index]],
                descending=True,
            ).tolist()
            predictions: list[str] = []
            seen: set[str] = set()
            for token_id in ordered_ids:
                reading = normalized_token_by_id[token_id]
                if len(reading) < 2 or reading in seen:
                    continue
                seen.add(reading)
                predictions.append(reading)
            predictions_by_target[(item["siglum"], item["word_id"])] = predictions
        print(
            f"scored targets: {min(start + args.batch_size, len(target_items))}"
            f"/{len(target_items)}",
            flush=True,
        )

    scored: list[dict[str, Any]] = []
    for row in rows:
        predictions = predictions_by_target[(row["siglum"], row["word_id"])]
        gold = row["reading_hebrew"]
        rank = predictions.index(gold) if gold in predictions else None
        qd_normalized = hebrew_letters(row["qd_initial_reading"])
        scored.append(
            {
                **row,
                "rank": rank,
                "top_predictions": predictions[:20],
                "distinct_from_qd": gold != qd_normalized,
            }
        )

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored:
        by_source[record["bibliography_abbreviation"]].append(record)
    source_results = {
        source: {
            **summarize(records),
            "bibliography_id": records[0]["bibliography_id"],
            "bibliography_formatted": records[0]["bibliography_formatted"],
            "distinct_only": summarize(
                [record for record in records if record["distinct_from_qd"]]
            ),
        }
        for source, records in sorted(by_source.items())
    }
    report = {
        "protocol": {
            "model": str(args.model),
            "device": DEVICE,
            "target": "whole masked word",
            "candidate_normalization": "Hebrew consonants only",
            "training_overlap": (
                "none at scroll level: only the pre-existing heldout split"
            ),
            "surrounding_reconstructions": "square-bracket words redacted",
            "interpretation": (
                "contextual agreement with attributed published readings; "
                "not manuscript-grounded restoration accuracy"
            ),
            "selection_bias": (
                "only variants QD selected for inline transcription display"
            ),
            "quality_filter": {
                "minimum_visible_context_words": args.min_context_words,
                "incomplete_readings_excluded": ["○", "--", "."],
                "input_rows": len(all_rows),
                "scored_rows": len(rows),
            },
        },
        "overall": summarize(scored),
        "distinct_from_qd_only": summarize(
            [record for record in scored if record["distinct_from_qd"]]
        ),
        "by_bibliographic_source": source_results,
        "records": scored,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved -> {args.report}")


if __name__ == "__main__":
    main()
