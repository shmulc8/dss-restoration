"""Run the QD real-lacuna benchmark with a character-level masked LM.

The QD scorer supplies target eligibility, physical constraints, and rank
scoring. This engine changes only candidate generation: it places the visibly
preserved letters in every physically compatible position, masks the unknown
character slots, and beam-decodes those slots.

The external comparison code is expected at
``scratch/external_impl/new_dead_sea_scrolls``; see ``README.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = HERE / "results" / "qd_char"

sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(TUNING))

import score_qd_researcher_benchmark as qd  # noqa: E402
from eval.masking import MultiSpanPredictionPolicy  # noqa: E402


LENGTH_TOLERANCE = 1
POOL_TOLERANCE = 2
MIN_CONTEXT_WORDS = 10
BEAM_WIDTH = 50
TOP_K = 50
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAFE_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model ID or local model path")
    parser.add_argument(
        "tag",
        help="Safe filename stem for external_comparison/results/qd_char/<tag>.json",
    )
    args = parser.parse_args()
    if not SAFE_TAG_RE.fullmatch(args.tag):
        parser.error(
            "tag may contain only letters, digits, dot, underscore, and hyphen"
        )
    if not TUNING.is_dir():
        parser.error(f"external tuning checkout not found: {TUNING}")
    return args


def load_eligible_targets() -> dict[tuple[str, int], dict[str, Any]]:
    """Apply the QD scorer's pool and final target-inclusion rules."""
    target_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in qd.read_jsonl(qd.DEFAULT_INPUT):
        target_rows[(str(row["siglum"]), int(row["word_id"]))].append(row)

    eligible: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in target_rows.items():
        representative = rows[0]
        visible_context = sum(
            word not in {"<GAP>", "<TARGET>"}
            for word in representative["context_words"]
        )
        if visible_context < MIN_CONTEXT_WORDS:
            continue
        constraint, _ = qd.build_constraint(representative)
        if constraint is None:
            continue

        readings: dict[str, set[str]] = {}
        for row in rows:
            normalized, _ = qd.parse_attributed_reading(row, constraint, POOL_TOLERANCE)
            if normalized is not None:
                readings.setdefault(normalized, set()).add(str(row["reading"]))
        # The main scorer builds a tolerance-2 pool, then drops targets that have
        # no attributed reading compatible with the primary tolerance (here ±1).
        primary_readings = {
            reading: raw
            for reading, raw in readings.items()
            if constraint.matches(reading, LENGTH_TOLERANCE)
        }
        if not primary_readings:
            continue
        eligible[key] = {
            **representative,
            "constraint": constraint,
            "readings": primary_readings,
        }
    return eligible


def placement_templates(
    constraint: qd.PhysicalConstraint,
    length: int,
    limit: int = 60,
) -> list[list[str | None]]:
    """Return ordered, anchored placements of visible segments in a word."""
    segments = list(constraint.visible_segments)
    output: list[list[str | None]] = []

    def place(segment_index: int, position: int, template: list[str | None]) -> None:
        if len(output) >= limit:
            return
        if segment_index == len(segments):
            output.append(template)
            return

        segment = segments[segment_index]
        latest_start = length - sum(len(item) for item in segments[segment_index:])
        earliest_start = position
        if segment_index == 0 and constraint.anchored_left:
            latest_start = min(latest_start, 0)
            earliest_start = max(earliest_start, 0)
        if segment_index == len(segments) - 1 and constraint.anchored_right:
            right_anchor = length - len(segment)
            earliest_start = max(earliest_start, right_anchor)
            latest_start = min(latest_start, right_anchor)

        for start in range(max(earliest_start, position), latest_start + 1):
            placed = list(template)
            placed[start : start + len(segment)] = segment
            place(segment_index + 1, start + len(segment), placed)

    place(0, 0, [None] * length)
    return output


def target_text(item: dict[str, Any], mask: str, length: int) -> str:
    return " ".join(
        mask * length if word == "<TARGET>" else (mask if word == "<GAP>" else word)
        for word in item["context_words"]
    )


def target_span(
    policy: MultiSpanPredictionPolicy,
    encoded: Any,
    item: dict[str, Any],
    length: int,
) -> list[int] | None:
    """Locate the target mask run, with a tokenizer-based fallback."""
    target_index = int(item["target_index"])
    start = 1 + sum(
        (length if index == target_index else (1 if word == "<GAP>" else len(word))) + 1
        for index, word in enumerate(item["context_words"][:target_index])
    )
    ids = encoded.input_ids[0]
    span = list(range(start, start + length))
    if span[-1] < len(ids) and all(
        ids[position].item() == policy.tokenizer.mask_token_id for position in span
    ):
        return span

    positions = (ids == policy.tokenizer.mask_token_id).nonzero().flatten()
    runs = policy._group_mask_spans(positions)
    candidates = [run for run in runs if len(run) == length]
    return candidates[0] if len(candidates) == 1 else None


def decode_target(
    policy: MultiSpanPredictionPolicy,
    tokenizer: Any,
    item: dict[str, Any],
) -> tuple[list[str], int]:
    """Generate and rank normalized candidates for one target."""
    constraint = item["constraint"]
    lengths: Iterable[int] = sorted(
        length
        for length in range(
            constraint.estimated_length - 1, constraint.estimated_length + 2
        )
        if length >= 2
    )
    merged: dict[str, float] = {}
    skipped = 0

    for length in lengths:
        for template in placement_templates(constraint, length):
            encoded = policy._encode(target_text(item, tokenizer.mask_token, length))
            span = target_span(policy, encoded, item, length)
            if span is None:
                skipped += 1
                continue

            free_positions: list[int] = []
            for offset, character in enumerate(template):
                if character is None:
                    free_positions.append(span[offset])
                else:
                    token_id = tokenizer.convert_tokens_to_ids(character)
                    encoded.input_ids[0, span[offset]] = token_id

            decoded = (
                policy._decode_span_with_beam(encoded, free_positions)
                if free_positions
                else [{"score": 0.0, "token_ids": []}]
            )
            for candidate in decoded:
                filled = dict(zip(free_positions, candidate["token_ids"]))
                word = "".join(
                    character
                    if character is not None
                    else tokenizer.convert_ids_to_tokens(filled[span[offset]])
                    for offset, character in enumerate(template)
                )
                normalized = qd.hebrew_letters(word)
                if len(normalized) < 2:
                    continue
                mean_log_probability = candidate["score"] / max(len(free_positions), 1)
                merged[normalized] = max(
                    mean_log_probability, merged.get(normalized, float("-inf"))
                )

    ordered = [
        candidate for candidate, _ in sorted(merged.items(), key=lambda row: -row[1])
    ]
    return (
        [
            candidate
            for candidate in ordered
            if constraint.matches(candidate, LENGTH_TOLERANCE)
        ],
        skipped,
    )


def hit_rate(ranks: list[int | None], cutoff: int) -> float:
    return sum(rank is not None and rank < cutoff for rank in ranks) / len(ranks)


def main() -> None:
    args = parse_args()
    eligible = load_eligible_targets()
    print(f"eligible targets: {len(eligible)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(DEVICE).eval()
    policy = MultiSpanPredictionPolicy(
        model=model,
        tokenizer=tokenizer,
        top_k=TOP_K,
        beam_width=BEAM_WIDTH,
        beam_depth=32,
        device=DEVICE,
    )

    ranks: list[int | None] = []
    skipped = 0
    for completed, item in enumerate(eligible.values(), 1):
        candidates, target_skipped = decode_target(policy, tokenizer, item)
        skipped += target_skipped
        target_ranks = [qd.rank_of(reading, candidates) for reading in item["readings"]]
        finite = [rank for rank in target_ranks if rank is not None]
        ranks.append(min(finite) if finite else None)
        if completed % 10 == 0:
            print(f"{completed}/{len(eligible)} targets", flush=True)

    result = {
        "model": args.model,
        "engine": "char-beam-partial-letters",
        "targets": len(ranks),
        "skipped_length_hypotheses": skipped,
        "beam_width": BEAM_WIDTH,
        "length_tolerance": LENGTH_TOLERANCE,
        "top1": hit_rate(ranks, 1),
        "top10": hit_rate(ranks, 10),
        "top20": hit_rate(ranks, 20),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
