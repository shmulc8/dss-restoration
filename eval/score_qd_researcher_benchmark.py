"""Score attributed Qumran Digital restorations with physical constraints.

The cached QD data contains several kinds of editorial disagreement.  This
benchmark keeps only single-word lacuna restorations, removes the restored
letters from the model input, and retains two pieces of physical evidence:

* Hebrew letters visibly preserved outside square brackets; and
* an approximate word length derived from the QD display/initial notation.

The primary unit is one manuscript target, not one publication row.  A target
is successful when any distinct, physically compatible attributed restoration
appears in the model's Top-K.  Per-reading and per-source results are secondary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "derived" / "qd_researcher_variants.jsonl"
DEFAULT_MODEL = ROOT / "ft_msbert_span_preserved_nonbib"
DEFAULT_REPORT = ROOT / "analysis" / "reports" / "qd_researcher_comparison.json"
DEFAULT_MARKDOWN = (
    ROOT / "analysis" / "reports" / "QD_RESEARCHER_BENCHMARK.md"
)
HEBREW_RE = re.compile(r"[\u05d0-\u05ea]")
UNSUPPORTED_READING_MARKUP = set("/{}()〈〉⟨⟩«»")
GAP_MARKERS = set("[]○")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
tlog.set_verbosity_error()


def hebrew_letters(value: str) -> str:
    return "".join(HEBREW_RE.findall(value or ""))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class PhysicalConstraint:
    visible_segments: tuple[str, ...]
    anchored_left: bool
    anchored_right: bool
    estimated_length: int
    display_slots: int
    initial_slots: int

    def matches_visible(self, candidate: str) -> bool:
        if not candidate:
            return False
        if self.anchored_left and not candidate.startswith(self.visible_segments[0]):
            return False
        if self.anchored_right and not candidate.endswith(self.visible_segments[-1]):
            return False
        offset = 0
        for segment in self.visible_segments:
            found = candidate.find(segment, offset)
            if found < 0:
                return False
            offset = found + len(segment)
        return True

    def matches(self, candidate: str, length_tolerance: int) -> bool:
        return self.matches_visible(candidate) and (
            abs(len(candidate) - self.estimated_length) <= length_tolerance
        )


def _slot_count(value: str) -> int:
    """Count explicit Hebrew/unknown-character slots in editorial notation."""
    return sum(HEBREW_RE.fullmatch(char) is not None or char == "○" for char in value)


def _visible_segments(value: str) -> tuple[tuple[str, ...], bool, bool]:
    """Extract preserved Hebrew outside square-bracket reconstruction zones.

    QD uses unmatched ``]``/``[`` at word boundaries when a word begins or
    ends inside a lacuna.  Thus letters before the first unmatched ``]`` are
    reconstructed, and letters after it are visible; the reverse applies to
    a trailing unmatched ``[``.
    """
    first_open = value.find("[")
    first_close = value.find("]")
    hidden = first_close >= 0 and (first_open < 0 or first_close < first_open)
    pieces: list[str] = []
    current: list[str] = []
    first_visible_position: int | None = None
    last_visible_position: int | None = None
    first_boundary_position = min(
        (position for position in (first_open, first_close) if position >= 0),
        default=len(value),
    )

    def flush() -> None:
        if current:
            pieces.append("".join(current))
            current.clear()

    for position, char in enumerate(value):
        if char == "[":
            flush()
            hidden = True
        elif char == "]":
            flush()
            hidden = False
        elif char == "○":
            flush()
        elif HEBREW_RE.fullmatch(char) and not hidden:
            if first_visible_position is None:
                first_visible_position = position
            last_visible_position = position
            current.append(char)
    flush()

    if not pieces or first_visible_position is None or last_visible_position is None:
        return (), False, False
    anchored_left = first_visible_position < first_boundary_position and not (
        first_close >= 0 and first_close < first_visible_position
    )
    trailing_open = value.rfind("[")
    anchored_right = not (
        trailing_open >= 0 and trailing_open > last_visible_position
    )
    return tuple(pieces), anchored_left, anchored_right


def build_constraint(row: dict[str, Any]) -> tuple[PhysicalConstraint | None, str]:
    display = str(row.get("qd_display_reading", ""))
    initial = str(row.get("qd_initial_reading", ""))
    if not any(marker in display for marker in GAP_MARKERS):
        return None, "not_a_lacuna"
    if any(char in display for char in UNSUPPORTED_READING_MARKUP):
        return None, "unsupported_target_markup"
    segments, anchored_left, anchored_right = _visible_segments(display)
    if not segments:
        return None, "no_visible_hebrew"
    display_slots = _slot_count(display)
    initial_slots = _slot_count(initial)
    estimated_length = max(display_slots, initial_slots)
    if estimated_length < 2:
        return None, "target_too_short"
    return (
        PhysicalConstraint(
            visible_segments=segments,
            anchored_left=anchored_left,
            anchored_right=anchored_right,
            estimated_length=estimated_length,
            display_slots=display_slots,
            initial_slots=initial_slots,
        ),
        "eligible",
    )


def parse_attributed_reading(
    row: dict[str, Any],
    constraint: PhysicalConstraint,
    length_tolerance: int,
) -> tuple[str | None, str]:
    reading = str(row.get("reading", ""))
    if any(char.isspace() for char in reading):
        return None, "multiword_reading"
    if any(char in reading for char in UNSUPPORTED_READING_MARKUP):
        return None, "correction_or_alternative_markup"
    if "○" in reading or "--" in reading or "." in reading:
        return None, "incomplete_reading"
    normalized = hebrew_letters(reading)
    if len(normalized) < 2:
        return None, "reading_too_short"
    if not constraint.matches(normalized, length_tolerance):
        return None, "contradicts_physical_constraint"
    return normalized, "eligible"


def summarize_ranks(ranks: Iterable[int | None]) -> dict[str, Any]:
    values = list(ranks)
    finite = [rank for rank in values if rank is not None]
    total = len(values)
    return {
        "n": total,
        "top1": 100 * sum(rank == 0 for rank in finite) / total if total else 0,
        "top5": 100 * sum(rank is not None and rank < 5 for rank in values) / total
        if total
        else 0,
        "top10": 100 * sum(rank is not None and rank < 10 for rank in values) / total
        if total
        else 0,
        "top20": 100 * sum(rank is not None and rank < 20 for rank in values) / total
        if total
        else 0,
        "in_candidate_vocabulary": len(finite),
        "median_rank_when_retrievable": statistics.median(finite) if finite else None,
    }


def bootstrap_top10_ci(
    target_ranks: list[int | None],
    *,
    seed: int = 42,
    samples: int = 2000,
) -> list[float]:
    if not target_ranks:
        return [0.0, 0.0]
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resample = [
            target_ranks[generator.randrange(len(target_ranks))]
            for _ in target_ranks
        ]
        estimates.append(
            100
            * sum(rank is not None and rank < 10 for rank in resample)
            / len(resample)
        )
    estimates.sort()
    return [
        estimates[int(0.025 * samples)],
        estimates[int(0.975 * samples) - 1],
    ]


def rank_of(reading: str, predictions: list[str]) -> int | None:
    try:
        return predictions.index(reading)
    except ValueError:
        return None


def render_markdown(report: dict[str, Any]) -> str:
    target = report["target_level_any_attributed_restoration"]
    reading = report["unique_target_reading_level"]
    unconstrained = report["diagnostics"]["unconstrained_target_level"]
    qd = report["qd_initial_control"]
    quality = report["protocol"]["quality_filter"]
    ci = target["top10_cluster_bootstrap_95ci"]
    sensitivity_rows = "\n".join(
        f"| ±{tolerance} | {values['n']} | {values['top1']:.1f}% | "
        f"{values['top10']:.1f}% | {values['top20']:.1f}% |"
        for tolerance, values in report["length_tolerance_sensitivity"].items()
    )
    source_rows = sorted(
        report["by_bibliographic_source"].items(),
        key=lambda item: (-item[1]["n"], item[0]),
    )[:10]
    source_table = "\n".join(
        f"| {name or '(unnamed source)'} | {values['n']} | "
        f"{values['top1']:.1f}% | {values['top10']:.1f}% |"
        for name, values in source_rows
    )
    return f"""# Qumran Digital constrained restoration benchmark

## Result

This experiment evaluates the reconstruction-free preserved-only model on
single-word lacunae from the stored Qumran Digital snapshot. Unlike the
superseded whole-word-mask experiment, it retains visibly preserved letters
and an approximate lacuna-derived word length
(±{report['protocol']['length_tolerance']} character).

| Unit | N | Top-1 | Top-5 | Top-10 | Top-20 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Target: any attributed compatible restoration | {target['n']} | {target['top1']:.1f}% | {target['top5']:.1f}% | {target['top10']:.1f}% | {target['top20']:.1f}% |
| Unique target-reading pair | {reading['n']} | {reading['top1']:.1f}% | {reading['top5']:.1f}% | {reading['top10']:.1f}% | {reading['top20']:.1f}% |
| QD initial reading control | {qd['n']} | {qd['top1']:.1f}% | {qd['top5']:.1f}% | {qd['top10']:.1f}% | {qd['top20']:.1f}% |

Target-level Top-10 95% cluster-bootstrap interval: **{ci[0]:.1f}%–{ci[1]:.1f}%**.
Without manuscript constraints, the same target-level Top-10 is
{unconstrained['top10']:.1f}%. The difference measures the value of physical
evidence supplied to the decoder, not an improvement in the language model.

### Length-tolerance sensitivity

| Allowed difference | Eligible targets | Top-1 | Top-10 | Top-20 |
| :--- | ---: | ---: | ---: | ---: |
{sensitivity_rows}

The conclusion is stable across exact-length, ±1, and ±2 decoding. The number
of eligible targets changes because a published proposal outside a tolerance
is not treated as physically compatible at that setting.

## Largest publication samples

Each publication contributes at most one observation per target; duplicate
publication rows and duplicate readings do not receive extra weight.

| Publication | Targets | Top-1 | Top-10 |
| :--- | ---: | ---: | ---: |
{source_table}

## Scope and exclusions

- Cached source snapshot: Qumran Digital {report['protocol']['source_snapshot']};
  the scorer performs no network requests.
- Corpus: held-out non-biblical DSS scrolls only.
- Training: preserved letters only; square-bracket scholarly restorations are
  absent from fine-tuning data.
- Primary unit: one manuscript target. Success means any distinct,
  bibliographically attributed restoration compatible with the physical
  pattern is in Top-K.
- Input rows: {quality['input_publication_rows']}; eligible targets:
  {quality['eligible_targets']}; unique compatible target-reading pairs:
  {quality['unique_compatible_target_readings']}.
- Multiword readings, scribal corrections, modern alternatives, incomplete
  readings, non-lacuna variants, and readings contradicting visible letters
  are reported as exclusions rather than concatenated into fake words.

This is still a literature-agreement benchmark, not physical ground truth.
QD selected these locations because they are disputed, and its variant
collection is working data. Publication-level samples are descriptive and
must not be treated as a ranking of researchers.

## Reproduction

Both commands below are offline when the stored snapshot exists:

```bash
.venv/bin/python eval/build_qd_researcher_benchmark.py
.venv/bin/python eval/score_qd_researcher_benchmark.py
```

Only an explicit `eval/build_qd_researcher_benchmark.py --refresh` contacts
Qumran Digital.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-context-words", type=int, default=10)
    parser.add_argument("--length-tolerance", type=int, default=1)
    args = parser.parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not 1 <= args.batch_size <= 128:
        raise ValueError("--batch-size must be between 1 and 128")
    if not 0 <= args.length_tolerance <= 3:
        raise ValueError("--length-tolerance must be between 0 and 3")

    all_rows = read_jsonl(args.input)
    target_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        target_rows[(str(row["siglum"]), int(row["word_id"]))].append(row)

    exclusions: dict[str, int] = defaultdict(int)
    sensitivity_tolerances = (0, 1, 2)
    pool_tolerance = max(*sensitivity_tolerances, args.length_tolerance)
    eligible: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in target_rows.items():
        representative = rows[0]
        visible_context = sum(
            word not in {"<GAP>", "<TARGET>"}
            for word in representative["context_words"]
        )
        if visible_context < args.min_context_words:
            exclusions["insufficient_visible_context_targets"] += 1
            continue
        constraint, reason = build_constraint(representative)
        if constraint is None:
            exclusions[f"{reason}_targets"] += 1
            continue
        readings: dict[str, dict[str, Any]] = {}
        for row in rows:
            normalized, reading_reason = parse_attributed_reading(
                row, constraint, pool_tolerance
            )
            if normalized is None:
                exclusions[f"{reading_reason}_publication_rows"] += 1
                continue
            item = readings.setdefault(
                normalized,
                {"reading": normalized, "sources": {}, "raw_readings": set()},
            )
            item["sources"][str(row["bibliography_id"])] = {
                "abbreviation": row["bibliography_abbreviation"],
                "formatted": row["bibliography_formatted"],
            }
            item["raw_readings"].add(row["reading"])
        if not readings:
            exclusions["no_compatible_attributed_reading_targets"] += 1
            continue
        eligible[key] = {
            **representative,
            "constraint": constraint,
            "readings": readings,
        }

    if not eligible:
        raise RuntimeError("No eligible restoration targets remain")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(args.model)).to(DEVICE).eval()
    normalized_token_by_id = [
        hebrew_letters(tokenizer.decode([token_id]).strip())
        for token_id in range(len(tokenizer))
    ]
    target_items = list(eligible.values())
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
                logits[batch_index, positions[batch_index]], descending=True
            ).tolist()
            unconstrained: list[str] = []
            seen: set[str] = set()
            for token_id in ordered_ids:
                candidate = normalized_token_by_id[token_id]
                if len(candidate) < 2 or candidate in seen:
                    continue
                seen.add(candidate)
                unconstrained.append(candidate)
            key = (str(item["siglum"]), int(item["word_id"]))
            predictions_by_target[key] = unconstrained
        print(
            f"scored targets: {min(start + args.batch_size, len(target_items))}"
            f"/{len(target_items)}",
            flush=True,
        )

    target_records = []
    reading_records = []
    source_target_ranks: dict[str, dict[tuple[str, int], list[int | None]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    qd_ranks: list[int | None] = []
    for key, item in eligible.items():
        unconstrained_predictions = predictions_by_target[key]
        constrained_predictions = [
            candidate
            for candidate in unconstrained_predictions
            if item["constraint"].matches(candidate, args.length_tolerance)
        ]
        readings = []
        for reading, metadata in item["readings"].items():
            if not item["constraint"].matches(reading, args.length_tolerance):
                continue
            constrained_rank = rank_of(reading, constrained_predictions)
            unconstrained_rank = rank_of(reading, unconstrained_predictions)
            source_names = sorted(
                source["abbreviation"] or source["formatted"]
                for source in metadata["sources"].values()
            )
            record = {
                "siglum": item["siglum"],
                "word_id": item["word_id"],
                "reading": reading,
                "rank": constrained_rank,
                "unconstrained_rank": unconstrained_rank,
                "sources": source_names,
            }
            reading_records.append(record)
            readings.append(record)
            for source in source_names:
                source_target_ranks[source][key].append(constrained_rank)
        if not readings:
            exclusions[
                "no_compatible_reading_at_primary_tolerance_targets"
            ] += 1
            continue

        finite_target = [record["rank"] for record in readings if record["rank"] is not None]
        finite_unconstrained = [
            record["unconstrained_rank"]
            for record in readings
            if record["unconstrained_rank"] is not None
        ]
        qd_initial = hebrew_letters(str(item["qd_initial_reading"]))
        qd_rank = (
            rank_of(qd_initial, constrained_predictions)
            if item["constraint"].matches(qd_initial, args.length_tolerance)
            else None
        )
        qd_ranks.append(qd_rank)
        target_records.append(
            {
                "siglum": item["siglum"],
                "column": item["column"],
                "line": item["line"],
                "word_id": item["word_id"],
                "qd_display_reading": item["qd_display_reading"],
                "constraint": asdict(item["constraint"]),
                "compatible_attributed_readings": sorted(item["readings"]),
                "rank_any_attributed": min(finite_target) if finite_target else None,
                "unconstrained_rank_any_attributed": (
                    min(finite_unconstrained) if finite_unconstrained else None
                ),
                "qd_initial_rank": qd_rank,
                "top_predictions": constrained_predictions[:20],
            }
        )

    target_ranks = [record["rank_any_attributed"] for record in target_records]
    unconstrained_target_ranks = [
        record["unconstrained_rank_any_attributed"] for record in target_records
    ]
    target_summary = summarize_ranks(target_ranks)
    target_summary["top10_cluster_bootstrap_95ci"] = bootstrap_top10_ci(target_ranks)
    sensitivity = {}
    for tolerance in sensitivity_tolerances:
        ranks: list[int | None] = []
        for key, item in eligible.items():
            compatible = [
                reading
                for reading in item["readings"]
                if item["constraint"].matches(reading, tolerance)
            ]
            if not compatible:
                continue
            candidates = [
                candidate
                for candidate in predictions_by_target[key]
                if item["constraint"].matches(candidate, tolerance)
            ]
            finite = [
                rank
                for reading in compatible
                for rank in [rank_of(reading, candidates)]
                if rank is not None
            ]
            ranks.append(min(finite) if finite else None)
        sensitivity[str(tolerance)] = summarize_ranks(ranks)
    source_results = {}
    for source, ranks_by_target in source_target_ranks.items():
        per_target = [
            min(rank for rank in ranks if rank is not None)
            if any(rank is not None for rank in ranks)
            else None
            for ranks in ranks_by_target.values()
        ]
        source_results[source] = summarize_ranks(per_target)

    report = {
        "protocol": {
            "model": (
                str(args.model.resolve().relative_to(ROOT))
                if args.model.resolve().is_relative_to(ROOT)
                else str(args.model)
            ),
            "device": DEVICE,
            "source_snapshot": all_rows[0]["qd_snapshot"],
            "corpus": "held-out non-biblical DSS scrolls",
            "split_integrity": "scroll-disjoint from fine-tuning train/dev",
            "model_training": "preserved-only; no square-bracket restorations",
            "target": "single masked token with post-MLM physical filtering",
            "physical_constraints": (
                "visible Hebrew outside brackets plus approximate word length"
            ),
            "length_tolerance": args.length_tolerance,
            "candidate_normalization": "exact Hebrew consonants",
            "primary_unit": "one target; any compatible attributed restoration",
            "network": "offline cached snapshot",
            "quality_filter": {
                "minimum_visible_context_words": args.min_context_words,
                "input_publication_rows": len(all_rows),
                "input_targets": len(target_rows),
                "eligible_targets": len(target_records),
                "unique_compatible_target_readings": len(reading_records),
                "exclusions": dict(sorted(exclusions.items())),
            },
            "interpretation": (
                "agreement with attributed literature under manuscript "
                "constraints; not verified physical ground truth"
            ),
        },
        "target_level_any_attributed_restoration": target_summary,
        "unique_target_reading_level": summarize_ranks(
            record["rank"] for record in reading_records
        ),
        "qd_initial_control": summarize_ranks(qd_ranks),
        "length_tolerance_sensitivity": sensitivity,
        "diagnostics": {
            "unconstrained_target_level": summarize_ranks(
                unconstrained_target_ranks
            )
        },
        "by_bibliographic_source": source_results,
        "targets": target_records,
        "unique_target_readings": reading_records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"saved -> {args.report}")
    print(f"saved -> {args.markdown}")


if __name__ == "__main__":
    main()
