"""Fail closed when the locked paper protocol loses essential safeguards."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "eval" / "paper_protocol_v1.json"


def validate(protocol: dict) -> list[str]:
    errors: list[str] = []
    if protocol.get("primary_metric") != "exact_complete_span_top10":
        errors.append("primary metric must be exact_complete_span_top10")
    condition = protocol.get("primary_condition", {})
    for field in ("length", "word_count", "word_boundaries"):
        if condition.get(field) != "unknown":
            errors.append(f"primary {field} must be unknown")
    if condition.get("failed_decode") != "count_as_miss":
        errors.append("failed decodes must count as misses")
    splits = protocol.get("splits", {})
    if splits.get("primary") != "scroll_disjoint":
        errors.append("primary split must be scroll-disjoint")
    if splits.get("retrieval_index") != "training_scrolls_only":
        errors.append("retrieval must be train-only")
    if splits.get("test_used_for_selection") is not False:
        errors.append("test data cannot be used for model selection")
    training = protocol.get("training", {})
    if len(set(training.get("seeds", []))) < 3:
        errors.append("at least three unique training seeds are required")
    required_diagnostics = set(protocol.get("mandatory_diagnostics", []))
    for name in ("tokenizer_word_coverage", "tokenizer_complete_span_coverage"):
        if name not in required_diagnostics:
            errors.append(f"missing mandatory diagnostic: {name}")
    gate = set(protocol.get("promotion_gate", {}).get("paper_result_requires", []))
    for requirement in (
        "clustered_confidence_intervals",
        "exact_multiword_sequence_scoring",
        "unknown_length_primary_condition",
    ):
        if requirement not in gate:
            errors.append(f"missing promotion gate: {requirement}")
    return errors


def main() -> int:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    errors = validate(protocol)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"paper protocol valid: {DEFAULT_PROTOCOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
