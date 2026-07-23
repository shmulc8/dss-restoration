"""Benchmark retrieval as a strict auxiliary signal on single-word cases.

Inputs:
- case export with real model top-5 predictions
- fit-only similar passages export

Outputs:
1. Retrieval coverage: how often the gold word appears in the retrieved passages.
2. A fixed retrieval-assisted rerank over the model's top-5 candidates.

This benchmark is intentionally conservative:
- retrieval uses only the allowed source partition already baked into the JSON
- reranking only reorders existing model candidates; it never injects the gold
"""
import csv
import json
import os
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "analysis" / "reports"

from utils.composition_lookup import composition_group_for_scroll

CASES_CSV = Path(os.environ.get(
    "CASES_CSV",
    REPORTS / "full_single_word_cases_refined_hebrew_only.csv",
))
SIMILAR_JSON = Path(os.environ.get(
    "SIMILAR_JSON",
    REPORTS / "similar_passages_fit_for_full_single_word_cases.json",
))
OUT_JSON = Path(os.environ.get(
    "OUT_JSON",
    REPORTS / "retrieval_benchmark_single_word.json",
))


def load_inputs():
    with CASES_CSV.open() as fh:
        cases = list(csv.DictReader(fh))
    similar = json.loads(SIMILAR_JSON.read_text(encoding="utf-8"))
    return cases, similar


def top_candidates(row):
    return [row.get(f"model_top{i}", "") for i in range(1, 6) if row.get(f"model_top{i}", "")]


def support_scores(row, passages):
    candidates = top_candidates(row)
    support = {candidate: 0.0 for candidate in candidates}
    hit_count = {candidate: 0 for candidate in candidates}
    for passage in passages:
        score = float(passage.get("score", 0.0))
        hits = set(passage.get("candidate_hits", []))
        for candidate in candidates:
            if candidate in hits:
                support[candidate] += score
                hit_count[candidate] += 1
    return support, hit_count


def rerank(row, passages):
    candidates = top_candidates(row)
    support, hit_count = support_scores(row, passages)
    original_rank = {candidate: idx for idx, candidate in enumerate(candidates)}
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            support[candidate] > 0,
            support[candidate],
            hit_count[candidate],
            -original_rank[candidate],
        ),
        reverse=True,
    )
    return ranked, support, hit_count


def pct(num, den):
    return 0.0 if den == 0 else num / den * 100.0


def evaluate(name, cases, similar, include_passage):
    total = len(cases)
    misses = 0
    cases_with_passages = 0
    baseline_top1 = 0
    baseline_top5 = 0
    rerank_top1 = 0
    rerank_top5 = 0
    improved_top1 = 0
    hurt_top1 = 0
    gold_in_similar_all = 0
    gold_in_similar_miss = 0
    rerank_ceiling_top1 = 0
    supported_top1 = 0

    for row in cases:
        gold = row["target_word"]
        candidates = top_candidates(row)
        case_group = composition_group_for_scroll(row.get("scroll", ""))
        passages = [
            passage
            for passage in similar.get(str(row["row_id"]), [])
            if include_passage(passage, case_group)
        ]
        if passages:
            cases_with_passages += 1
        gold_in_similar = any(passage.get("gold_present") for passage in passages)
        if gold_in_similar:
            gold_in_similar_all += 1

        baseline_hit1 = bool(candidates) and candidates[0] == gold
        baseline_hit5 = gold in candidates
        baseline_top1 += baseline_hit1
        baseline_top5 += baseline_hit5

        if not baseline_hit5:
            misses += 1
            if gold_in_similar:
                gold_in_similar_miss += 1

        reranked, support, hit_count = rerank(row, passages)
        rerank_hit1 = bool(reranked) and reranked[0] == gold
        rerank_hit5 = gold in reranked
        rerank_top1 += rerank_hit1
        rerank_top5 += rerank_hit5

        if rerank_hit1 and not baseline_hit1:
            improved_top1 += 1
        if baseline_hit1 and not rerank_hit1:
            hurt_top1 += 1

        if baseline_hit5 and gold_in_similar:
            rerank_ceiling_top1 += 1
        if candidates:
            supported_top1 += support.get(candidates[0], 0.0) > 0

    return {
        "condition": name,
        "coverage": {
            "cases": total,
            "cases_with_passages": cases_with_passages,
            "cases_with_passages_pct": round(pct(cases_with_passages, total), 1),
            "misses": misses,
            "gold_present_all_count": gold_in_similar_all,
            "gold_present_all_pct": round(pct(gold_in_similar_all, total), 1),
            "gold_present_miss_count": gold_in_similar_miss,
            "gold_present_miss_pct": round(pct(gold_in_similar_miss, misses), 1),
        },
        "rerank": {
            "baseline_top1_count": baseline_top1,
            "baseline_top1_pct": round(pct(baseline_top1, total), 1),
            "reranked_top1_count": rerank_top1,
            "reranked_top1_pct": round(pct(rerank_top1, total), 1),
            "delta_cases": rerank_top1 - baseline_top1,
            "delta_pts": round(pct(rerank_top1, total) - pct(baseline_top1, total), 1),
            "improved_top1_cases": improved_top1,
            "hurt_top1_cases": hurt_top1,
            "baseline_top5_count": baseline_top5,
            "baseline_top5_pct": round(pct(baseline_top5, total), 1),
            "reranked_top5_count": rerank_top5,
            "reranked_top5_pct": round(pct(rerank_top5, total), 1),
            "supported_top1_count": supported_top1,
            "supported_top1_pct": round(pct(supported_top1, total), 1),
            "rerank_ceiling_top1_count": rerank_ceiling_top1,
            "rerank_ceiling_top1_pct": round(pct(rerank_ceiling_top1, total), 1),
        },
    }

def main():
    cases, similar = load_inputs()

    results = {
        "fit_any_composition": evaluate(
            "fit_any_composition",
            cases,
            similar,
            include_passage=lambda passage, case_group: True,
        ),
        "fit_cross_composition_only": evaluate(
            "fit_cross_composition_only",
            cases,
            similar,
            include_passage=lambda passage, case_group: not passage.get("same_composition", False),
        ),
    }

    result = {
        "cases": len(cases),
        "conditions": results,
    }

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for key, section in results.items():
        coverage = section["coverage"]
        rerank = section["rerank"]
        print(f"=== {key} ===")
        print(f"cases with retrieved passages: {coverage['cases_with_passages']}/{coverage['cases']} = {coverage['cases_with_passages_pct']:4.1f}%")
        print(f"gold present in retrieved passages (all cases):   {coverage['gold_present_all_count']}/{coverage['cases']} = {coverage['gold_present_all_pct']:4.1f}%")
        print(f"gold present in retrieved passages (top-5 misses): {coverage['gold_present_miss_count']}/{coverage['misses']} = {coverage['gold_present_miss_pct']:4.1f}%")
        print(f"baseline exact top-1: {rerank['baseline_top1_count']}/{coverage['cases']} = {rerank['baseline_top1_pct']:4.1f}%")
        print(f"reranked exact top-1: {rerank['reranked_top1_count']}/{coverage['cases']} = {rerank['reranked_top1_pct']:4.1f}%")
        print(f"net top-1 delta: {rerank['delta_cases']:+d} cases ({rerank['delta_pts']:+.1f} pts)")
        print(f"improved top-1 cases: {rerank['improved_top1_cases']}")
        print(f"hurt top-1 cases:     {rerank['hurt_top1_cases']}")
        print()
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
