"""Prepare static JSON assets for the local demo site."""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "analysis" / "reports"
DEMO_DATA = ROOT / "demo" / "data"

from utils.composition_lookup import composition_group_for_scroll

FAILURES_CSV = REPORTS / "full_single_word_cases_refined_hebrew_only.csv"
SPANS_CSV = REPORTS / "full_multi_word_cases_refined_hebrew_only.csv"
UNKNOWN_CSV = REPORTS / "full_unknown_gap_cases_hebrew_only.csv"
FAILURES_GUIDE = REPORTS / "heldout_failures_refined_hebrew_only_reader_friendly.md"
BENCHMARK_TXT = REPORTS / "heldout_gaplen_sampled_hebrew_only.txt"
SIMILAR_JSON = REPORTS / "similar_passages_fit_for_full_single_word_cases.json"
RETRIEVAL_BENCHMARK_JSON = REPORTS / "retrieval_benchmark_single_word.json"
PARALLEL_LOOKUP_JSON = REPORTS / "parallel_lookup_benchmark_single_word.json"


def load_failures():
    with FAILURES_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    similar = {}
    if SIMILAR_JSON.exists():
        similar = json.loads(SIMILAR_JSON.read_text(encoding="utf-8"))
    for idx, row in enumerate(rows, start=1):
        row["id"] = f"case-{idx}"
        row["composition_group"] = composition_group_for_scroll(row.get("scroll", ""))
        try:
            row["target_fit_frequency"] = int(row["target_fit_frequency"])
        except Exception:
            row["target_fit_frequency"] = None
        row["top_candidates"] = [row.get(f"model_top{i}", "") for i in range(1, 6)]
        row["has_short_top1"] = bool(row["model_top1"]) and len(row["model_top1"]) + 1 < len(row["target_word"])
        row["rarity_bucket"] = (
            "unseen" if row["target_fit_frequency"] == 0 else
            "rare" if row["target_fit_frequency"] is not None and row["target_fit_frequency"] <= 3 else
            "medium" if row["target_fit_frequency"] is not None and row["target_fit_frequency"] <= 20 else
            "common"
        )
        issue_map = {
            "exact_top1_hit": "Exact top-1 hit",
            "exact_top5_hit": "Exact top-5 hit",
            "particle_or_too_short": "Split-clitic noise / short-word bias",
            "rare_or_unseen": "Rare or unseen target word",
            "same_length_semantic": "Plausible but wrong content-word choice",
            "other_miss": "Needs manual inspection",
        }
        row["likely_issue_label"] = issue_map.get(row["likely_issue"], row["likely_issue"])
        row["similar_passages"] = similar.get(row.get("row_id", ""), [])
        for passage in row["similar_passages"]:
            passage["same_composition"] = passage.get("same_composition", False) or (
                composition_group_for_scroll(passage.get("book", "")) == row["composition_group"]
            )
        row["gold_in_similar"] = any(passage.get("gold_present") for passage in row["similar_passages"])
        row["top1_in_similar"] = any(passage.get("top1_present") for passage in row["similar_passages"])
    return rows


def load_spans():
    if not SPANS_CSV.exists():
        return []
    with SPANS_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for idx, row in enumerate(rows, start=1):
        row["id"] = f"span-{idx}"
        row["composition_group"] = composition_group_for_scroll(row.get("scroll", ""))
        row["gap_length"] = int(row["gap_length"])
        row["slot_top1_hits"] = int(row["slot_top1_hits"])
        row["slot_top5_hits"] = int(row["slot_top5_hits"])
        row["target_words"] = json.loads(row["target_words"])
        row["target_fit_frequencies"] = json.loads(row["target_fit_frequencies"])
        row["top5_phrases"] = json.loads(row["top5_phrases"])
        row["slot_details"] = json.loads(row["slot_details_json"])
        row["top_candidates"] = [slot.get("top1", "") for slot in row["slot_details"]]
        row["likely_issue_label"] = (
            "Exact span hit" if row["case_status"] == "hit" else
            "Partial span recovery" if row["slot_top5_hits"] > 0 else
            "Span miss"
        )
    return rows


def load_unknowns():
    if not UNKNOWN_CSV.exists():
        return []
    with UNKNOWN_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for idx, row in enumerate(rows, start=1):
        row["id"] = f"unknown-{idx}"
        row["composition_group"] = composition_group_for_scroll(row.get("scroll", ""))
        row["run_length"] = int(row["run_length"])
        row["start_index"] = int(row["start_index"])
        row["end_index"] = int(row["end_index"])
        row["flag_list"] = [flag for flag in row.get("flags", "").split(" | ") if flag]
        row["likely_issue_label"] = row["category"]
        row["case_status"] = "unknown"
        row["top1_phrase"] = row.get("top1_phrase", "")
        row["top1_constrained_phrase"] = row.get("top1_constrained_phrase", "")
        row["slot_details"] = json.loads(row["slot_details_json"]) if row.get("slot_details_json") else []
    return rows


def load_guide():
    return FAILURES_GUIDE.read_text(encoding="utf-8")


def parse_metric_section(lines, section_name):
    section = {}
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"=== {section_name} by gap length (words) ===":
            start = i + 2
            break
    if start is None:
        return section
    for line in lines[start:]:
        raw = line.rstrip()
        if not raw:
            break
        if raw.startswith("n per bucket:"):
            break
        if raw.startswith("==="):
            break
        if raw.startswith("model"):
            continue
        parts = raw.split()
        if len(parts) < 6:
            continue
        label = " ".join(parts[:-5])
        values = [float(part.rstrip("%")) for part in parts[-5:]]
        section[label] = {
            "1": values[0],
            "2": values[1],
            "3": values[2],
            "4-5": values[3],
            "6+": values[4],
        }
    return section


def load_benchmark():
    text = BENCHMARK_TXT.read_text(encoding="utf-8")
    lines = text.splitlines()
    benchmark = {
        "raw_text": text,
        "top_1": parse_metric_section(lines, "top-1"),
        "top_5": parse_metric_section(lines, "top-5"),
        "top_10": parse_metric_section(lines, "top-10"),
        "top_20": parse_metric_section(lines, "top-20"),
        "n_per_bucket": {},
    }
    match = re.search(r"n per bucket:\s*\{(.+)\}", text)
    if match:
        payload = match.group(1)
        for key, value in re.findall(r"'([^']+)':\s*(\d+)", payload):
            benchmark["n_per_bucket"][key] = int(value)
    if RETRIEVAL_BENCHMARK_JSON.exists():
        benchmark["retrieval"] = json.loads(RETRIEVAL_BENCHMARK_JSON.read_text(encoding="utf-8"))
    else:
        benchmark["retrieval"] = None
    if PARALLEL_LOOKUP_JSON.exists():
        benchmark["parallel_lookup"] = json.loads(PARALLEL_LOOKUP_JSON.read_text(encoding="utf-8"))
    else:
        benchmark["parallel_lookup"] = None
    return benchmark


def build_summary(rows):
    by_issue = Counter(row["likely_issue_label"] for row in rows)
    by_rarity = Counter(row["rarity_bucket"] for row in rows)
    by_status = Counter(row["case_status"] for row in rows)
    short_top1 = sum(1 for row in rows if row["has_short_top1"])
    gold_in_similar = sum(1 for row in rows if row["gold_in_similar"])
    top1_in_similar = sum(1 for row in rows if row["top1_in_similar"])
    return {
        "total_cases": len(rows),
        "issues": by_issue,
        "rarity": by_rarity,
        "status": by_status,
        "short_top1_count": short_top1,
        "gold_in_similar_count": gold_in_similar,
        "top1_in_similar_count": top1_in_similar,
    }


def main():
    DEMO_DATA.mkdir(parents=True, exist_ok=True)
    failures = load_failures()
    spans = load_spans()
    unknowns = load_unknowns()
    benchmark = load_benchmark()
    guide = load_guide()
    summary = build_summary(failures)

    (DEMO_DATA / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DEMO_DATA / "spans.json").write_text(
        json.dumps(spans, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DEMO_DATA / "unknowns.json").write_text(
        json.dumps(unknowns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DEMO_DATA / "benchmark.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DEMO_DATA / "guide.json").write_text(
        json.dumps({"markdown": guide, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {DEMO_DATA / 'failures.json'}")
    print(f"wrote {DEMO_DATA / 'spans.json'}")
    print(f"wrote {DEMO_DATA / 'unknowns.json'}")
    print(f"wrote {DEMO_DATA / 'benchmark.json'}")
    print(f"wrote {DEMO_DATA / 'guide.json'}")


if __name__ == "__main__":
    main()
