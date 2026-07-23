"""Benchmark exact parallel lookup under contamination-aware settings.

Two conditions are reported:
1. `strict_preserved_only`
   - retrieval index built from fit scrolls only
   - target word must be preserved in the source passage
   - all context words used for matching must also be preserved
2. `relaxed_preserved_target_only`
   - retrieval index built from fit scrolls only
   - target word must be preserved in the source passage
   - context words may include reconstructed tokens

This separates clean evidence from reconstruction-assisted evidence.
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tf.fabric import Fabric

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.book_filters import resolve_book_exclusions
from utils.composition_lookup import composition_group_for_scroll
from utils.eval_split import resolve_scroll_filter

REPORTS = ROOT / "analysis" / "reports"
CASES_CSV = Path(os.environ.get(
    "CASES_CSV",
    REPORTS / "full_single_word_cases_refined_hebrew_only.csv",
))
OUT_JSON = Path(os.environ.get(
    "OUT_JSON",
    REPORTS / "parallel_lookup_benchmark_single_word.json",
))
WINDOWS = [int(part) for part in os.environ.get("WINDOWS", "5,4,3,2").split(",") if part.strip()]
BOOK_FILTER_MODE = os.environ.get("BOOK_FILTER_MODE", "no-aram")

HEB = set(chr(c) for c in range(0x05D0, 0x05EB))


def load_cases():
    with CASES_CSV.open() as fh:
        return list(csv.DictReader(fh))


def load_fit_scroll_words():
    allowed_scrolls, _ = resolve_scroll_filter("fit")
    excluded_books, _ = resolve_book_exclusions(BOOK_FILTER_MODE)
    tf_dir = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
    tf = Fabric(locations=str(tf_dir), silent="deep")
    api = tf.load("otype glyph rec biblical scroll", silent="deep")
    if api is False:
        raise RuntimeError(f"Could not load cached DSS corpus from {tf_dir}")
    F, L = api.F, api.L

    def winfo(word_node):
        signs = L.d(word_node, "sign")
        glyph = "".join(F.glyph.v(sign) or "" for sign in signs)
        recs = [F.rec.v(sign) for sign in signs]
        reconstructed = bool(signs) and all(r == 1 for r in recs)
        preserved = bool(signs) and all(r != 1 for r in recs)
        return glyph, reconstructed, preserved

    scrolls = {}
    for word_node in F.otype.s("word"):
        if F.biblical.v(word_node):
            continue
        scroll = L.u(word_node, "scroll")
        if not scroll:
            continue
        scroll_name = F.scroll.v(scroll[0])
        if allowed_scrolls is not None and scroll_name not in allowed_scrolls:
            continue
        if scroll_name in excluded_books:
            continue
        glyph, reconstructed, preserved = winfo(word_node)
        if len(glyph) < 1 or any(ch not in HEB for ch in glyph):
            continue
        bucket = scrolls.setdefault(scroll_name, {
            "group": composition_group_for_scroll(scroll_name),
            "words": [],
        })
        bucket["words"].append({
            "word": glyph,
            "reconstructed": reconstructed,
            "preserved": preserved,
        })
    return scrolls


def build_indexes(scrolls):
    strict = {window: defaultdict(lambda: defaultdict(Counter)) for window in WINDOWS}
    relaxed = {window: defaultdict(lambda: defaultdict(Counter)) for window in WINDOWS}

    for scroll_data in scrolls.values():
        group = scroll_data["group"]
        words = scroll_data["words"]
        for i, item in enumerate(words):
            if not item["preserved"]:
                continue
            for window in WINDOWS:
                if i - window < 0 or i + window >= len(words):
                    continue
                left_items = words[i - window:i]
                right_items = words[i + 1:i + 1 + window]
                left = tuple(entry["word"] for entry in left_items)
                right = tuple(entry["word"] for entry in right_items)
                relaxed[window][(left, right)][group][item["word"]] += 1
                if all(entry["preserved"] for entry in left_items + right_items):
                    strict[window][(left, right)][group][item["word"]] += 1
    return strict, relaxed


def parse_raw_context(raw_context):
    words = raw_context.split()
    target_pos = words.index("⬚⬚⬚")
    return words, target_pos


def predict(indexes, words, target_pos, excluded_group=None):
    for window in WINDOWS:
        if target_pos - window < 0 or target_pos + window >= len(words):
            continue
        left = tuple(words[target_pos - window:target_pos])
        right = tuple(words[target_pos + 1:target_pos + 1 + window])
        hits_by_group = indexes[window].get((left, right))
        if not hits_by_group:
            continue
        hits = Counter()
        source_groups = 0
        for group, counter in hits_by_group.items():
            if excluded_group is not None and group == excluded_group:
                continue
            hits.update(counter)
            source_groups += 1
        if not hits:
            continue
        predicted, support_count = hits.most_common(1)[0]
        return {
            "window": window,
            "predicted": predicted,
            "support_count": support_count,
            "candidate_counts": hits.most_common(5),
            "source_groups": source_groups,
        }
    return None


def pct(num, den):
    return 0.0 if den == 0 else num / den * 100.0


def evaluate(name, indexes, cases, exclude_same_group=False):
    total = len(cases)
    matched = 0
    correct = 0
    by_window = Counter()
    support_counts = []
    source_group_counts = []
    examples = []

    for row in cases:
        words, target_pos = parse_raw_context(row["raw_context"])
        case_group = composition_group_for_scroll(row["scroll"])
        result = predict(
            indexes,
            words,
            target_pos,
            excluded_group=case_group if exclude_same_group else None,
        )
        if not result:
            continue
        matched += 1
        by_window[result["window"]] += 1
        support_counts.append(result["support_count"])
        source_group_counts.append(result["source_groups"])
        hit = result["predicted"] == row["target_word"]
        correct += int(hit)
        if len(examples) < 12:
            examples.append({
                "row_id": row["row_id"],
                "scroll": row["scroll"],
                "excluded_group": case_group if exclude_same_group else None,
                "gold": row["target_word"],
                "predicted": result["predicted"],
                "hit": hit,
                "window": result["window"],
                "support_count": result["support_count"],
                "candidate_counts": result["candidate_counts"],
                "source_groups": result["source_groups"],
                "context": row["context_for_reading"],
            })

    return {
        "condition": name,
        "total_cases": total,
        "matched_cases": matched,
        "matched_pct": round(pct(matched, total), 1),
        "correct_cases": correct,
        "correct_pct_over_all": round(pct(correct, total), 1),
        "correct_pct_over_matched": round(pct(correct, matched), 1),
        "window_counts": dict(by_window),
        "mean_support_count": round(sum(support_counts) / len(support_counts), 2) if support_counts else 0.0,
        "mean_source_groups": round(sum(source_group_counts) / len(source_group_counts), 2) if source_group_counts else 0.0,
        "examples": examples,
    }


def main():
    cases = load_cases()
    scrolls = load_fit_scroll_words()
    strict, relaxed = build_indexes(scrolls)

    results = {
        "strict_preserved_any_composition": evaluate("strict_preserved_any_composition", strict, cases),
        "strict_preserved_cross_composition": evaluate("strict_preserved_cross_composition", strict, cases, exclude_same_group=True),
        "relaxed_preserved_target_any_composition": evaluate("relaxed_preserved_target_any_composition", relaxed, cases),
        "relaxed_preserved_target_cross_composition": evaluate("relaxed_preserved_target_cross_composition", relaxed, cases, exclude_same_group=True),
    }

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for key, result in results.items():
        print(f"=== {key} ===")
        print(f"matched exact-parallel cases: {result['matched_cases']}/{result['total_cases']} = {result['matched_pct']:4.1f}%")
        print(f"correct over all cases:       {result['correct_cases']}/{result['total_cases']} = {result['correct_pct_over_all']:4.1f}%")
        print(f"correct over matched cases:   {result['correct_cases']}/{result['matched_cases']} = {result['correct_pct_over_matched']:4.1f}%")
        print(f"window counts: {result['window_counts']}")
        print(f"mean support count: {result['mean_support_count']}")
        print(f"mean source groups: {result['mean_source_groups']}")
        print()
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
