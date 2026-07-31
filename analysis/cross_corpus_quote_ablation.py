"""Decompose DSS cross-corpus matches into quotation and residual affinity.

The literal condition ranks preserved DSS passages against an external Hebrew
shelf. The residual condition first masks every DSS token participating in an
exact external three-word match, then repeats the same ranking. Persistence
after this deliberately severe ablation is a hypothesis for non-verbatim
affinity; it is not proof of borrowing, authorship, or direction of influence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.cross_corpus_connections import (
    DEFAULT_TARGETS,
    DEFAULT_TF_ROOT,
    PESHER_SOURCE_BOOKS,
    QUMRAN_EXTRA_BOOKS,
    SOURCE_WEIGHTS,
    Passage,
    _tfidf_similarities,
    evaluate_pesher_source_control,
    load_preserved_dss_queries,
    load_tf_passages,
    row_percentiles,
    sha256_files,
    source_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
QUOTE_GAP = "__QUOTE_GAP__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dss-csv", type=Path, default=ROOT / "dss_chunks.csv")
    parser.add_argument(
        "--dss-preserved-jsonl",
        type=Path,
        default=ROOT / "data" / "derived" / "preserved_nonbib_chunks.jsonl",
    )
    parser.add_argument(
        "--bhsa-path",
        type=Path,
        default=DEFAULT_TF_ROOT / "bhsa" / "tf" / "2021",
    )
    parser.add_argument(
        "--extrabiblical-path",
        type=Path,
        default=DEFAULT_TF_ROOT / "extrabiblical" / "tf" / "0.2",
    )
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--chunk-words", type=int, default=100)
    parser.add_argument("--overlap-words", type=int, default=15)
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--residual-min-words", type=int, default=20)
    parser.add_argument("--quote-ngram", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT
        / "analysis"
        / "reports"
        / "cross_corpus_quote_ablation.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT
        / "analysis"
        / "reports"
        / "CROSS_CORPUS_QUOTE_ABLATION.md",
    )
    return parser.parse_args()


def ngram_inventory(passages: Sequence[Passage], n: int) -> set[tuple[str, ...]]:
    if n < 2:
        raise ValueError("quotation n-gram must contain at least two words")
    return {
        tuple(passage.tokens[start : start + n])
        for passage in passages
        for start in range(len(passage.tokens) - n + 1)
    }


def mask_external_ngrams(
    passage: Passage,
    inventory: set[tuple[str, ...]],
    *,
    n: int,
) -> tuple[Passage, dict]:
    """Mask complete matching runs without joining their former neighbors."""
    masked = [False] * len(passage.tokens)
    matched_windows = 0
    for start in range(len(passage.tokens) - n + 1):
        if tuple(passage.tokens[start : start + n]) in inventory:
            matched_windows += 1
            for position in range(start, start + n):
                masked[position] = True

    residual_tokens = tuple(
        token for token, is_masked in zip(passage.tokens, masked) if not is_masked
    )
    text_parts = []
    inside_gap = False
    for token, is_masked in zip(passage.tokens, masked):
        if is_masked:
            if not inside_gap:
                text_parts.append(QUOTE_GAP)
            inside_gap = True
        else:
            text_parts.append(token)
            inside_gap = False
    residual = Passage(
        passage_id=passage.passage_id,
        corpus=passage.corpus,
        book=passage.book,
        reference=passage.reference,
        text=" ".join(text_parts),
        tokens=residual_tokens,
        composition=passage.composition,
        genre=passage.genre,
        section=passage.section,
    )
    masked_words = sum(masked)
    return residual, {
        "matched_windows": matched_windows,
        "masked_words": masked_words,
        "original_words": len(passage.tokens),
        "residual_words": len(residual_tokens),
        "masked_fraction": masked_words / len(passage.tokens),
    }


def surviving_inventory_matches(
    text: str,
    inventory: set[tuple[str, ...]],
    *,
    n: int,
) -> int:
    """Count matches inside unmasked runs; boundaries never bridge quote gaps."""
    total = 0
    for run in text.split(QUOTE_GAP):
        tokens = run.split()
        total += sum(
            tuple(tokens[start : start + n]) in inventory
            for start in range(len(tokens) - n + 1)
        )
    return total


def rank_passages(
    queries: Sequence[Passage],
    external: Sequence[Passage],
    *,
    top_k: int,
    lexical_weight: float = SOURCE_WEIGHTS["lexical"],
    orthographic_weight: float = SOURCE_WEIGHTS["orthographic"],
) -> list[dict]:
    if lexical_weight < 0 or orthographic_weight < 0:
        raise ValueError("ranking weights must be non-negative")
    if not np.isclose(lexical_weight + orthographic_weight, 1.0):
        raise ValueError("ranking weights must sum to one")
    word_scores, char_scores = _tfidf_similarities(queries, external)
    word_percentiles = row_percentiles(word_scores)
    char_percentiles = row_percentiles(char_scores)
    combined = (
        lexical_weight * word_percentiles
        + orthographic_weight * char_percentiles
    )
    output = []
    for query_index, query in enumerate(queries):
        best_by_book = {}
        for external_index, candidate in enumerate(external):
            key = (candidate.corpus, candidate.book)
            previous = best_by_book.get(key)
            if (
                previous is None
                or combined[query_index, external_index]
                > combined[query_index, previous]
            ):
                best_by_book[key] = external_index
        ranked = sorted(
            best_by_book.values(),
            key=lambda index: (
                -combined[query_index, index],
                external[index].corpus,
                external[index].book,
            ),
        )[:top_k]
        output.append(
            {
                "query": asdict(query),
                "matches": [
                    {
                        "rank": rank,
                        "passage_id": external[index].passage_id,
                        "corpus": external[index].corpus,
                        "book": external[index].book,
                        "reference": external[index].reference,
                        "combined_score": round(float(combined[query_index, index]), 6),
                        "lexical_score": round(
                            float(word_scores[query_index, index]), 6
                        ),
                        "orthographic_score": round(
                            float(char_scores[query_index, index]), 6
                        ),
                    }
                    for rank, index in enumerate(ranked, start=1)
                ],
            }
        )
    return output


def benjamini_hochberg(rows: list[dict], key: str) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index][key])
    total = len(rows)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = total - reverse_rank + 1
        adjusted = min(running, rows[index][key] * total / rank)
        rows[index]["bh_adjusted_p"] = adjusted
        running = adjusted


def clustered_composition_enrichment(
    ranked: Sequence[dict],
    *,
    permutations: int,
    seed: int = 17,
) -> dict:
    """Permutation inference with composition labels shuffled by DSS scroll."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    books = sorted(
        {
            f"{match['corpus']}:{match['book']}"
            for row in ranked
            for match in row["matches"][:3]
        }
    )
    book_index = {book: index for index, book in enumerate(books)}
    incidence = np.zeros((len(ranked), len(books)), dtype=np.int8)
    scrolls = []
    labels_by_scroll = {}
    compositions = []
    for row_index, row in enumerate(ranked):
        query = row["query"]
        scroll = query["book"]
        composition = query["composition"]
        scrolls.append(scroll)
        labels_by_scroll.setdefault(scroll, composition)
        if labels_by_scroll[scroll] != composition:
            raise ValueError(f"scroll {scroll} has inconsistent composition labels")
        compositions.append(composition)
        for match in row["matches"][:3]:
            key = f"{match['corpus']}:{match['book']}"
            incidence[row_index, book_index[key]] = 1

    unique_scrolls = sorted(labels_by_scroll)
    scroll_labels = np.array(
        [labels_by_scroll[scroll] for scroll in unique_scrolls], dtype=object
    )
    scroll_to_index = {scroll: index for index, scroll in enumerate(unique_scrolls)}
    query_scroll_indices = np.array([scroll_to_index[scroll] for scroll in scrolls])
    composition_names = sorted(set(compositions))
    observed = {}
    for composition in composition_names:
        mask = np.array(compositions) == composition
        observed[composition] = incidence[mask].mean(axis=0)

    exceedances = {
        composition: np.zeros(len(books), dtype=np.int64)
        for composition in composition_names
    }
    rng = np.random.default_rng(seed)
    for _ in range(permutations):
        shuffled_query_labels = rng.permutation(scroll_labels)[query_scroll_indices]
        for composition in composition_names:
            mask = shuffled_query_labels == composition
            if not np.any(mask):
                continue
            permuted = incidence[mask].mean(axis=0)
            exceedances[composition] += permuted >= observed[composition] - 1e-12

    rows = []
    passage_counts = Counter(compositions)
    scroll_counts = Counter(labels_by_scroll.values())
    for composition in composition_names:
        for index, book in enumerate(books):
            support = float(observed[composition][index])
            rows.append(
                {
                    "composition": composition,
                    "source_book": book,
                    "top3_support_fraction": support,
                    "top3_support_passages": round(
                        support * passage_counts[composition]
                    ),
                    "passages": passage_counts[composition],
                    "scrolls": scroll_counts[composition],
                    "cluster_permutation_p": float(
                        (1 + exceedances[composition][index]) / (permutations + 1)
                    ),
                }
            )
    benjamini_hochberg(rows, "cluster_permutation_p")
    tested_pairs = len(rows)
    rows.sort(
        key=lambda row: (
            row["bh_adjusted_p"],
            -row["top3_support_fraction"],
            row["composition"],
            row["source_book"],
        )
    )
    return {
        "unit": "DSS scroll",
        "iterations": permutations,
        "tested_pairs": tested_pairs,
        "rows": [row for row in rows if row["top3_support_fraction"] > 0],
    }


def nested_feature_control(
    rankings: dict[str, Sequence[dict]],
    *,
    permutations: int = 10_000,
    seed: int = 29,
) -> dict:
    """Select a feature family without using the held-out Pesher manuscript."""
    feature_order = tuple(rankings)
    by_feature: dict[str, dict[str, list[dict]]] = {}
    for feature, rows in rankings.items():
        grouped = defaultdict(list)
        for row in rows:
            query = row["query"]
            if (
                query["composition"] == "Pesharim"
                and query["book"] in PESHER_SOURCE_BOOKS
            ):
                grouped[query["book"]].append(row)
        by_feature[feature] = grouped
    manuscripts = sorted(
        set.intersection(*(set(rows) for rows in by_feature.values()))
    )
    if len(manuscripts) < 2:
        return {"status": "not_run", "reason": "fewer than two mapped manuscripts"}

    def score(feature: str, manuscript: str, source: str, cutoff: int) -> float:
        rows = by_feature[feature][manuscript]
        return float(
            np.mean(
                [
                    source
                    in [
                        match["book"]
                        for match in row["matches"][:cutoff]
                        if match["corpus"] == "bhsa"
                    ]
                    for row in rows
                ]
            )
        )

    expected = np.array(
        [PESHER_SOURCE_BOOKS[manuscript] for manuscript in manuscripts],
        dtype=object,
    )

    def nested_scores(labels: np.ndarray) -> tuple[float, float, list[dict]]:
        top1 = []
        top3 = []
        selections = []
        for heldout_index, heldout in enumerate(manuscripts):
            training_indices = [
                index for index in range(len(manuscripts)) if index != heldout_index
            ]
            feature_scores = {}
            for feature in feature_order:
                train_top3 = np.mean(
                    [
                        score(
                            feature,
                            manuscripts[index],
                            str(labels[index]),
                            3,
                        )
                        for index in training_indices
                    ]
                )
                train_top1 = np.mean(
                    [
                        score(
                            feature,
                            manuscripts[index],
                            str(labels[index]),
                            1,
                        )
                        for index in training_indices
                    ]
                )
                feature_scores[feature] = (train_top3, train_top1)
            selected = max(
                feature_order,
                key=lambda feature: (
                    feature_scores[feature][0],
                    feature_scores[feature][1],
                    -feature_order.index(feature),
                ),
            )
            heldout_top1 = score(
                selected, heldout, str(labels[heldout_index]), 1
            )
            heldout_top3 = score(
                selected, heldout, str(labels[heldout_index]), 3
            )
            top1.append(heldout_top1)
            top3.append(heldout_top3)
            selections.append(
                {
                    "heldout_manuscript": heldout,
                    "selected_feature": selected,
                    "training_macro_top1": feature_scores[selected][1],
                    "training_macro_top3": feature_scores[selected][0],
                    "heldout_top1": heldout_top1,
                    "heldout_top3": heldout_top3,
                }
            )
        return float(np.mean(top1)), float(np.mean(top3)), selections

    observed_top1, observed_top3, selections = nested_scores(expected)
    rng = np.random.default_rng(seed)
    null_top1 = np.empty(permutations)
    null_top3 = np.empty(permutations)
    for iteration in range(permutations):
        null_top1[iteration], null_top3[iteration], _ = nested_scores(
            rng.permutation(expected)
        )
    return {
        "status": "nested_leave_one_manuscript_out",
        "selection_unit": "Pesher manuscript",
        "feature_candidates": list(feature_order),
        "manuscripts": len(manuscripts),
        "macro_top1_recovery": observed_top1,
        "macro_top3_recovery": observed_top3,
        "permutation_iterations": permutations,
        "top1_permutation_p": float(
            (1 + np.sum(null_top1 >= observed_top1)) / (permutations + 1)
        ),
        "top3_permutation_p": float(
            (1 + np.sum(null_top3 >= observed_top3)) / (permutations + 1)
        ),
        "folds": selections,
    }


def compare_conditions(
    literal: Sequence[dict],
    residual: Sequence[dict],
) -> list[dict]:
    residual_by_id = {row["query"]["passage_id"]: row for row in residual}
    by_composition = defaultdict(
        lambda: {"n": 0, "literal": Counter(), "residual": Counter()}
    )
    for row in literal:
        query = row["query"]
        residual_row = residual_by_id.get(query["passage_id"])
        if residual_row is None:
            continue
        bucket = by_composition[query["composition"]]
        bucket["n"] += 1
        for match in row["matches"][:3]:
            bucket["literal"][f"{match['corpus']}:{match['book']}"] += 1
        for match in residual_row["matches"][:3]:
            bucket["residual"][f"{match['corpus']}:{match['book']}"] += 1

    output = []
    for composition, bucket in sorted(by_composition.items()):
        candidates = set(bucket["literal"]) | set(bucket["residual"])
        rows = []
        for book in candidates:
            literal_count = bucket["literal"][book]
            residual_count = bucket["residual"][book]
            rows.append(
                {
                    "source_book": book,
                    "literal_top3_support": literal_count,
                    "residual_top3_support": residual_count,
                    "literal_fraction": literal_count / bucket["n"],
                    "residual_fraction": residual_count / bucket["n"],
                    "retained_fraction": (
                        residual_count / literal_count if literal_count else None
                    ),
                }
            )
        rows.sort(
            key=lambda row: (
                -row["residual_top3_support"],
                -row["literal_top3_support"],
                row["source_book"],
            )
        )
        output.append(
            {
                "composition": composition,
                "passages_retained": bucket["n"],
                "sources": rows[:10],
            }
        )
    return output


def markdown_report(report: dict) -> str:
    mask = report["masking"]
    literal = report["positive_controls"]["literal"]
    residual = report["positive_controls"]["residual"]
    lines = [
        "# Quote-aware cross-corpus connection analysis",
        "",
        "Status: **paper-method candidate; substantive connections remain exploratory**.",
        "",
        "This analysis separates literal source recovery from residual affinity.",
        f"Every DSS token participating in an exact external {mask['ngram_words']}-word",
        "match is replaced by a boundary marker before the residual ranking.",
        "",
        "## Validation and ablation",
        "",
        f"- DSS passages before masking: {mask['passages_before']}",
        f"- DSS passages retaining at least {mask['residual_min_words']} words: "
        f"{mask['passages_after']}",
        f"- Mean words masked: {100 * mask['mean_masked_fraction']:.1f}%",
        f"- Median words masked: {100 * mask['median_masked_fraction']:.1f}%",
        f"- Surviving exact {mask['ngram_words']}-word matches inside residual "
        f"runs: {mask['surviving_matching_runs']}",
        f"- Known-source Pesher Top-1 / Top-3 before masking: "
        f"{100 * literal.get('macro_top1_recovery', 0):.1f}% / "
        f"{100 * literal.get('macro_top3_recovery', 0):.1f}%",
        f"- Known-source Pesher Top-1 / Top-3 after masking: "
        f"{100 * residual.get('macro_top1_recovery', 0):.1f}% / "
        f"{100 * residual.get('macro_top3_recovery', 0):.1f}%",
        "",
        "A large control drop is expected and useful: it demonstrates that the",
        "literal channel detects known quotations. Connections that persist in the",
        f"residual channel cannot be explained by an exact external "
        f"{mask['ngram_words']}-word run left in the query, although genre, date,",
        "dialect, and formulaic language remain alternative explanations.",
        "",
        "### Feature-family sensitivity",
        "",
        "| Ranking signal | Literal Top-3 | Residual Top-3 | Residual p |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for name, conditions in report["feature_family_sensitivity"].items():
        feature_residual = conditions["residual"]
        lines.append(
            f"| {name} | "
            f"{100 * conditions['literal'].get('macro_top3_recovery', 0):.1f}% | "
            f"{100 * feature_residual.get('macro_top3_recovery', 0):.1f}% | "
            f"{feature_residual.get('top3_permutation_p', 1):.4g} |"
        )
    nested = report["nested_feature_validation"]["residual"]
    lines.extend(
        [
        "",
        "Feature choice was also repeated in nested leave-one-manuscript-out",
        "validation, selecting among combined, word-only, and character-only",
        "rankings on the other manuscripts before scoring each held-out Pesher.",
        f"Nested residual Top-1 / Top-3: "
        f"{100 * nested.get('macro_top1_recovery', 0):.1f}% / "
        f"{100 * nested.get('macro_top3_recovery', 0):.1f}% "
        f"(Top-3 permutation p={nested.get('top3_permutation_p', 1):.4g}).",
        "",
        "## Composition-level decomposition",
        "",
        ]
    )
    for block in report["composition_comparison"]:
        lines.extend(
            [
                f"### {block['composition']} (N={block['passages_retained']})",
                "",
                "| Source | Literal Top-3 | Residual Top-3 | Residual support |",
                "| :--- | ---: | ---: | ---: |",
            ]
        )
        for row in block["sources"]:
            lines.append(
                f"| {row['source_book']} | {row['literal_top3_support']} | "
                f"{row['residual_top3_support']} | "
                f"{100 * row['residual_fraction']:.1f}% |"
            )
        lines.append("")

    significant = [
        row
        for row in report["residual_cluster_enrichment"]["rows"]
        if row["bh_adjusted_p"] <= 0.05
    ]
    lines.extend(
        [
            "## Scroll-cluster residual enrichment",
            "",
            "Composition labels are shuffled at the scroll level. Benjamini-Hochberg",
            "correction covers every observed composition/source pair.",
            "",
        ]
    )
    if significant:
        lines.extend(
            [
                "| Composition | Residual source | Top-3 support | Scrolls | BH q |",
                "| :--- | :--- | ---: | ---: | ---: |",
            ]
        )
        for row in significant:
            lines.append(
                f"| {row['composition']} | {row['source_book']} | "
                f"{100 * row['top3_support_fraction']:.1f}% | "
                f"{row['scrolls']} | {row['bh_adjusted_p']:.4f} |"
            )
    else:
        lines.append("No residual pair survives correction at q <= 0.05.")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            report["warning"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    targets = {
        value.strip() for value in args.targets.split(",") if value.strip()
    }
    queries = load_preserved_dss_queries(
        args.dss_preserved_jsonl,
        metadata_csv=args.dss_csv,
        targets=targets,
        min_words=args.min_words,
    )
    external = load_tf_passages(
        args.bhsa_path,
        corpus="bhsa",
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_words=args.min_words,
    )
    external.extend(
        load_tf_passages(
            args.extrabiblical_path,
            corpus="extrabiblical",
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            min_words=args.min_words,
            excluded_books=QUMRAN_EXTRA_BOOKS,
        )
    )
    inventory = ngram_inventory(external, args.quote_ngram)
    masked_rows = [
        mask_external_ngrams(query, inventory, n=args.quote_ngram)
        for query in queries
    ]
    residual_queries = [
        residual
        for residual, audit in masked_rows
        if audit["residual_words"] >= args.residual_min_words
    ]
    retained_ids = {query.passage_id for query in residual_queries}
    literal_queries = [
        query for query in queries if query.passage_id in retained_ids
    ]
    audits = [
        audit
        for residual, audit in masked_rows
        if residual.passage_id in retained_ids
    ]
    surviving_matches = sum(
        surviving_inventory_matches(
            residual.text,
            inventory,
            n=args.quote_ngram,
        )
        for residual in residual_queries
    )
    if surviving_matches:
        raise RuntimeError(
            f"quotation ablation left {surviving_matches} matched runs"
        )
    literal_ranked = rank_passages(literal_queries, external, top_k=args.top_k)
    residual_ranked = rank_passages(residual_queries, external, top_k=args.top_k)
    literal_feature_rankings = {"combined": literal_ranked}
    residual_feature_rankings = {"combined": residual_ranked}
    feature_sensitivity = {}
    for name, lexical_weight, orthographic_weight in (
        ("word_only", 1.0, 0.0),
        ("character_only", 0.0, 1.0),
    ):
        literal_feature_rankings[name] = rank_passages(
            literal_queries,
            external,
            top_k=args.top_k,
            lexical_weight=lexical_weight,
            orthographic_weight=orthographic_weight,
        )
        residual_feature_rankings[name] = rank_passages(
            residual_queries,
            external,
            top_k=args.top_k,
            lexical_weight=lexical_weight,
            orthographic_weight=orthographic_weight,
        )
        feature_sensitivity[name] = {
            "literal": evaluate_pesher_source_control(
                literal_feature_rankings[name]
            ),
            "residual": evaluate_pesher_source_control(
                residual_feature_rankings[name]
            ),
        }
    report = {
        "status": "paper_method_candidate_quote_aware_connection_screen",
        "warning": (
            "Residual enrichment is evidence of a reproducible corpus affinity, "
            "not proof of authorship, direct borrowing, a lost source, or direction "
            "of influence. Genre, date, dialect, shared tradition, and formulaic "
            "language require philological adjudication."
        ),
        "masking": {
            "ngram_words": args.quote_ngram,
            "passages_before": len(queries),
            "passages_after": len(residual_queries),
            "residual_min_words": args.residual_min_words,
            "mean_masked_fraction": float(
                np.mean([audit["masked_fraction"] for audit in audits])
            ),
            "median_masked_fraction": float(
                np.median([audit["masked_fraction"] for audit in audits])
            ),
            "total_matching_windows": sum(
                audit["matched_windows"] for audit in audits
            ),
            "surviving_matching_runs": surviving_matches,
        },
        "positive_controls": {
            "literal": evaluate_pesher_source_control(literal_ranked),
            "residual": evaluate_pesher_source_control(residual_ranked),
        },
        "feature_family_sensitivity": feature_sensitivity,
        "nested_feature_validation": {
            "literal": nested_feature_control(literal_feature_rankings),
            "residual": nested_feature_control(residual_feature_rankings),
        },
        "composition_comparison": compare_conditions(
            literal_ranked, residual_ranked
        ),
        "residual_cluster_enrichment": clustered_composition_enrichment(
            residual_ranked,
            permutations=args.permutations,
        ),
        "protocol": {
            "targets": sorted(targets),
            "query_source_sha256": sha256_files(
                args.dss_preserved_jsonl.parent,
                [args.dss_preserved_jsonl.name],
            ),
            "label_metadata_sha256": sha256_files(
                args.dss_csv.parent, [args.dss_csv.name]
            ),
            "bhsa": source_receipt(args.bhsa_path),
            "extrabiblical": source_receipt(args.extrabiblical_path),
            "chunk_words": args.chunk_words,
            "overlap_words": args.overlap_words,
            "ranking_weights": SOURCE_WEIGHTS,
            "top_k": args.top_k,
            "permutation_seed": 17,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_markdown}")
    print(
        json.dumps(
            {
                "masking": report["masking"],
                "positive_controls": report["positive_controls"],
                "significant_residual_pairs": sum(
                    row["bh_adjusted_p"] <= 0.05
                    for row in report["residual_cluster_enrichment"]["rows"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
