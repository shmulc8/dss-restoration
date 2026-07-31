"""Leakage-audited external-retrieval ablation for DSS span restoration.

The experiment freezes the reconstruction-free 60-development/300-heldout
synthetic-damage sample used by the expanded word-model pilot.  It generates
the preserved-only word model's unknown-length candidate pool once, then
reranks that identical pool under seven retrieval shelves:

* no retrieval;
* preserved DSS training scrolls only;
* the Hebrew Bible;
* early Hebrew inscriptions (excluding Aramaic and Moabite comparators);
* later rabbinic comparators;
* all external sources;
* DSS training scrolls plus all external sources.

Retrieval sees only the visible left and right context.  The held-out answer is
used only for scoring and for a separately labelled answer-removal stress test.
Each retrieval weight is selected on development spans and frozen for heldout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cross_corpus_connections import (  # noqa: E402
    DEFAULT_TF_ROOT,
    EPIGRAPHIC_BOOKS,
    QUMRAN_EXTRA_BOOKS,
    RABBINIC_BOOKS,
    Passage,
    load_tf_passages,
    source_receipt,
)
from eval.tf_embible_dss_benchmark import (  # noqa: E402
    Item,
    file_sha256,
    fit_penalty,
    rank_scores,
    rank_with_penalty,
    sample_items,
    sample_sha256,
    word_candidates,
)
from utils.preserved_corpus import GAP_TOKEN, load_chunks, split_scrolls  # noqa: E402

tlog.set_verbosity_error()

DEV_PER_LENGTH = 20
TEST_PER_LENGTH = 100
MAX_WORDS = 3
MAX_CHARS = 18
CONTEXT_WORDS = 8
SAMPLE_SEEDS = {"dev": 71, "heldout": 73}
RETRIEVAL_TOP_K = 20
RERANK_ALPHAS = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
WORD_WEIGHT = 0.8
CHAR_WEIGHT = 0.2
HEBREW_EPIGRAPHIC_BOOKS = EPIGRAPHIC_BOOKS - {"Balaam", "Mesa"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DSS external-retrieval restoration ablation."
    )
    parser.add_argument(
        "--word-model",
        type=Path,
        default=ROOT / "ft_msbert_span_preserved_nonbib",
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
    parser.add_argument("--chunk-words", type=int, default=100)
    parser.add_argument("--overlap-words", type=int, default=15)
    parser.add_argument("--retrieval-top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--top-k-per-step", type=int, default=24)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=ROOT
        / "analysis"
        / "cache"
        / "external_retrieval_candidates.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT
        / "analysis"
        / "reports"
        / "cross_corpus_retrieval_ablation.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT
        / "analysis"
        / "reports"
        / "CROSS_CORPUS_RETRIEVAL_ABLATION.md",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def item_payload(item: Item) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "scroll": item.scroll,
        "left": list(item.left),
        "gold": list(item.gold),
        "right": list(item.right),
    }


def item_from_payload(row: dict[str, Any]) -> Item:
    return Item(
        item_id=row["item_id"],
        scroll=row["scroll"],
        left=tuple(row["left"]),
        gold=tuple(row["gold"]),
        right=tuple(row["right"]),
    )


def candidate_cache_receipt(
    *,
    model_path: Path,
    dev_items: Sequence[Item],
    heldout_items: Sequence[Item],
    beam_width: int,
    top_k_per_step: int,
) -> dict[str, Any]:
    checkpoint = model_path / "model.safetensors"
    return {
        "model_path": str(model_path),
        "model_sha256": file_sha256(checkpoint),
        "dev_sample_sha256": sample_sha256(list(dev_items)),
        "heldout_sample_sha256": sample_sha256(list(heldout_items)),
        "max_words": MAX_WORDS,
        "beam_width": beam_width,
        "top_k_per_step": top_k_per_step,
        "candidate_generation_uses_gold": False,
    }


def generate_candidate_records(
    items: Sequence[Item],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    beam_width: int,
    top_k_per_step: int,
    label: str,
) -> list[dict[str, Any]]:
    records = []
    for number, item in enumerate(items, start=1):
        rows = word_candidates(
            item.left,
            item.right,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_words=MAX_WORDS,
            beam_width=beam_width,
            top_k_per_step=top_k_per_step,
        )
        records.append(
            {
                "item": item_payload(item),
                "gold": item.gold_text,
                "word": [
                    [candidate, float(score), int(word_count)]
                    for candidate, score, word_count in rows
                ],
            }
        )
        if number % 10 == 0 or number == len(items):
            print(f"generated {label} candidates: {number}/{len(items)}", flush=True)
    return records


def load_or_generate_candidates(
    *,
    cache_path: Path,
    model_path: Path,
    dev_items: Sequence[Item],
    heldout_items: Sequence[Item],
    device: str,
    beam_width: int,
    top_k_per_step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected = candidate_cache_receipt(
        model_path=model_path,
        dev_items=dev_items,
        heldout_items=heldout_items,
        beam_width=beam_width,
        top_k_per_step=top_k_per_step,
    )
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("receipt") != expected:
            raise ValueError(
                f"candidate cache does not match the frozen protocol: {cache_path}"
            )
        return payload["dev"], payload["heldout"], {
            **expected,
            "cache": str(cache_path),
            "cache_hit": True,
        }

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=True,
        local_files_only=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device).eval()
    dev_records = generate_candidate_records(
        dev_items,
        tokenizer=tokenizer,
        model=model,
        device=device,
        beam_width=beam_width,
        top_k_per_step=top_k_per_step,
        label="dev",
    )
    heldout_records = generate_candidate_records(
        heldout_items,
        tokenizer=tokenizer,
        model=model,
        device=device,
        beam_width=beam_width,
        top_k_per_step=top_k_per_step,
        label="heldout",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "receipt": expected,
                "dev": dev_records,
                "heldout": heldout_records,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return dev_records, heldout_records, {
        **expected,
        "cache": str(cache_path),
        "cache_hit": False,
    }


def dss_train_passages() -> list[Passage]:
    passages = []
    for row in load_chunks("train"):
        tokens = tuple(
            word for word in row["text"].split() if word and word != GAP_TOKEN
        )
        if not tokens:
            continue
        passages.append(
            Passage(
                passage_id=f"dss_train:{row['scroll']}:{row['chunk_index']}",
                corpus="dss_train",
                book=row["scroll"],
                reference=f"{row['scroll']} chunk {row['chunk_index']}",
                text=" ".join(tokens),
                tokens=tokens,
            )
        )
    return passages


def load_shelves(
    *,
    bhsa_path: Path,
    extrabiblical_path: Path,
    chunk_words: int,
    overlap_words: int,
) -> tuple[dict[str, list[Passage]], dict[str, Any]]:
    bible = load_tf_passages(
        bhsa_path,
        corpus="bhsa",
        chunk_words=chunk_words,
        overlap_words=overlap_words,
        min_words=max(20, chunk_words // 2),
    )
    extra = load_tf_passages(
        extrabiblical_path,
        corpus="extrabiblical",
        chunk_words=chunk_words,
        overlap_words=overlap_words,
        min_words=max(20, chunk_words // 2),
        excluded_books=QUMRAN_EXTRA_BOOKS,
    )
    inscriptions = [
        row for row in extra if row.book in HEBREW_EPIGRAPHIC_BOOKS
    ]
    rabbinic = [row for row in extra if row.book in RABBINIC_BOOKS]
    dss = dss_train_passages()
    external = [*bible, *inscriptions, *rabbinic]
    shelves = {
        "dss_train_only": dss,
        "hebrew_bible": bible,
        "early_hebrew_inscriptions": inscriptions,
        "later_rabbinic": rabbinic,
        "all_external": external,
        "dss_plus_external": [*dss, *external],
    }
    empty = [name for name, rows in shelves.items() if not rows]
    if empty:
        raise ValueError(f"empty retrieval shelves: {', '.join(empty)}")
    metadata = {
        name: {
            "passages": len(rows),
            "books_or_scrolls": len({(row.corpus, row.book) for row in rows}),
        }
        for name, rows in shelves.items()
    }
    return shelves, metadata


def visible_query(item: Item) -> str:
    """Return only observable context; the hidden span is never included."""
    return " ".join((*item.left, *item.right))


def contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    target = tuple(phrase)
    return any(tuple(tokens[start : start + width]) == target for start in range(
        len(tokens) - width + 1
    ))


def phrase_document_frequency(
    documents: Sequence[Passage],
    phrase_lengths: Sequence[int],
) -> Counter[tuple[str, ...]]:
    wanted = sorted(set(phrase_lengths))
    counts: Counter[tuple[str, ...]] = Counter()
    for document in documents:
        for width in wanted:
            if width > len(document.tokens):
                continue
            seen = {
                tuple(document.tokens[start : start + width])
                for start in range(len(document.tokens) - width + 1)
            }
            counts.update(seen)
    return counts


def retrieve_documents(
    queries: Sequence[Item],
    documents: Sequence[Passage],
    *,
    top_k: int,
) -> tuple[list[list[tuple[int, float]]], dict[str, Any]]:
    if top_k < 1:
        raise ValueError("retrieval top_k must be positive")
    texts = [row.text for row in documents]
    query_texts = [visible_query(item) for item in queries]
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    document_word = word_vectorizer.fit_transform(texts)
    query_word = word_vectorizer.transform(query_texts)
    word_scores = cosine_similarity(query_word, document_word)
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    document_char = char_vectorizer.fit_transform(texts)
    query_char = char_vectorizer.transform(query_texts)
    char_scores = cosine_similarity(query_char, document_char)
    scores = WORD_WEIGHT * word_scores + CHAR_WEIGHT * char_scores

    rankings = []
    limit = min(top_k, len(documents))
    for row in scores:
        indices = np.argsort(-row, kind="stable")[:limit]
        rankings.append(
            [
                (int(index), float(row[index]))
                for index in indices
                if row[index] > 0
            ]
        )
    return rankings, {
        "word_weight": WORD_WEIGHT,
        "character_weight": CHAR_WEIGHT,
        "word_features": len(word_vectorizer.vocabulary_),
        "character_features": len(char_vectorizer.vocabulary_),
        "query_uses_gold": False,
    }


def candidate_support(
    candidate: str,
    ranking: Sequence[tuple[int, float]],
    documents: Sequence[Passage],
    *,
    removed_phrase: Sequence[str] | None = None,
) -> float:
    candidate_tokens = tuple(candidate.split())
    numerator = 0.0
    denominator = 0.0
    for index, score in ranking:
        document_tokens = documents[index].tokens
        if removed_phrase and contains_phrase(document_tokens, removed_phrase):
            continue
        denominator += score
        if contains_phrase(document_tokens, candidate_tokens):
            numerator += score
    return numerator / denominator if denominator else 0.0


def rerank_candidates(
    baseline_rows: Sequence[tuple[str, float]],
    supports: dict[str, float],
    alpha: float,
) -> list[tuple[str, float]]:
    base = rank_scores(list(baseline_rows))
    original_rank = {candidate: rank for rank, (candidate, _) in enumerate(baseline_rows)}
    return sorted(
        (
            (candidate, score + alpha * supports.get(candidate, 0.0))
            for candidate, score in base.items()
        ),
        key=lambda row: (-row[1], original_rank[row[0]], row[0]),
    )


def exact_rank(rows: Sequence[tuple[str, float]], gold: str) -> int:
    return next(
        (rank for rank, (candidate, _) in enumerate(rows) if candidate == gold),
        999,
    )


def build_support_records(
    candidate_records: Sequence[dict[str, Any]],
    rankings: Sequence[Sequence[tuple[int, float]]],
    documents: Sequence[Passage],
    *,
    word_penalty: float,
) -> list[dict[str, Any]]:
    if len(candidate_records) != len(rankings):
        raise ValueError("candidate records and retrieval rankings differ in length")
    phrase_lengths = [
        len(record["item"]["gold"]) for record in candidate_records
    ]
    frequencies = phrase_document_frequency(documents, phrase_lengths)
    output = []
    for record, ranking in zip(candidate_records, rankings):
        item = item_from_payload(record["item"])
        raw_rows = [
            (str(candidate), float(score), int(size))
            for candidate, score, size in record["word"]
        ]
        baseline = rank_with_penalty(raw_rows, word_penalty)
        gold_tokens = item.gold
        standard = {
            candidate: candidate_support(
                candidate,
                ranking,
                documents,
            )
            for candidate, _ in baseline
        }
        redacted = {
            candidate: candidate_support(
                candidate,
                ranking,
                documents,
                removed_phrase=gold_tokens,
            )
            for candidate, _ in baseline
        }
        relevant_ranks = [
            rank
            for rank, (index, _) in enumerate(ranking, start=1)
            if contains_phrase(documents[index].tokens, gold_tokens)
        ]
        relevant_total = frequencies[tuple(gold_tokens)]
        dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(relevant_total, len(ranking)) + 1)
        )
        output.append(
            {
                "item": record["item"],
                "gold": record["gold"],
                "baseline": baseline,
                "support": standard,
                "support_answer_removed": redacted,
                "retrieval": {
                    "documents_returned": len(ranking),
                    "gold_document_frequency": relevant_total,
                    "gold_first_rank": relevant_ranks[0] if relevant_ranks else None,
                    "gold_recall_at_5": any(rank <= 5 for rank in relevant_ranks),
                    "gold_recall_at_10": any(rank <= 10 for rank in relevant_ranks),
                    "gold_recall_at_20": bool(relevant_ranks),
                    "gold_reciprocal_rank": (
                        1.0 / relevant_ranks[0] if relevant_ranks else 0.0
                    ),
                    "gold_ndcg_at_20": dcg / ideal if ideal else 0.0,
                    "gold_in_candidate_pool": any(
                        candidate == record["gold"] for candidate, _ in baseline
                    ),
                    "gold_candidate_has_support": standard.get(
                        record["gold"], 0.0
                    )
                    > 0,
                },
            }
        )
    return output


def fit_alpha(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grid = {}
    for alpha in RERANK_ALPHAS:
        ranks = [
            exact_rank(
                rerank_candidates(record["baseline"], record["support"], alpha),
                record["gold"],
            )
            for record in records
        ]
        grid[str(alpha)] = {
            "n": len(ranks),
            "top1": 100 * sum(rank == 0 for rank in ranks) / len(ranks),
            "top10": 100 * sum(rank < 10 for rank in ranks) / len(ranks),
        }
    selected = max(
        RERANK_ALPHAS,
        key=lambda alpha: (
            grid[str(alpha)]["top10"],
            grid[str(alpha)]["top1"],
            -alpha,
        ),
    )
    return {
        "selection_split": "dev",
        "objective": "exact complete-span Top-10, then Top-1, then smaller alpha",
        "selected_alpha": selected,
        "grid": grid,
    }


def summarize_condition(
    records: Sequence[dict[str, Any]],
    *,
    alpha: float,
    support_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranks = []
    cases = []
    by_words: dict[int, list[int]] = defaultdict(list)
    for record in records:
        rows = rerank_candidates(record["baseline"], record[support_key], alpha)
        rank = exact_rank(rows, record["gold"])
        ranks.append(rank)
        word_count = len(record["item"]["gold"])
        by_words[word_count].append(rank)
        cases.append(
            {
                "item_id": record["item"]["item_id"],
                "scroll": record["item"]["scroll"],
                "word_count": word_count,
                "gold": record["gold"],
                "rank": rank if rank != 999 else None,
                "hit_top1": rank == 0,
                "hit_top10": rank < 10,
                "top10": [candidate for candidate, _ in rows[:10]],
                "gold_support": record[support_key].get(record["gold"], 0.0),
            }
        )

    def metrics(local_ranks: Sequence[int]) -> dict[str, Any]:
        return {
            "n": len(local_ranks),
            "exact_top1": 100 * sum(rank == 0 for rank in local_ranks) / len(local_ranks),
            "exact_top5": 100 * sum(rank < 5 for rank in local_ranks) / len(local_ranks),
            "exact_top10": 100 * sum(rank < 10 for rank in local_ranks) / len(local_ranks),
            "exact_top20": 100 * sum(rank < 20 for rank in local_ranks) / len(local_ranks),
            "mean_reciprocal_rank": float(
                np.mean([1.0 / (rank + 1) if rank != 999 else 0.0 for rank in local_ranks])
            ),
            "candidate_pool_recall": 100 * sum(rank != 999 for rank in local_ranks) / len(local_ranks),
        }

    return {
        **metrics(ranks),
        "by_word_count": {
            str(word_count): metrics(local_ranks)
            for word_count, local_ranks in sorted(by_words.items())
        },
    }, cases


def retrieval_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [record["retrieval"] for record in records]
    return {
        "n": len(rows),
        "queries_with_any_documents": sum(row["documents_returned"] > 0 for row in rows),
        "gold_exists_in_shelf": sum(row["gold_document_frequency"] > 0 for row in rows),
        "gold_recall_at_5": 100 * np.mean([row["gold_recall_at_5"] for row in rows]),
        "gold_recall_at_10": 100 * np.mean([row["gold_recall_at_10"] for row in rows]),
        "gold_recall_at_20": 100 * np.mean([row["gold_recall_at_20"] for row in rows]),
        "gold_in_candidate_pool": sum(row["gold_in_candidate_pool"] for row in rows),
        "gold_candidate_has_retrieval_support": sum(
            row["gold_candidate_has_support"] for row in rows
        ),
        "mean_reciprocal_rank": float(
            np.mean([row["gold_reciprocal_rank"] for row in rows])
        ),
        "mean_ndcg_at_20": float(np.mean([row["gold_ndcg_at_20"] for row in rows])),
    }


def paired_cluster_statistics(
    baseline_cases: Sequence[dict[str, Any]],
    condition_cases: Sequence[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")
    condition_by_id = {row["item_id"]: row for row in condition_cases}
    clusters: dict[str, list[float]] = defaultdict(list)
    for baseline in baseline_cases:
        condition = condition_by_id[baseline["item_id"]]
        clusters[baseline["scroll"]].append(
            float(condition["hit_top10"]) - float(baseline["hit_top10"])
        )
    names = sorted(clusters)
    observed = 100 * np.mean([value for rows in clusters.values() for value in rows])
    rng = np.random.default_rng(seed)
    bootstraps = np.empty(iterations)
    null = np.empty(iterations)
    for iteration in range(iterations):
        sampled = rng.choice(names, size=len(names), replace=True)
        sampled_values = [value for name in sampled for value in clusters[name]]
        bootstraps[iteration] = 100 * np.mean(sampled_values)
        signs = rng.choice((-1.0, 1.0), size=len(names))
        null_values = [
            sign * value
            for sign, name in zip(signs, names)
            for value in clusters[name]
        ]
        null[iteration] = 100 * np.mean(null_values)
    return {
        "unit": "scroll",
        "clusters": len(names),
        "delta_exact_top10_points": float(observed),
        "paired_cluster_bootstrap_95_ci": [
            float(np.percentile(bootstraps, 2.5)),
            float(np.percentile(bootstraps, 97.5)),
        ],
        "cluster_sign_flip_p": float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (iterations + 1)
        ),
        "iterations": iterations,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["results"]["no_retrieval"]["standard"]
    lines = [
        "# Cross-corpus retrieval ablation for DSS restoration",
        "",
        "Status: **single-checkpoint exploratory ablation**, not a final paper result.",
        "All conditions rerank the same unknown-length candidate pool on the same",
        "reconstruction-free held-out spans.",
        "",
        "## Exact complete-span recovery",
        "",
        "| Retrieval shelf | Dev alpha | Top-1 | Top-10 | Delta Top-10 | 95% cluster CI | Holm p | Answer-removed Top-10 |",
        "| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |",
        (
            f"| No retrieval | 0 | {baseline['exact_top1']:.1f}% | "
            f"{baseline['exact_top10']:.1f}% | — | — | — | — |"
        ),
    ]
    for name in report["shelf_order"]:
        row = report["results"][name]
        standard = row["standard"]
        stress = row["answer_removed_stress"]
        statistics = row["statistics"]
        low, high = statistics["paired_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {name} | {row['dev_fit']['selected_alpha']} | "
            f"{standard['exact_top1']:.1f}% | {standard['exact_top10']:.1f}% | "
            f"{statistics['delta_exact_top10_points']:+.1f} | "
            f"[{low:+.1f}, {high:+.1f}] | "
            f"{statistics['holm_adjusted_p']:.4f} | "
            f"{stress['exact_top10']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Candidate-generation ceiling",
            "",
            (
                f"The frozen candidate pool contains the complete held-out answer "
                f"for {baseline['candidate_pool_recall']:.1f}% of spans. By hidden "
                f"word count the ceilings are "
                f"{baseline['by_word_count']['1']['candidate_pool_recall']:.1f}% "
                f"(one), "
                f"{baseline['by_word_count']['2']['candidate_pool_recall']:.1f}% "
                f"(two), and "
                f"{baseline['by_word_count']['3']['candidate_pool_recall']:.1f}% "
                f"(three). Retrieval can reorder candidates but cannot recover an "
                f"answer absent from this pool."
            ),
            "",
            "## Retrieval diagnostics",
            "",
            "| Retrieval shelf | Passages | Gold exists in shelf | Gold recall@20 | Supported gold candidate | Retrieval MRR | nDCG@20 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in report["shelf_order"]:
        shelf = report["protocol"]["shelves"][name]
        diagnostics = report["results"][name]["retrieval"]
        lines.append(
            f"| {name} | {shelf['passages']} | "
            f"{diagnostics['gold_exists_in_shelf']}/{diagnostics['n']} | "
            f"{diagnostics['gold_recall_at_20']:.1f}% | "
            f"{diagnostics['gold_candidate_has_retrieval_support']}/{diagnostics['n']} | "
            f"{diagnostics['mean_reciprocal_rank']:.3f} | "
            f"{diagnostics['mean_ndcg_at_20']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "Retrieval uses only the visible eight words on each side. DSS retrieval",
            "indexes preserved training scrolls only. The answer-removal stress test",
            "drops every retrieved document containing the complete held-out answer",
            "before candidate support is calculated.",
            "",
            "This run uses one trained checkpoint and one frozen 300-span pilot sample.",
            "It does not satisfy the locked three-seed paper promotion gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    model_path = args.word_model.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"word model not found: {model_path}")
    for path in (args.bhsa_path, args.extrabiblical_path):
        if not path.is_dir():
            raise FileNotFoundError(f"Text-Fabric source not found: {path}")
    if args.chunk_words < 20:
        raise ValueError("--chunk-words must be at least 20")
    if not 0 <= args.overlap_words < args.chunk_words:
        raise ValueError("--overlap-words must be smaller than --chunk-words")
    if not 1 <= args.retrieval_top_k <= 100:
        raise ValueError("--retrieval-top-k must be between 1 and 100")

    dev_items, dev_eligible = sample_items(
        "dev",
        per_length=DEV_PER_LENGTH,
        max_words=MAX_WORDS,
        max_chars=MAX_CHARS,
        context_words=CONTEXT_WORDS,
        seed=SAMPLE_SEEDS["dev"],
    )
    heldout_items, heldout_eligible = sample_items(
        "heldout",
        per_length=TEST_PER_LENGTH,
        max_words=MAX_WORDS,
        max_chars=MAX_CHARS,
        context_words=CONTEXT_WORDS,
        seed=SAMPLE_SEEDS["heldout"],
    )
    if len(dev_items) != DEV_PER_LENGTH * MAX_WORDS:
        raise RuntimeError("frozen development sample is incomplete")
    if len(heldout_items) != TEST_PER_LENGTH * MAX_WORDS:
        raise RuntimeError("frozen heldout sample is incomplete")

    train_scrolls = split_scrolls("train")
    heldout_scrolls = {item.scroll for item in heldout_items}
    overlap = train_scrolls & heldout_scrolls
    if overlap:
        raise RuntimeError(f"retrieval leakage: heldout scrolls in train: {overlap}")

    device = choose_device(args.device)
    dev_candidates, heldout_candidates, candidate_receipt = (
        load_or_generate_candidates(
            cache_path=args.candidate_cache,
            model_path=model_path,
            dev_items=dev_items,
            heldout_items=heldout_items,
            device=device,
            beam_width=args.beam_width,
            top_k_per_step=args.top_k_per_step,
        )
    )
    word_penalty = fit_penalty(dev_candidates, "word")
    shelves, shelf_metadata = load_shelves(
        bhsa_path=args.bhsa_path,
        extrabiblical_path=args.extrabiblical_path,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )

    baseline_dev_records = []
    baseline_heldout_records = []
    for records, output in (
        (dev_candidates, baseline_dev_records),
        (heldout_candidates, baseline_heldout_records),
    ):
        for record in records:
            raw = [
                (str(candidate), float(score), int(size))
                for candidate, score, size in record["word"]
            ]
            output.append(
                {
                    "item": record["item"],
                    "gold": record["gold"],
                    "baseline": rank_with_penalty(raw, word_penalty),
                    "support": {},
                    "support_answer_removed": {},
                }
            )
    baseline_metrics, baseline_cases = summarize_condition(
        baseline_heldout_records,
        alpha=0.0,
        support_key="support",
    )

    results: dict[str, Any] = {
        "no_retrieval": {
            "standard": baseline_metrics,
            "cases": baseline_cases,
        }
    }
    p_values = {}
    shelf_order = list(shelves)
    for shelf_index, (name, documents) in enumerate(shelves.items()):
        print(f"retrieving shelf {name}: {len(documents)} passages", flush=True)
        all_items = [*dev_items, *heldout_items]
        rankings, retrieval_protocol = retrieve_documents(
            all_items,
            documents,
            top_k=args.retrieval_top_k,
        )
        dev_rankings = rankings[: len(dev_items)]
        heldout_rankings = rankings[len(dev_items) :]
        dev_support = build_support_records(
            dev_candidates,
            dev_rankings,
            documents,
            word_penalty=word_penalty,
        )
        heldout_support = build_support_records(
            heldout_candidates,
            heldout_rankings,
            documents,
            word_penalty=word_penalty,
        )
        dev_fit = fit_alpha(dev_support)
        alpha = float(dev_fit["selected_alpha"])
        standard, standard_cases = summarize_condition(
            heldout_support,
            alpha=alpha,
            support_key="support",
        )
        stress, stress_cases = summarize_condition(
            heldout_support,
            alpha=alpha,
            support_key="support_answer_removed",
        )
        statistics = paired_cluster_statistics(
            baseline_cases,
            standard_cases,
            iterations=args.bootstrap,
            seed=1000 + shelf_index,
        )
        p_values[name] = statistics["cluster_sign_flip_p"]
        results[name] = {
            "retrieval_protocol": retrieval_protocol,
            "dev_fit": dev_fit,
            "retrieval": retrieval_summary(heldout_support),
            "standard": standard,
            "answer_removed_stress": stress,
            "statistics": statistics,
            "cases": standard_cases,
            "answer_removed_cases": stress_cases,
        }

    adjusted = holm_adjust(p_values)
    for name, value in adjusted.items():
        results[name]["statistics"]["holm_adjusted_p"] = value

    best_name = max(
        shelf_order,
        key=lambda name: results[name]["standard"]["exact_top10"],
    )
    best = results[best_name]
    best_delta = best["statistics"]["delta_exact_top10_points"]
    best_ci = best["statistics"]["paired_cluster_bootstrap_95_ci"]
    if best_delta > 0 and best_ci[0] > 0:
        interpretation = (
            f"{best_name} produced the strongest positive held-out Top-10 change "
            f"({best_delta:+.1f} points), with a cluster interval above zero. "
            "This remains a one-checkpoint pilot pending the full promotion gate."
        )
    elif best_delta > 0:
        interpretation = (
            f"{best_name} had the largest observed Top-10 gain "
            f"({best_delta:+.1f} points), but its cluster interval includes zero. "
            "The result is inconclusive rather than evidence of improvement."
        )
    else:
        interpretation = (
            "No external shelf improved held-out exact Top-10 under the "
            "development-selected reranking rule. The negative result is retained."
        )

    report = {
        "status": "single_checkpoint_exploratory_ablation",
        "protocol": {
            "target": (
                "synthetic lacunae made by hiding contiguous physically preserved "
                "non-biblical DSS words"
            ),
            "modern_reconstructions_used": False,
            "candidate_generation": candidate_receipt,
            "candidate_pool_frozen_across_conditions": True,
            "candidate_pool_model": "preserved-only word span model",
            "unknown_length": True,
            "max_words_searched": MAX_WORDS,
            "context_words_each_side": CONTEXT_WORDS,
            "retrieval_query": "visible left and right context only",
            "retrieval_top_k": args.retrieval_top_k,
            "dev_items": len(dev_items),
            "heldout_items": len(heldout_items),
            "dev_eligible_by_words": dev_eligible,
            "heldout_eligible_by_words": heldout_eligible,
            "dev_sample_seed": SAMPLE_SEEDS["dev"],
            "heldout_sample_seed": SAMPLE_SEEDS["heldout"],
            "dev_sample_sha256": sample_sha256(dev_items),
            "heldout_sample_sha256": sample_sha256(heldout_items),
            "word_length_penalty_selected_on_dev": word_penalty,
            "rerank_alpha_grid": RERANK_ALPHAS,
            "heldout_used_for_selection": False,
            "split": "scroll-disjoint preserved_nonbib train/dev/heldout",
            "train_heldout_scroll_intersection": sorted(overlap),
            "answer_string_removal_stress_test": True,
            "statistics": {
                "paired_cluster": "scroll",
                "bootstrap_iterations": args.bootstrap,
                "multiple_comparison_correction": "Holm across six retrieval shelves",
            },
            "shelves": shelf_metadata,
            "source_receipts": {
                "bhsa": source_receipt(args.bhsa_path),
                "extrabiblical": source_receipt(args.extrabiblical_path),
            },
            "paper_gate_missing": [
                "three trained seeds",
                "composition-disjoint hard split",
                "final tokenization-free sequence model",
            ],
        },
        "shelf_order": shelf_order,
        "results": results,
        "interpretation": interpretation,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"saved {args.output_json}")
    print(f"saved {args.output_markdown}")


if __name__ == "__main__":
    main()
