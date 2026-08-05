"""Retrieval-conditioned candidate generation for DSS span restoration.

This experiment follows the reranking-only ablation. Retrieval uses visible
context to propose one-, two-, and three-word phrases from the top documents.
The preserved-only word model then scores every proposed phrase with the same
left-to-right masked-token likelihood used by its original beam search.

The held-out word count remains unknown: every condition proposes an equal
maximum number of phrases for each candidate length. Development selects only
the per-length proposal limit. A gold-answer-removal stress test excludes every
retrieved document containing the complete answer before proposals are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cross_corpus_connections import Passage, source_receipt  # noqa: E402
from eval.cross_corpus_retrieval_ablation import (  # noqa: E402
    CONTEXT_WORDS,
    DEV_PER_LENGTH,
    MAX_CHARS,
    MAX_WORDS,
    SAMPLE_SEEDS,
    TEST_PER_LENGTH,
    candidate_cache_receipt,
    choose_device,
    contains_phrase,
    holm_adjust,
    item_from_payload,
    load_shelves,
    paired_cluster_statistics,
    retrieve_documents,
)
from eval.tf_embible_dss_benchmark import (  # noqa: E402
    Item,
    deduplicate,
    fit_penalty,
    rank_with_penalty,
    sample_items,
    sample_sha256,
    valid_word_token,
)

tlog.set_verbosity_error()

PROPOSAL_LIMITS = (0, 5, 10, 25, 50)
MAX_PROPOSALS_PER_LENGTH = max(PROPOSAL_LIMITS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval-conditioned DSS candidate generation."
    )
    parser.add_argument(
        "--word-model",
        type=Path,
        default=ROOT / "ft_msbert_span_preserved_nonbib",
    )
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=ROOT / "analysis" / "cache" / "external_retrieval_candidates.json",
    )
    parser.add_argument(
        "--injection-cache-dir",
        type=Path,
        default=ROOT / "analysis" / "cache" / "retrieval_injection",
    )
    parser.add_argument(
        "--bhsa-path",
        type=Path,
        default=Path.home()
        / "text-fabric-data"
        / "github"
        / "ETCBC"
        / "bhsa"
        / "tf"
        / "2021",
    )
    parser.add_argument(
        "--extrabiblical-path",
        type=Path,
        default=Path.home()
        / "text-fabric-data"
        / "github"
        / "ETCBC"
        / "extrabiblical"
        / "tf"
        / "0.2",
    )
    parser.add_argument("--chunk-words", type=int, default=100)
    parser.add_argument("--overlap-words", type=int, default=15)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--top-k-per-step", type=int, default=24)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "analysis" / "reports" / "cross_corpus_retrieval_injection.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "analysis" / "reports" / "CROSS_CORPUS_RETRIEVAL_INJECTION.md",
    )
    return parser.parse_args()


def passage_fingerprint(passages: Sequence[Passage]) -> str:
    digest = hashlib.sha256()
    for passage in passages:
        digest.update(passage.passage_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(passage.text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def extract_proposals(
    ranking: Sequence[tuple[int, float]],
    documents: Sequence[Passage],
    *,
    max_per_length: int,
    removed_phrase: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return relevance-ranked n-grams with an equal cap for each length."""
    scores: dict[int, dict[tuple[str, ...], float]] = {
        length: defaultdict(float) for length in range(1, MAX_WORDS + 1)
    }
    for document_index, relevance in ranking:
        tokens = documents[document_index].tokens
        if removed_phrase and contains_phrase(tokens, removed_phrase):
            continue
        for length in range(1, MAX_WORDS + 1):
            if length > len(tokens):
                continue
            phrases = {
                tuple(tokens[start : start + length])
                for start in range(len(tokens) - length + 1)
            }
            for phrase in phrases:
                if all(valid_word_token(word) for word in phrase):
                    scores[length][phrase] += relevance

    proposals = []
    for length, local_scores in scores.items():
        ranked = sorted(
            local_scores.items(),
            key=lambda row: (-row[1], row[0]),
        )[:max_per_length]
        proposals.extend(
            {
                "text": " ".join(phrase),
                "word_count": length,
                "retrieval_support": float(score),
                "proposal_rank": rank,
            }
            for rank, (phrase, score) in enumerate(ranked, start=1)
        )
    return proposals


def token_ids_for_phrase(phrase: str, tokenizer: Any) -> tuple[int, ...] | None:
    ids = []
    for word in phrase.split():
        encoded = tokenizer(
            word,
            add_special_tokens=False,
        )["input_ids"]
        if len(encoded) != 1:
            return None
        token_id = int(encoded[0])
        if tokenizer.decode([token_id]).strip() != word:
            return None
        ids.append(token_id)
    return tuple(ids)


def selected_position_logits(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Compute MLM logits only at requested positions, not every sequence token."""
    if not hasattr(model, "bert") or not hasattr(model, "cls"):
        raise TypeError("proposal scorer requires a BERT masked-LM with .bert and .cls")
    hidden = model.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).last_hidden_state
    rows = torch.arange(len(input_ids), device=input_ids.device)
    return model.cls(hidden[rows, positions])


def score_proposed_phrases(
    item: Item,
    phrases: Sequence[str],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
) -> dict[str, tuple[float, int]]:
    """Score phrases with the baseline decoder's sequential MLM likelihood."""
    if batch_size < 1:
        raise ValueError("score batch size must be positive")
    by_length: dict[int, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for phrase in sorted(set(phrases)):
        token_ids = token_ids_for_phrase(phrase, tokenizer)
        if token_ids is None or not 1 <= len(token_ids) <= MAX_WORDS:
            continue
        by_length[len(token_ids)].append((phrase, token_ids))

    output: dict[str, tuple[float, int]] = {}
    for word_count, rows in by_length.items():
        words = [
            *item.left,
            *([tokenizer.mask_token] * word_count),
            *item.right,
        ]
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        base_ids = encoding["input_ids"][0]
        positions = [
            position
            for position, token_id in enumerate(base_ids.tolist())
            if token_id == tokenizer.mask_token_id
        ]
        if len(positions) != word_count:
            continue
        base_attention = encoding["attention_mask"][0]
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            input_ids = base_ids.unsqueeze(0).repeat(len(batch), 1).to(device)
            attention = base_attention.unsqueeze(0).repeat(len(batch), 1).to(device)
            scores = torch.zeros(len(batch), device=device)
            candidate_ids = torch.tensor(
                [token_ids for _, token_ids in batch],
                dtype=torch.long,
                device=device,
            )
            with torch.inference_mode():
                for offset, position in enumerate(positions):
                    local_positions = torch.full(
                        (len(batch),),
                        position,
                        dtype=torch.long,
                        device=device,
                    )
                    logits = selected_position_logits(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention,
                        positions=local_positions,
                    )
                    log_probs = torch.log_softmax(logits, dim=-1)
                    selected = candidate_ids[:, offset]
                    scores += log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
                    input_ids[:, position] = selected
            for (phrase, _), score in zip(batch, scores.cpu().tolist()):
                output[phrase] = (float(score), word_count)
    return output


def score_record_proposals(
    records: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    label: str,
) -> list[dict[str, tuple[float, int]]]:
    """Batch proposal scoring across different DSS contexts."""
    if batch_size < 1:
        raise ValueError("score batch size must be positive")
    token_cache: dict[str, tuple[int, ...] | None] = {}
    work: dict[int, list[tuple[int, str, tuple[int, ...]]]] = defaultdict(list)
    for record_index, record in enumerate(records):
        phrases = {
            proposal["text"]
            for proposal in [
                *record["standard_proposals"],
                *record["answer_removed_proposals"],
            ]
        }
        for phrase in phrases:
            if phrase not in token_cache:
                token_cache[phrase] = token_ids_for_phrase(phrase, tokenizer)
            token_ids = token_cache[phrase]
            if token_ids is None or not 1 <= len(token_ids) <= MAX_WORDS:
                continue
            work[len(token_ids)].append((record_index, phrase, token_ids))

    output: list[dict[str, tuple[float, int]]] = [{} for _ in records]
    total = sum(len(rows) for rows in work.values())
    completed = 0
    for word_count, rows in sorted(work.items()):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            contexts = []
            positions = []
            for record_index, _, _ in batch:
                item = item_from_payload(records[record_index]["item"])
                contexts.append(
                    [
                        *item.left,
                        *([tokenizer.mask_token] * word_count),
                        *item.right,
                    ]
                )
            encoding = tokenizer(
                contexts,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            for row_index in range(len(batch)):
                local_positions = [
                    position
                    for position, token_id in enumerate(
                        encoding["input_ids"][row_index].tolist()
                    )
                    if token_id == tokenizer.mask_token_id
                ]
                if len(local_positions) != word_count:
                    raise RuntimeError(
                        "proposal scoring context did not preserve mask positions"
                    )
                positions.append(local_positions)

            input_ids = encoding["input_ids"].to(device)
            attention = encoding["attention_mask"].to(device)
            candidate_ids = torch.tensor(
                [token_ids for _, _, token_ids in batch],
                dtype=torch.long,
                device=device,
            )
            scores = torch.zeros(len(batch), device=device)
            row_ids = torch.arange(len(batch), device=device)
            with torch.inference_mode():
                for offset in range(word_count):
                    local_positions = torch.tensor(
                        [row[offset] for row in positions],
                        dtype=torch.long,
                        device=device,
                    )
                    position_logits = selected_position_logits(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention,
                        positions=local_positions,
                    )
                    selected = candidate_ids[:, offset]
                    log_probs = torch.log_softmax(position_logits, dim=-1)
                    scores += log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
                    input_ids[row_ids, local_positions] = selected
            for (record_index, phrase, _), score in zip(
                batch,
                scores.cpu().tolist(),
            ):
                output[record_index][phrase] = (float(score), word_count)
            completed += len(batch)
            if completed % (batch_size * 10) < len(batch) or completed == total:
                print(
                    f"scored {label} proposal pairs: {completed}/{total}",
                    flush=True,
                )
    return output


def baseline_raw_rows(record: dict[str, Any]) -> list[tuple[str, float, int]]:
    rows = record.get("word", record.get("baseline"))
    if rows is None:
        raise KeyError("candidate record has neither 'word' nor 'baseline' rows")
    return [
        (str(candidate), float(score), int(size)) for candidate, score, size in rows
    ]


def merge_candidates(
    baseline_rows: Sequence[tuple[str, float, int]],
    proposals: Sequence[dict[str, Any]],
    proposal_scores: dict[str, tuple[float, int]],
    *,
    limit_per_length: int,
) -> list[tuple[str, float, int]]:
    injected = []
    for proposal in proposals:
        if proposal["proposal_rank"] > limit_per_length:
            continue
        scored = proposal_scores.get(proposal["text"])
        if scored is None:
            continue
        score, word_count = scored
        injected.append((proposal["text"], score, word_count))
    return deduplicate([*baseline_rows, *injected])


def score_audit(
    baseline_rows: Sequence[tuple[str, float, int]],
    proposal_scores: dict[str, tuple[float, int]],
) -> list[float]:
    baseline = {candidate: score for candidate, score, _ in baseline_rows}
    return [
        abs(baseline[candidate] - proposal_score)
        for candidate, (proposal_score, _) in proposal_scores.items()
        if candidate in baseline
    ]


def build_injection_records(
    candidate_records: Sequence[dict[str, Any]],
    rankings: Sequence[Sequence[tuple[int, float]]],
    documents: Sequence[Passage],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    score_batch_size: int,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    for record, ranking in zip(candidate_records, rankings):
        item = item_from_payload(record["item"])
        standard = extract_proposals(
            ranking,
            documents,
            max_per_length=MAX_PROPOSALS_PER_LENGTH,
        )
        redacted = extract_proposals(
            ranking,
            documents,
            max_per_length=MAX_PROPOSALS_PER_LENGTH,
            removed_phrase=item.gold,
        )
        baseline = baseline_raw_rows(record)
        output.append(
            {
                "item": record["item"],
                "gold": record["gold"],
                "baseline": baseline,
                "standard_proposals": standard,
                "answer_removed_proposals": redacted,
                "proposal_scores": {},
            }
        )
    all_scores = score_record_proposals(
        output,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=score_batch_size,
        label=label,
    )
    audit_differences = []
    for record, scored in zip(output, all_scores):
        record["proposal_scores"] = {
            phrase: [score, word_count]
            for phrase, (score, word_count) in scored.items()
        }
        audit_differences.extend(score_audit(record["baseline"], scored))
    maximum_difference = max(audit_differences, default=0.0)
    if maximum_difference > 1e-3:
        raise RuntimeError(
            "injected-candidate scorer disagrees with baseline beam scores: "
            f"max difference {maximum_difference}"
        )
    return output, {
        "overlapping_baseline_candidates_checked": len(audit_differences),
        "maximum_absolute_score_difference": maximum_difference,
        "tolerance": 1e-3,
    }


def injection_cache_receipt(
    *,
    shelf_name: str,
    documents: Sequence[Passage],
    candidate_receipt: dict[str, Any],
    retrieval_top_k: int,
) -> dict[str, Any]:
    return {
        "shelf": shelf_name,
        "shelf_passage_fingerprint": passage_fingerprint(documents),
        "candidate_model_sha256": candidate_receipt["model_sha256"],
        "dev_sample_sha256": candidate_receipt["dev_sample_sha256"],
        "heldout_sample_sha256": candidate_receipt["heldout_sample_sha256"],
        "retrieval_top_k": retrieval_top_k,
        "max_proposals_per_length": MAX_PROPOSALS_PER_LENGTH,
        "candidate_lengths": list(range(1, MAX_WORDS + 1)),
        "proposal_query_uses_gold": False,
        "answer_removal_uses_gold_for_stress_test_only": True,
    }


def load_or_build_shelf_records(
    *,
    cache_path: Path,
    receipt: dict[str, Any],
    dev_candidates: Sequence[dict[str, Any]],
    heldout_candidates: Sequence[dict[str, Any]],
    dev_rankings: Sequence[Sequence[tuple[int, float]]],
    heldout_rankings: Sequence[Sequence[tuple[int, float]]],
    documents: Sequence[Passage],
    tokenizer: Any,
    model: Any,
    device: str,
    score_batch_size: int,
    shelf_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("receipt") != receipt:
            raise ValueError(f"injection cache receipt mismatch: {cache_path}")
        return (
            payload["dev"],
            payload["heldout"],
            {
                **payload["audit"],
                "cache": str(cache_path),
                "cache_hit": True,
            },
        )

    dev, dev_audit = build_injection_records(
        dev_candidates,
        dev_rankings,
        documents,
        tokenizer=tokenizer,
        model=model,
        device=device,
        score_batch_size=score_batch_size,
        label=f"{shelf_name} dev",
    )
    heldout, heldout_audit = build_injection_records(
        heldout_candidates,
        heldout_rankings,
        documents,
        tokenizer=tokenizer,
        model=model,
        device=device,
        score_batch_size=score_batch_size,
        label=f"{shelf_name} heldout",
    )
    audit = {
        "dev": dev_audit,
        "heldout": heldout_audit,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "receipt": receipt,
                "audit": audit,
                "dev": dev,
                "heldout": heldout,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        dev,
        heldout,
        {
            **audit,
            "cache": str(cache_path),
            "cache_hit": False,
        },
    )


def ranked_condition(
    record: dict[str, Any],
    *,
    proposal_key: str,
    limit_per_length: int,
    word_penalty: float,
) -> list[tuple[str, float]]:
    scores = {
        phrase: (float(score), int(word_count))
        for phrase, (score, word_count) in record["proposal_scores"].items()
    }
    raw = merge_candidates(
        baseline_raw_rows(record),
        record[proposal_key],
        scores,
        limit_per_length=limit_per_length,
    )
    return rank_with_penalty(raw, word_penalty)


def exact_rank(rows: Sequence[tuple[str, float]], gold: str) -> int:
    return next(
        (rank for rank, (candidate, _) in enumerate(rows) if candidate == gold),
        999,
    )


def fit_proposal_limit(
    records: Sequence[dict[str, Any]],
    *,
    word_penalty: float,
) -> dict[str, Any]:
    grid = {}
    for limit in PROPOSAL_LIMITS:
        ranks = [
            exact_rank(
                ranked_condition(
                    record,
                    proposal_key="standard_proposals",
                    limit_per_length=limit,
                    word_penalty=word_penalty,
                ),
                record["gold"],
            )
            for record in records
        ]
        grid[str(limit)] = {
            "n": len(ranks),
            "exact_top1": 100 * sum(rank == 0 for rank in ranks) / len(ranks),
            "exact_top10": 100 * sum(rank < 10 for rank in ranks) / len(ranks),
            "candidate_pool_recall": 100
            * sum(rank != 999 for rank in ranks)
            / len(ranks),
        }
    selected = max(
        PROPOSAL_LIMITS,
        key=lambda limit: (
            grid[str(limit)]["exact_top10"],
            grid[str(limit)]["exact_top1"],
            -limit,
        ),
    )
    return {
        "selection_split": "dev",
        "objective": ("exact complete-span Top-10, then Top-1, then fewer proposals"),
        "selected_proposals_per_length": selected,
        "grid": grid,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    *,
    proposal_key: str,
    limit_per_length: int,
    word_penalty: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranks = []
    cases = []
    by_words: dict[int, list[int]] = defaultdict(list)
    proposed_gold = 0
    representable_gold = 0
    for record in records:
        rows = ranked_condition(
            record,
            proposal_key=proposal_key,
            limit_per_length=limit_per_length,
            word_penalty=word_penalty,
        )
        rank = exact_rank(rows, record["gold"])
        ranks.append(rank)
        word_count = len(record["item"]["gold"])
        by_words[word_count].append(rank)
        selected_proposals = [
            proposal
            for proposal in record[proposal_key]
            if proposal["proposal_rank"] <= limit_per_length
        ]
        gold_is_proposed = any(
            proposal["text"] == record["gold"] for proposal in selected_proposals
        )
        proposed_gold += gold_is_proposed
        representable_gold += (
            gold_is_proposed and record["gold"] in record["proposal_scores"]
        )
        cases.append(
            {
                "item_id": record["item"]["item_id"],
                "scroll": record["item"]["scroll"],
                "word_count": word_count,
                "gold": record["gold"],
                "rank": rank if rank != 999 else None,
                "hit_top1": rank == 0,
                "hit_top10": rank < 10,
                "gold_proposed": gold_is_proposed,
                "gold_proposal_model_representable": (
                    gold_is_proposed and record["gold"] in record["proposal_scores"]
                ),
                "top10": [candidate for candidate, _ in rows[:10]],
            }
        )

    def metrics(local_ranks: Sequence[int]) -> dict[str, Any]:
        return {
            "n": len(local_ranks),
            "exact_top1": 100
            * sum(rank == 0 for rank in local_ranks)
            / len(local_ranks),
            "exact_top5": 100
            * sum(rank < 5 for rank in local_ranks)
            / len(local_ranks),
            "exact_top10": 100
            * sum(rank < 10 for rank in local_ranks)
            / len(local_ranks),
            "exact_top20": 100
            * sum(rank < 20 for rank in local_ranks)
            / len(local_ranks),
            "mean_reciprocal_rank": float(
                np.mean(
                    [1.0 / (rank + 1) if rank != 999 else 0.0 for rank in local_ranks]
                )
            ),
            "candidate_pool_recall": 100
            * sum(rank != 999 for rank in local_ranks)
            / len(local_ranks),
        }

    return {
        **metrics(ranks),
        "gold_proposed": proposed_gold,
        "gold_proposal_model_representable": representable_gold,
        "by_word_count": {
            str(word_count): metrics(local_ranks)
            for word_count, local_ranks in sorted(by_words.items())
        },
    }, cases


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["results"]["no_injection"]["standard"]
    lines = [
        "# Retrieval-conditioned candidate generation",
        "",
        "Status: **single-checkpoint exploratory ablation**, not a final paper result.",
        "Retrieval proposes phrases, but the preserved-only word model scores them",
        "before they enter the unknown-length candidate ranking.",
        "",
        "## Exact complete-span recovery",
        "",
        "| Retrieval shelf | Dev proposals/length | Top-1 | Top-10 | Candidate recall | Delta Top-10 | 95% cluster CI | Holm p | Answer-removed Top-10 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |",
        (
            f"| No injection | 0 | {baseline['exact_top1']:.1f}% | "
            f"{baseline['exact_top10']:.1f}% | "
            f"{baseline['candidate_pool_recall']:.1f}% | — | — | — | — |"
        ),
    ]
    for name in report["shelf_order"]:
        row = report["results"][name]
        standard = row["standard"]
        stress = row["answer_removed_stress"]
        statistics = row["statistics"]
        low, high = statistics["paired_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {name} | "
            f"{row['dev_fit']['selected_proposals_per_length']} | "
            f"{standard['exact_top1']:.1f}% | "
            f"{standard['exact_top10']:.1f}% | "
            f"{standard['candidate_pool_recall']:.1f}% | "
            f"{statistics['delta_exact_top10_points']:+.1f} | "
            f"[{low:+.1f}, {high:+.1f}] | "
            f"{statistics['holm_adjusted_p']:.4f} | "
            f"{stress['exact_top10']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Maximum-injection diagnostic (not selected)",
            "",
            (
                "The table below forces 50 proposals per candidate length only to "
                "locate the bottleneck. It is not a development-selected system."
            ),
            "",
            "| Retrieval shelf | Candidate recall | Gold proposed | Top-10 | Answer-removed recall |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in report["shelf_order"]:
        row = report["results"][name]
        maximum = row["maximum_injection_diagnostic"]
        redacted = row["maximum_injection_answer_removed"]
        lines.append(
            f"| {name} | {maximum['candidate_pool_recall']:.1f}% | "
            f"{maximum['gold_proposed']}/{maximum['n']} | "
            f"{maximum['exact_top10']:.1f}% | "
            f"{redacted['candidate_pool_recall']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "Every shelf proposes the same maximum number of one-, two-, and",
            "three-word phrases, so the hidden word count is not supplied. Retrieval",
            "queries contain only the visible context. The answer-removal stress test",
            "excludes every retrieved document containing the complete held-out answer",
            "before proposal extraction.",
            "",
            "This run uses one trained checkpoint and the frozen 300-span pilot.",
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
    if not args.candidate_cache.is_file():
        raise FileNotFoundError(
            f"frozen candidate cache not found: {args.candidate_cache}"
        )
    if not 1 <= args.score_batch_size <= 512:
        raise ValueError("--score-batch-size must be between 1 and 512")
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
    expected_receipt = candidate_cache_receipt(
        model_path=model_path,
        dev_items=dev_items,
        heldout_items=heldout_items,
        beam_width=args.beam_width,
        top_k_per_step=args.top_k_per_step,
    )
    candidate_payload = json.loads(args.candidate_cache.read_text(encoding="utf-8"))
    if candidate_payload.get("receipt") != expected_receipt:
        raise ValueError("frozen candidate cache receipt mismatch")
    dev_candidates = candidate_payload["dev"]
    heldout_candidates = candidate_payload["heldout"]
    word_penalty = fit_penalty(dev_candidates, "word")

    shelves, shelf_metadata = load_shelves(
        bhsa_path=args.bhsa_path,
        extrabiblical_path=args.extrabiblical_path,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=True,
        local_files_only=True,
    )
    model = (
        AutoModelForMaskedLM.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        .to(device)
        .eval()
    )

    baseline_records = [
        {
            "item": record["item"],
            "gold": record["gold"],
            "baseline": record["word"],
            "standard_proposals": [],
            "answer_removed_proposals": [],
            "proposal_scores": {},
        }
        for record in heldout_candidates
    ]
    baseline_metrics, baseline_cases = summarize(
        baseline_records,
        proposal_key="standard_proposals",
        limit_per_length=0,
        word_penalty=word_penalty,
    )
    results: dict[str, Any] = {
        "no_injection": {
            "standard": baseline_metrics,
            "cases": baseline_cases,
        }
    }
    p_values = {}
    shelf_order = list(shelves)
    cache_metadata = {}
    for shelf_index, (name, documents) in enumerate(shelves.items()):
        print(
            f"building injection condition {name}: {len(documents)} passages",
            flush=True,
        )
        all_items = [*dev_items, *heldout_items]
        rankings, retrieval_protocol = retrieve_documents(
            all_items,
            documents,
            top_k=args.retrieval_top_k,
        )
        receipt = injection_cache_receipt(
            shelf_name=name,
            documents=documents,
            candidate_receipt=expected_receipt,
            retrieval_top_k=args.retrieval_top_k,
        )
        dev_records, heldout_records, cache_audit = load_or_build_shelf_records(
            cache_path=args.injection_cache_dir / f"{name}.json",
            receipt=receipt,
            dev_candidates=dev_candidates,
            heldout_candidates=heldout_candidates,
            dev_rankings=rankings[: len(dev_items)],
            heldout_rankings=rankings[len(dev_items) :],
            documents=documents,
            tokenizer=tokenizer,
            model=model,
            device=device,
            score_batch_size=args.score_batch_size,
            shelf_name=name,
        )
        cache_metadata[name] = cache_audit
        dev_fit = fit_proposal_limit(
            dev_records,
            word_penalty=word_penalty,
        )
        selected = int(dev_fit["selected_proposals_per_length"])
        standard, standard_cases = summarize(
            heldout_records,
            proposal_key="standard_proposals",
            limit_per_length=selected,
            word_penalty=word_penalty,
        )
        stress, stress_cases = summarize(
            heldout_records,
            proposal_key="answer_removed_proposals",
            limit_per_length=selected,
            word_penalty=word_penalty,
        )
        maximum_diagnostic, _ = summarize(
            heldout_records,
            proposal_key="standard_proposals",
            limit_per_length=MAX_PROPOSALS_PER_LENGTH,
            word_penalty=word_penalty,
        )
        maximum_answer_removed, _ = summarize(
            heldout_records,
            proposal_key="answer_removed_proposals",
            limit_per_length=MAX_PROPOSALS_PER_LENGTH,
            word_penalty=word_penalty,
        )
        statistics = paired_cluster_statistics(
            baseline_cases,
            standard_cases,
            iterations=args.bootstrap,
            seed=2000 + shelf_index,
        )
        p_values[name] = statistics["cluster_sign_flip_p"]
        results[name] = {
            "retrieval_protocol": retrieval_protocol,
            "dev_fit": dev_fit,
            "standard": standard,
            "answer_removed_stress": stress,
            "maximum_injection_diagnostic": maximum_diagnostic,
            "maximum_injection_answer_removed": maximum_answer_removed,
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
            f"{best_name} produced the strongest held-out Top-10 gain "
            f"({best_delta:+.1f} points) with a scroll-cluster interval above "
            "zero. The answer-removal result determines whether this is robust "
            "contextual assistance or primarily verbatim answer transfer."
        )
    elif best_delta > 0:
        interpretation = (
            f"{best_name} produced the largest observed Top-10 gain "
            f"({best_delta:+.1f} points), but its scroll-cluster interval includes "
            "zero. This is inconclusive exploratory evidence."
        )
    else:
        maximum_pool = max(
            results[name]["maximum_injection_diagnostic"]["candidate_pool_recall"]
            for name in shelf_order
        )
        interpretation = (
            "Development-selected retrieval proposals did not improve held-out "
            "exact Top-10. Even the unselected maximum-injection diagnostic left "
            f"Top-10 unchanged while raising candidate recall as high as "
            f"{maximum_pool:.1f}%. Retrieval can supply missing answers, but this "
            "word model assigns them scores below the useful ranking boundary. "
            "The negative result is retained."
        )

    report = {
        "status": "single_checkpoint_exploratory_candidate_injection",
        "protocol": {
            "target": (
                "synthetic lacunae made by hiding contiguous physically preserved "
                "non-biblical DSS words"
            ),
            "modern_reconstructions_used": False,
            "candidate_cache_receipt": expected_receipt,
            "retrieval_query": "visible eight-word left and right context only",
            "retrieval_top_k": args.retrieval_top_k,
            "proposal_limits_per_length": PROPOSAL_LIMITS,
            "equal_proposal_limit_for_word_counts": list(range(1, MAX_WORDS + 1)),
            "proposal_scoring": (
                "preserved-only word model sequential masked-token log likelihood"
            ),
            "word_length_penalty_selected_on_dev": word_penalty,
            "dev_items": len(dev_items),
            "heldout_items": len(heldout_items),
            "dev_eligible_by_words": dev_eligible,
            "heldout_eligible_by_words": heldout_eligible,
            "dev_sample_sha256": sample_sha256(dev_items),
            "heldout_sample_sha256": sample_sha256(heldout_items),
            "heldout_used_for_selection": False,
            "answer_string_removal_stress_test": True,
            "shelves": shelf_metadata,
            "source_receipts": {
                "bhsa": source_receipt(args.bhsa_path),
                "extrabiblical": source_receipt(args.extrabiblical_path),
            },
            "scoring_cache_audits": cache_metadata,
            "statistics": {
                "paired_cluster": "scroll",
                "bootstrap_iterations": args.bootstrap,
                "multiple_comparison_correction": "Holm across six shelves",
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
