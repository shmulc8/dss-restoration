"""Embible-style character/word evaluation on synthetically damaged DSS text.

The primary conditions do not receive the gold character length, word count, or
word boundaries. Targets are contiguous physically preserved words hidden
artificially for evaluation: they are synthetic lacunae, not naturally damaged
manuscript locations. Modern reconstructions never become inputs, labels,
retrieval records, or gold answers.

Systems:
* uwc_word: preserved-only MsBERT, searching over 1..N word masks;
* char_unknown: TavBERT, searching over character lengths and whitespace;
* embible_overlap_ensemble: overlap of Top-5 character and word candidates,
  averaging normalized model scores and falling back to character candidates;
* rank_ensemble: additional dev-tuned rank fusion baseline;
* cwc_word_oracle: word candidates filtered by gold word lengths;
* char_oracle_length: per-character Hit@K with gold total slot count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.preserved_corpus import GAP_TOKEN, load_chunks

tlog.set_verbosity_error()

HEBREW = set(chr(codepoint) for codepoint in range(0x05D0, 0x05EB))
LENGTH_PENALTIES = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
ENSEMBLE_WEIGHTS = tuple(index / 10 for index in range(11))


@dataclass(frozen=True)
class Item:
    item_id: str
    scroll: str
    left: tuple[str, ...]
    gold: tuple[str, ...]
    right: tuple[str, ...]

    @property
    def gold_text(self) -> str:
        return " ".join(self.gold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--word-model",
        type=Path,
        default=ROOT / "ft_msbert_span_preserved_nonbib",
    )
    parser.add_argument("--char-model", default="tau/tavbert-he")
    parser.add_argument("--dev-per-length", type=int, default=5)
    parser.add_argument("--test-per-length", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=18)
    parser.add_argument("--context-words", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--top-k-per-step", type=int, default=24)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "analysis" / "reports" / "embible_dss_benchmark.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "analysis" / "reports" / "EMBIBLE_DSS_BENCHMARK.md",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def is_hebrew_word(value: str) -> bool:
    return len(value) >= 2 and all(character in HEBREW for character in value)


def is_hebrew_token(value: str) -> bool:
    return bool(value) and all(character in HEBREW for character in value)


def candidate_spans(
    split: str,
    *,
    max_words: int,
    max_chars: int,
    context_words: int,
) -> dict[int, list[Item]]:
    by_length: dict[int, list[Item]] = defaultdict(list)
    for row in load_chunks(split):
        words = row["text"].split()
        segment_start = 0
        while segment_start < len(words):
            while segment_start < len(words) and words[segment_start] == GAP_TOKEN:
                segment_start += 1
            segment_end = segment_start
            while segment_end < len(words) and words[segment_end] != GAP_TOKEN:
                segment_end += 1
            segment = words[segment_start:segment_end]
            for length in range(1, max_words + 1):
                minimum = context_words * 2 + length
                if len(segment) < minimum:
                    continue
                for local_start in range(context_words, len(segment) - context_words - length + 1):
                    gold = tuple(segment[local_start:local_start + length])
                    if not all(is_hebrew_word(word) for word in gold):
                        continue
                    gold_text = " ".join(gold)
                    if len(gold_text) > max_chars:
                        continue
                    left = tuple(segment[local_start - context_words:local_start])
                    right = tuple(
                        segment[
                            local_start + length:
                            local_start + length + context_words
                        ]
                    )
                    if not all(is_hebrew_token(word) for word in (*left, *right)):
                        continue
                    absolute_start = segment_start + local_start
                    by_length[length].append(
                        Item(
                            item_id=(
                                f"{split}:{row['scroll']}:{row['chunk_index']}:"
                                f"{absolute_start}:{length}"
                            ),
                            scroll=row["scroll"],
                            left=left,
                            gold=gold,
                            right=right,
                        )
                    )
            segment_start = max(segment_end, segment_start + 1)
    return by_length


def sample_items(
    split: str,
    *,
    per_length: int,
    max_words: int,
    max_chars: int,
    context_words: int,
    seed: int,
) -> tuple[list[Item], dict[str, int]]:
    pools = candidate_spans(
        split,
        max_words=max_words,
        max_chars=max_chars,
        context_words=context_words,
    )
    rng = random.Random(seed)
    selected: list[Item] = []
    eligible = {}
    for length in range(1, max_words + 1):
        pool = pools.get(length, [])
        eligible[str(length)] = len(pool)
        rng.shuffle(pool)
        selected.extend(pool[:per_length])
    return selected, eligible


def valid_word_token(token: str) -> bool:
    return bool(token) and all(character in HEBREW for character in token)


def word_candidates(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    max_words: int,
    beam_width: int,
    top_k_per_step: int,
) -> list[tuple[str, float, int]]:
    results: list[tuple[str, float, int]] = []
    for word_count in range(1, max_words + 1):
        words = [*left, *([tokenizer.mask_token] * word_count), *right]
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"][0].clone()
        positions = [
            position
            for position, token_id in enumerate(input_ids.tolist())
            if token_id == tokenizer.mask_token_id
        ]
        if len(positions) != word_count:
            continue
        beams: list[tuple[float, torch.Tensor, tuple[str, ...]]] = [
            (0.0, input_ids, ())
        ]
        for position in positions:
            batch_ids = torch.stack([beam[1] for beam in beams]).to(device)
            attention = encoding["attention_mask"].repeat(len(beams), 1).to(device)
            with torch.inference_mode():
                logits = model(
                    input_ids=batch_ids,
                    attention_mask=attention,
                ).logits[:, position].cpu()
            expanded = []
            for beam_index, (score, ids, predicted) in enumerate(beams):
                log_probs = torch.log_softmax(logits[beam_index], dim=-1)
                top = torch.topk(log_probs, min(top_k_per_step * 4, len(log_probs)))
                accepted = 0
                for token_id, token_score in zip(
                    top.indices.tolist(),
                    top.values.tolist(),
                ):
                    token = tokenizer.decode([token_id]).strip()
                    if not valid_word_token(token):
                        continue
                    new_ids = ids.clone()
                    new_ids[position] = token_id
                    expanded.append(
                        (score + token_score, new_ids, (*predicted, token))
                    )
                    accepted += 1
                    if accepted >= top_k_per_step:
                        break
            beams = sorted(expanded, key=lambda row: -row[0])[:beam_width]
            if not beams:
                break
        results.extend(
            (" ".join(predicted), score, word_count)
            for score, _, predicted in beams
            if len(predicted) == word_count
        )
    return deduplicate(results)


def valid_char_prefix(tokens: tuple[str, ...], remaining: int) -> bool:
    if not tokens:
        return False
    if tokens[0] == " ":
        return False
    if len(tokens) >= 2 and tokens[-1] == tokens[-2] == " ":
        return False
    if remaining == 0 and tokens[-1] == " ":
        return False
    return True


def char_candidates(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    max_chars: int,
    beam_width: int,
    top_k_per_step: int,
) -> tuple[list[tuple[str, float, int]], dict[int, torch.Tensor]]:
    allowed_tokens = sorted(HEBREW | {" "})
    allowed_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(token) for token in allowed_tokens],
        dtype=torch.long,
    )
    prefix = tokenizer(
        " ".join(left) + " ",
        add_special_tokens=False,
    )["input_ids"]
    suffix = tokenizer(
        " " + " ".join(right),
        add_special_tokens=False,
    )["input_ids"]
    sequences = []
    mask_positions_by_length = {}
    for char_count in range(2, max_chars + 1):
        ids = [
            tokenizer.cls_token_id,
            *prefix,
            *([tokenizer.mask_token_id] * char_count),
            *suffix,
            tokenizer.sep_token_id,
        ]
        sequences.append(torch.tensor(ids, dtype=torch.long))
        mask_start = 1 + len(prefix)
        mask_positions_by_length[char_count] = list(
            range(mask_start, mask_start + char_count)
        )
    max_sequence = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_sequence),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention = torch.zeros_like(input_ids)
    for row_index, sequence in enumerate(sequences):
        input_ids[row_index, :len(sequence)] = sequence
        attention[row_index, :len(sequence)] = 1
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention.to(device),
        ).logits.cpu()

    results = []
    oracle_logits = {}
    for row_index, char_count in enumerate(range(2, max_chars + 1)):
        positions = mask_positions_by_length[char_count]
        position_logits = logits[row_index, positions][:, allowed_ids]
        oracle_logits[char_count] = position_logits
        beams: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
        for offset, row_logits in enumerate(position_logits):
            log_probs = torch.log_softmax(row_logits, dim=-1)
            top = torch.topk(log_probs, min(top_k_per_step, len(allowed_tokens)))
            expanded = []
            remaining = char_count - offset - 1
            for score, prefix_tokens in beams:
                for local_id, token_score in zip(
                    top.indices.tolist(),
                    top.values.tolist(),
                ):
                    tokens = (*prefix_tokens, allowed_tokens[local_id])
                    if valid_char_prefix(tokens, remaining):
                        expanded.append((score + token_score, tokens))
            beams = sorted(expanded, key=lambda row: -row[0])[:beam_width]
            if not beams:
                break
        results.extend(
            ("".join(tokens), score, char_count)
            for score, tokens in beams
            if tokens and tokens[-1] != " "
        )
    return deduplicate(results), oracle_logits


def deduplicate(
    rows: Iterable[tuple[str, float, int]],
) -> list[tuple[str, float, int]]:
    best: dict[str, tuple[float, int]] = {}
    for text, score, size in rows:
        previous = best.get(text)
        if previous is None or score > previous[0]:
            best[text] = (score, size)
    return [
        (text, score, size)
        for text, (score, size) in sorted(
            best.items(),
            key=lambda row: (-row[1][0], row[0]),
        )
    ]


def rank_with_penalty(
    rows: list[tuple[str, float, int]],
    penalty: float,
) -> list[tuple[str, float]]:
    return [
        (text, score + penalty * size)
        for text, score, size in sorted(
            rows,
            key=lambda row: (-(row[1] + penalty * row[2]), row[0]),
        )
    ]


def exact_rank(rows: list[tuple[str, float]], gold: str) -> int:
    return next(
        (rank for rank, (text, _) in enumerate(rows) if text == gold),
        999,
    )


def fit_penalty(
    records: list[dict[str, Any]],
    key: str,
) -> float:
    def score(penalty: float) -> tuple[int, int, float]:
        ranks = [
            exact_rank(rank_with_penalty(record[key], penalty), record["gold"])
            for record in records
        ]
        return (
            sum(rank < 10 for rank in ranks),
            sum(rank == 0 for rank in ranks),
            -abs(penalty),
        )

    return max(LENGTH_PENALTIES, key=score)


def rank_scores(rows: list[tuple[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    denominator = max(1, len(rows) - 1)
    return {
        text: 1.0 - rank / denominator
        for rank, (text, _) in enumerate(rows)
    }


def ensemble_candidates(
    word_rows: list[tuple[str, float]],
    char_rows: list[tuple[str, float]],
    weight: float,
) -> list[tuple[str, float]]:
    word_scores = rank_scores(word_rows)
    char_scores = rank_scores(char_rows)
    candidates = set(word_scores) | set(char_scores)
    return sorted(
        (
            (
                candidate,
                weight * word_scores.get(candidate, -0.25)
                + (1.0 - weight) * char_scores.get(candidate, -0.25),
            )
            for candidate in candidates
        ),
        key=lambda row: (-row[1], row[0]),
    )


def normalize_candidate_scores(
    rows: list[tuple[str, float]],
) -> dict[str, float]:
    if not rows:
        return {}
    values = [score for _, score in rows]
    low = min(values)
    high = max(values)
    if high == low:
        return {text: 1.0 for text, _ in rows}
    return {
        text: (score - low) / (high - low)
        for text, score in rows
    }


def embible_overlap_candidates(
    word_rows: list[tuple[str, float]],
    char_rows: list[tuple[str, float]],
    *,
    char_limit: int = 5,
    word_limit: int = 1000,
) -> list[tuple[str, float]]:
    """Apply the paper-described Embible overlap/average/fallback rule.

    Our candidate generator may yield fewer than ``word_limit`` sequences; the
    actual pool size is recorded in the report protocol. This intentionally
    follows the paper rather than reproducing discrepancies in the public
    backend implementation.
    """
    limited_word = word_rows[:word_limit]
    limited_char = char_rows[:char_limit]
    word_scores = normalize_candidate_scores(limited_word)
    char_scores = normalize_candidate_scores(limited_char)
    overlap = set(word_scores) & set(char_scores)
    if overlap:
        return sorted(
            (
                (
                    candidate,
                    (word_scores[candidate] + char_scores[candidate]) / 2,
                )
                for candidate in overlap
            ),
            key=lambda row: (-row[1], row[0]),
        )
    return limited_char


def fit_ensemble_weight(
    records: list[dict[str, Any]],
    *,
    word_penalty: float,
    char_penalty: float,
) -> float:
    def score(weight: float) -> tuple[int, int, float]:
        ranks = []
        for record in records:
            rows = ensemble_candidates(
                rank_with_penalty(record["word"], word_penalty),
                rank_with_penalty(record["char"], char_penalty),
                weight,
            )
            ranks.append(exact_rank(rows, record["gold"]))
        return (
            sum(rank < 10 for rank in ranks),
            sum(rank == 0 for rank in ranks),
            -abs(weight - 0.5),
        )

    return max(ENSEMBLE_WEIGHTS, key=score)


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def boundary_f1(prediction: str, gold: str) -> float:
    predicted = {index for index, character in enumerate(prediction) if character == " "}
    expected = {index for index, character in enumerate(gold) if character == " "}
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted)
    recall = true_positive / len(expected)
    return 2 * precision * recall / (precision + recall) if true_positive else 0.0


def summarize(rows_by_item: list[tuple[Item, list[tuple[str, float]]]]) -> dict[str, Any]:
    ranks = []
    cers = []
    boundary_scores = []
    word_count_errors = []
    word_hit1 = 0
    word_hit5 = 0
    word_total = 0
    failures = 0
    for item, rows in rows_by_item:
        rank = exact_rank(rows, item.gold_text)
        ranks.append(rank)
        if not rows:
            failures += 1
            prediction = ""
        else:
            prediction = rows[0][0]
        cers.append(levenshtein(prediction, item.gold_text) / max(1, len(item.gold_text)))
        boundary_scores.append(boundary_f1(prediction, item.gold_text))
        predicted_words = len(prediction.split()) if prediction else 0
        word_count_errors.append(abs(predicted_words - len(item.gold)))
        for position, gold_word in enumerate(item.gold):
            word_total += 1
            word_hit1 += any(
                len(candidate.split()) == len(item.gold)
                and candidate.split()[position] == gold_word
                for candidate, _ in rows[:1]
            )
            word_hit5 += any(
                len(candidate.split()) == len(item.gold)
                and candidate.split()[position] == gold_word
                for candidate, _ in rows[:5]
            )
    total = len(rows_by_item)
    return {
        "n": total,
        "top1": 100 * sum(rank == 0 for rank in ranks) / total,
        "top5": 100 * sum(rank < 5 for rank in ranks) / total,
        "top10": 100 * sum(rank < 10 for rank in ranks) / total,
        "mean_top1_cer": sum(cers) / total,
        "mean_boundary_f1": sum(boundary_scores) / total,
        "mean_word_count_error": sum(word_count_errors) / total,
        "decoder_failure_rate": failures / total,
        "word_hit1": 100 * word_hit1 / word_total if word_total else 0.0,
        "word_hit5": 100 * word_hit5 / word_total if word_total else 0.0,
    }


def oracle_word_rows(
    item: Item,
    rows: list[tuple[str, float, int]],
) -> list[tuple[str, float]]:
    lengths = tuple(len(word) for word in item.gold)
    return [
        (text, score)
        for text, score, _ in rows
        if tuple(len(word) for word in text.split()) == lengths
    ]


def oracle_char_hits(
    records: list[dict[str, Any]],
) -> dict[str, float | int]:
    allowed_tokens = sorted(HEBREW | {" "})
    total = 0
    hit1 = 0
    hit5 = 0
    for record in records:
        gold = record["gold"]
        logits = record["oracle_logits"].get(len(gold))
        if logits is None or len(logits) != len(gold):
            continue
        for position, character in enumerate(gold):
            if character not in allowed_tokens:
                continue
            local_gold = allowed_tokens.index(character)
            ranking = torch.argsort(logits[position], descending=True).tolist()
            rank = ranking.index(local_gold)
            total += 1
            hit1 += rank == 0
            hit5 += rank < 5
    return {
        "n_characters": total,
        "char_hit1": 100 * hit1 / total if total else 0.0,
        "char_hit5": 100 * hit5 / total if total else 0.0,
    }


def generate_records(
    items: list[Item],
    *,
    word_tokenizer: Any,
    word_model: Any,
    char_tokenizer: Any,
    char_model: Any,
    device: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(items, 1):
        word_rows = word_candidates(
            item.left,
            item.right,
            tokenizer=word_tokenizer,
            model=word_model,
            device=device,
            max_words=args.max_words,
            beam_width=args.beam_width,
            top_k_per_step=args.top_k_per_step,
        )
        char_rows, oracle_logits = char_candidates(
            item.left,
            item.right,
            tokenizer=char_tokenizer,
            model=char_model,
            device=device,
            max_chars=args.max_chars,
            beam_width=args.beam_width,
            top_k_per_step=args.top_k_per_step,
        )
        records.append(
            {
                "item": item,
                "gold": item.gold_text,
                "word": word_rows,
                "char": char_rows,
                "oracle_logits": oracle_logits,
            }
        )
        print(f"generated candidates: {index}/{len(items)}", flush=True)
    return records


def evaluate_records(
    records: list[dict[str, Any]],
    *,
    word_penalty: float,
    char_penalty: float,
    ensemble_weight: float,
) -> dict[str, Any]:
    systems: dict[str, list[tuple[Item, list[tuple[str, float]]]]] = {
        "uwc_word": [],
        "char_unknown": [],
        "embible_overlap_ensemble": [],
        "rank_ensemble": [],
        "cwc_word_oracle": [],
    }
    for record in records:
        word_rows = rank_with_penalty(record["word"], word_penalty)
        char_rows = rank_with_penalty(record["char"], char_penalty)
        systems["uwc_word"].append((record["item"], word_rows))
        systems["char_unknown"].append((record["item"], char_rows))
        systems["embible_overlap_ensemble"].append(
            (
                record["item"],
                embible_overlap_candidates(word_rows, char_rows),
            )
        )
        systems["rank_ensemble"].append(
            (
                record["item"],
                ensemble_candidates(word_rows, char_rows, ensemble_weight),
            )
        )
        systems["cwc_word_oracle"].append(
            (record["item"], oracle_word_rows(record["item"], record["word"]))
        )
    overall = {name: summarize(rows) for name, rows in systems.items()}
    by_word_count = {}
    for word_count in sorted({len(record["item"].gold) for record in records}):
        by_word_count[str(word_count)] = {
            name: summarize(
                [
                    row
                    for row in system_rows
                    if len(row[0].gold) == word_count
                ]
            )
            for name, system_rows in systems.items()
        }
    cases = []
    for index, record in enumerate(records):
        item = record["item"]
        cases.append(
            {
                "item_id": item.item_id,
                "scroll": item.scroll,
                "left_context": list(item.left),
                "gold": list(item.gold),
                "right_context": list(item.right),
                "top10": {
                    name: [candidate for candidate, _ in rows[index][1][:10]]
                    for name, rows in systems.items()
                },
            }
        )
    return {
        **overall,
        "by_word_count": by_word_count,
        "char_oracle_length": oracle_char_hits(records),
        "cases": cases,
    }


def sample_sha256(items: list[Item]) -> str:
    payload = "\n".join(
        f"{item.item_id}\t{item.gold_text}"
        for item in sorted(items, key=lambda item: item.item_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_huggingface_revision(model_id: str) -> str | None:
    reference = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "refs"
        / "main"
    )
    return reference.read_text(encoding="utf-8").strip() if reference.is_file() else None


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for name in (
        "uwc_word",
        "char_unknown",
        "embible_overlap_ensemble",
        "rank_ensemble",
        "cwc_word_oracle",
    ):
        result = report["results"][name]
        rows.append(
            f"| {name} | {result['n']} | {result['top1']:.1f}% | "
            f"{result['top5']:.1f}% | {result['top10']:.1f}% | "
            f"{result['word_hit1']:.1f}% | {result['word_hit5']:.1f}% | "
            f"{result['mean_top1_cer']:.3f} | "
            f"{result['mean_boundary_f1']:.3f} | "
            f"{result['mean_word_count_error']:.3f} | "
            f"{100 * result['decoder_failure_rate']:.1f}% |"
        )
    oracle = report["results"]["char_oracle_length"]
    severity_rows = []
    for word_count, damage_label in (
        ("1", "5.9%"),
        ("2", "11.1%"),
        ("3", "15.8%"),
    ):
        result = report["results"]["by_word_count"][word_count]
        severity_rows.append(
            f"| {damage_label} / {word_count} word(s) | "
            f"{result['uwc_word']['top10']:.1f}% | "
            f"{result['char_unknown']['top10']:.1f}% | "
            f"{result['embible_overlap_ensemble']['top10']:.1f}% | "
            f"{result['rank_ensemble']['top10']:.1f}% |"
        )
    return f"""# Embible-style synthetic-damage DSS benchmark

## Held-out synthetic-lacuna results

| System | N spans | Exact Top-1 | Exact Top-5 | Exact Top-10 | Seq WordHit@1 | Seq WordHit@5 | Top-1 CER | Boundary F1 | Word-count MAE | Failure |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Character oracle-length diagnostic: CharHit@1
{oracle['char_hit1']:.1f}%, CharHit@5 {oracle['char_hit5']:.1f}% over
{oracle['n_characters']} characters.

## Contiguous damage severity

With eight context words on each side, hiding one, two, or three words removes
5.9%, 11.1%, or 15.8% of the displayed word sequence. These are close to
Embible's 5%, 10%, and 15% conditions, but the DSS targets remain contiguous.

| Approximate masked share | UWC Top-10 | Character Top-10 | Embible ensemble Top-10 | Rank ensemble Top-10 |
| :--- | ---: | ---: | ---: | ---: |
{chr(10).join(severity_rows)}

## Interpretation

`uwc_word`, `char_unknown`, `embible_overlap_ensemble`, and `rank_ensemble` do
not receive the gold span length, word count, or word boundaries.
`cwc_word_oracle` and `char_oracle_length` are ceiling diagnostics and must not
be compared as real-world systems.

`embible_overlap_ensemble` follows the rule described in the Embible paper:
intersect the Top-5 character sequences with the word candidates, average
normalized scores, and fall back to the character list when no overlap exists.
The candidate pool is smaller than the paper's Top-1,000 pool and is reported
as a scaled paper-protocol adaptation, not an exact code reproduction.
`rank_ensemble` is our separate dev-fitted baseline.

Targets are contiguous physically preserved words that we hide artificially in
reconstruction-free held-out DSS scrolls. They are **synthetic lacunae, not real
manuscript lacunae**. This is directly analogous to Embible's evaluation on
randomly masked Tanakh verses, which Embible itself lists as a limitation. The
character model is the cached TavBERT base checkpoint;
it has not yet been fine-tuned on the preserved-only DSS corpus. This report is
therefore an implemented baseline matrix, not a final paper result.

`Seq WordHit@K` asks whether a gold word appears in its correct position within
one of the top K complete sequences. It is stricter than, and not numerically
identical to, Embible's independently calculated WordHit@K.
"""


def main() -> None:
    args = parse_args()
    if args.dev_per_length < 1 or args.test_per_length < 1:
        raise ValueError("sample sizes must be positive")
    if args.max_words < 1 or args.max_chars < 2:
        raise ValueError("max_words must be >=1 and max_chars must be >=2")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dev_items, dev_eligible = sample_items(
        "dev",
        per_length=args.dev_per_length,
        max_words=args.max_words,
        max_chars=args.max_chars,
        context_words=args.context_words,
        seed=71,
    )
    test_items, test_eligible = sample_items(
        "heldout",
        per_length=args.test_per_length,
        max_words=args.max_words,
        max_chars=args.max_chars,
        context_words=args.context_words,
        seed=73,
    )
    expected_dev = args.dev_per_length * args.max_words
    expected_test = args.test_per_length * args.max_words
    if len(dev_items) != expected_dev or len(test_items) != expected_test:
        raise RuntimeError(
            f"insufficient eligible spans: dev={len(dev_items)}/{expected_dev}, "
            f"test={len(test_items)}/{expected_test}"
        )
    word_source = args.word_model.resolve()
    if not word_source.is_dir():
        raise FileNotFoundError(f"word checkpoint not found: {word_source}")
    word_tokenizer = AutoTokenizer.from_pretrained(
        str(word_source),
        use_fast=True,
        local_files_only=True,
    )
    word_model = AutoModelForMaskedLM.from_pretrained(
        str(word_source),
        local_files_only=True,
    ).to(device).eval()
    char_tokenizer = AutoTokenizer.from_pretrained(
        args.char_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    char_model = AutoModelForMaskedLM.from_pretrained(
        args.char_model,
        local_files_only=args.local_files_only,
    ).to(device).eval()

    print(f"device={device}; dev={len(dev_items)}; heldout={len(test_items)}")
    print("Generating development candidates...")
    dev_records = generate_records(
        dev_items,
        word_tokenizer=word_tokenizer,
        word_model=word_model,
        char_tokenizer=char_tokenizer,
        char_model=char_model,
        device=device,
        args=args,
    )
    word_penalty = fit_penalty(dev_records, "word")
    char_penalty = fit_penalty(dev_records, "char")
    ensemble_weight = fit_ensemble_weight(
        dev_records,
        word_penalty=word_penalty,
        char_penalty=char_penalty,
    )
    print(
        f"dev fit: word_penalty={word_penalty}, "
        f"char_penalty={char_penalty}, ensemble_word_weight={ensemble_weight}"
    )
    print("Generating held-out candidates...")
    test_records = generate_records(
        test_items,
        word_tokenizer=word_tokenizer,
        word_model=word_model,
        char_tokenizer=char_tokenizer,
        char_model=char_model,
        device=device,
        args=args,
    )
    evaluation = evaluate_records(
        test_records,
        word_penalty=word_penalty,
        char_penalty=char_penalty,
        ensemble_weight=ensemble_weight,
    )
    word_checkpoint = word_source / "model.safetensors"
    report = {
        "protocol": {
            "target": "synthetic lacunae made by hiding contiguous physically preserved DSS words",
            "natural_lacunae_evaluated": False,
            "embible_reference": {
                "paper": "Findings of EACL 2024, 2024.findings-eacl.56",
                "repository": "https://github.com/harelm4/Embible",
                "repository_commit": "63dc79f1e4240b01883f5fe03e4e3389b8f2bc0d",
                "backend_commit_audited": "7c9e769",
                "implementation": "scaled adaptation of paper-described rule, not exact upstream code reproduction",
            },
            "modern_reconstructions_used": False,
            "split": "scroll-disjoint preserved_nonbib dev/heldout",
            "word_model": word_source.name,
            "word_model_sha256": (
                file_sha256(word_checkpoint)
                if word_checkpoint.is_file()
                else None
            ),
            "char_model": args.char_model,
            "char_model_revision": cached_huggingface_revision(args.char_model),
            "char_model_dss_finetuned": False,
            "primary_information": (
                "unknown character length, word count, and word boundaries"
            ),
            "oracle_conditions": (
                "gold word lengths or total character slots; diagnostics only"
            ),
            "max_words_searched": args.max_words,
            "max_characters_searched": args.max_chars,
            "maximum_generated_word_candidates": args.beam_width * args.max_words,
            "embible_word_candidate_target": 1000,
            "embible_character_candidate_limit": 5,
            "context_words_each_side": args.context_words,
            "dev_items": len(dev_items),
            "heldout_items": len(test_items),
            "dev_sample_seed": 71,
            "heldout_sample_seed": 73,
            "dev_sample_sha256": sample_sha256(dev_items),
            "heldout_sample_sha256": sample_sha256(test_items),
            "eligible_dev_by_words": dev_eligible,
            "eligible_heldout_by_words": test_eligible,
            "dev_fit": {
                "word_length_penalty": word_penalty,
                "char_length_penalty": char_penalty,
                "ensemble_word_weight": ensemble_weight,
            },
        },
        "results": {
            key: value
            for key, value in evaluation.items()
            if key != "cases"
        },
        "cases": evaluation["cases"],
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
