"""Held-out single- and multi-word lacuna evaluation for preserved MLM + RAG.

The editorial reconstructions in Text-Fabric are used only as anonymous
evaluation labels. They are never inserted into model input, fine-tuning data,
or the retrieval index. The retrieval index contains preserved words from the
non-biblical training split only, and its scalar weight is chosen on dev.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tf.fabric import Fabric
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.score_qd_researcher_benchmark import (  # noqa: E402
    DEVICE,
    GAP_TOKEN,
    RAG_ALPHAS,
    contiguous_context,
    hebrew_letters,
    rag_context_keys,
    rag_normalize,
    rag_score,
)
from utils.preserved_corpus import load_chunks, split_scrolls  # noqa: E402

tlog.set_verbosity_error()

DEFAULT_MODEL = ROOT / "ft_msbert_span_preserved_nonbib"
DEFAULT_REPORT = ROOT / "analysis" / "reports" / "preserved_rag_lacuna_lengths.json"
DEFAULT_MARKDOWN = ROOT / "analysis" / "reports" / "PRESERVED_RAG_LACUNA_LENGTHS.md"
DEFAULT_TF_DIR = Path(
    os.environ.get(
        "DSS_TF_DIR",
        "/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0",
    )
)
BUCKETS = ("1", "2", "3", "4-5", "6+")
WINDOW = 40
MIN_PRESERVED = 6
DEV_ITEMS = 250
CANDIDATE_TOPN = 500
BEAM_WIDTH = 20
BEAM_EXPANSION = 20


def bucket(length: int) -> str:
    if length == 1:
        return "1"
    if length == 2:
        return "2"
    if length == 3:
        return "3"
    if length <= 5:
        return "4-5"
    return "6+"


def build_token_rag_index() -> tuple[dict[Any, Counter[str]], dict[str, Any]]:
    """Build an exact-context index without joining Text-Fabric clitic tokens."""
    index: dict[Any, Counter[str]] = {}
    indexed_targets = 0
    rows = load_chunks("train")
    for row in rows:
        words = row["text"].split()
        for target_index, target in enumerate(words):
            normalized = rag_normalize(target)
            if not normalized:
                continue
            left, right = contiguous_context(words, target_index)
            for key in rag_context_keys(left, right):
                index.setdefault(key, Counter())[normalized] += 1
            indexed_targets += 1
    return index, {
        "source_split": "preserved_nonbib train",
        "train_chunks": len(rows),
        "indexed_targets": indexed_targets,
        "context_keys": len(index),
        "tokenization": "Text-Fabric words; clitics remain separate",
    }


def candidate_rows(
    logits: torch.Tensor,
    normalized_token_by_id: list[str],
    index: dict[Any, Counter[str]],
    left: tuple[str, ...],
    right: tuple[str, ...],
    alpha: float,
    *,
    limit: int = CANDIDATE_TOPN,
) -> list[tuple[str, int, float]]:
    top = torch.topk(logits, min(limit, logits.shape[-1]))
    rows = []
    seen = set()
    for token_id, model_score in zip(top.indices.tolist(), top.values.tolist()):
        candidate = normalized_token_by_id[token_id]
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        retrieval_score, _, _ = rag_score(index, left, right, candidate)
        rows.append(
            (candidate, token_id, float(model_score) + alpha * retrieval_score)
        )
    rows.sort(key=lambda row: (-row[2], row[0]))
    return rows


def build_dev_items(limit: int = DEV_ITEMS) -> list[dict[str, Any]]:
    items = []
    for row in load_chunks("dev"):
        words = row["text"].split()
        if sum(word != GAP_TOKEN for word in words) < 10:
            continue
        for target_index, gold in enumerate(words):
            if not rag_normalize(gold):
                continue
            left, right = contiguous_context(words, target_index)
            if not rag_context_keys(left, right):
                continue
            lo = max(0, target_index - WINDOW)
            hi = min(len(words), target_index + WINDOW + 1)
            context = words[lo:hi]
            local_target = target_index - lo
            context[local_target] = GAP_TOKEN
            items.append(
                {
                    "context": context,
                    "target": local_target,
                    "gold": rag_normalize(gold),
                    "left": left,
                    "right": right,
                }
            )
    random.Random(42).shuffle(items)
    return items[:limit]


def fit_alpha(
    model: Any,
    tokenizer: Any,
    normalized_token_by_id: list[str],
    index: dict[Any, Counter[str]],
    batch_size: int,
) -> dict[str, Any]:
    records = []
    items = build_dev_items()
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        words = [
            [
                tokenizer.mask_token if word == GAP_TOKEN else word
                for word in item["context"]
            ]
            for item in batch
        ]
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"].clone()
        positions = []
        valid_items = []
        for row_index, item in enumerate(batch):
            ps = [
                position
                for position, word_id in enumerate(
                    encoding.word_ids(batch_index=row_index)
                )
                if word_id == item["target"]
            ]
            if len(ps) != 1:
                continue
            input_ids[row_index, ps[0]] = tokenizer.mask_token_id
            positions.append((row_index, ps[0]))
            valid_items.append(item)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE),
            ).logits.cpu()
        for (row_index, position), item in zip(positions, valid_items):
            raw = candidate_rows(
                logits[row_index, position],
                normalized_token_by_id,
                index,
                item["left"],
                item["right"],
                0.0,
            )
            records.append(
                {
                    "gold": item["gold"],
                    "candidates": [
                        (
                            candidate,
                            score,
                            rag_score(
                                index,
                                item["left"],
                                item["right"],
                                candidate,
                            )[0],
                        )
                        for candidate, _, score in raw
                    ],
                }
            )

    grid = {}
    for alpha in RAG_ALPHAS:
        ranks = []
        for record in records:
            ranked = sorted(
                record["candidates"],
                key=lambda row: (-(row[1] + alpha * row[2]), row[0]),
            )
            rank = next(
                (
                    position
                    for position, row in enumerate(ranked)
                    if rag_normalize(row[0]) == record["gold"]
                ),
                999,
            )
            ranks.append(rank)
        grid[str(alpha)] = {
            "n": len(ranks),
            "top1": 100 * sum(rank == 0 for rank in ranks) / len(ranks),
            "top10": 100 * sum(rank < 10 for rank in ranks) / len(ranks),
            "top20": 100 * sum(rank < 20 for rank in ranks) / len(ranks),
        }
    selected = max(
        RAG_ALPHAS,
        key=lambda alpha: (
            grid[str(alpha)]["top10"],
            grid[str(alpha)]["top1"],
            -alpha,
        ),
    )
    return {
        "fit_split": "preserved_nonbib dev",
        "sample_seed": 42,
        "n": len(records),
        "alpha": selected,
        "baseline": grid["0.0"],
        "selected": grid[str(selected)],
        "grid": grid,
    }


def load_lacunae(tf_dir: Path) -> list[dict[str, Any]]:
    tf = Fabric(locations=str(tf_dir), silent="deep")
    api = tf.load("otype glyph rec biblical scroll", silent="deep")
    if api is False:
        raise RuntimeError(f"Could not load cached DSS corpus from {tf_dir}")
    F, L = api.F, api.L
    heldout = split_scrolls("heldout")
    scroll_words: dict[str, list[tuple[str, bool, bool]]] = defaultdict(list)
    for word in F.otype.s("word"):
        if bool(F.biblical.v(word)):
            continue
        scroll_nodes = L.u(word, "scroll")
        if not scroll_nodes:
            continue
        scroll = F.scroll.v(scroll_nodes[0])
        if scroll not in heldout:
            continue
        signs = L.d(word, "sign")
        glyph = "".join(F.glyph.v(sign) or "" for sign in signs)
        reconstructed = bool(signs) and all(F.rec.v(sign) == 1 for sign in signs)
        preserved = bool(signs) and all(F.rec.v(sign) != 1 for sign in signs)
        scroll_words[scroll].append((glyph, reconstructed, preserved))

    items = []
    for scroll, words in scroll_words.items():
        start = 0
        while start < len(words):
            if not words[start][1]:
                start += 1
                continue
            end = start
            while end < len(words) and words[end][1]:
                end += 1
            target_indices = [
                index
                for index in range(start, end)
                if hebrew_letters(words[index][0])
            ]
            if 1 <= len(target_indices) <= 12:
                lo = max(0, start - WINDOW)
                hi = min(len(words), end + WINDOW)
                context = []
                target_positions = []
                golds = []
                preserved_count = 0
                for index in range(lo, hi):
                    glyph, reconstructed, preserved = words[index]
                    if index in target_indices:
                        target_positions.append(len(context))
                        golds.append(hebrew_letters(glyph))
                        context.append(GAP_TOKEN)
                    elif preserved and hebrew_letters(glyph):
                        context.append(hebrew_letters(glyph))
                        preserved_count += 1
                    else:
                        context.append(GAP_TOKEN)
                if preserved_count >= MIN_PRESERVED:
                    items.append(
                        {
                            "scroll": scroll,
                            "context": context,
                            "target_positions": target_positions,
                            "golds": golds,
                            "length": len(golds),
                        }
                    )
            start = end
    return items


def encode_item(item: dict[str, Any], tokenizer: Any):
    words = [
        tokenizer.mask_token if word == GAP_TOKEN else word
        for word in item["context"]
    ]
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = encoding["input_ids"][0].clone()
    token_positions = []
    for target in item["target_positions"]:
        ps = [
            position
            for position, word_id in enumerate(encoding.word_ids(0))
            if word_id == target
        ]
        if len(ps) != 1:
            return None
        input_ids[ps[0]] = tokenizer.mask_token_id
        token_positions.append(ps[0])
    return input_ids, encoding["attention_mask"][0], token_positions


def rank_gold(rows: list[tuple[str, int, float]], gold: str) -> int:
    normalized_gold = rag_normalize(gold)
    return next(
        (
            position
            for position, row in enumerate(rows)
            if rag_normalize(row[0]) == normalized_gold
        ),
        999,
    )


def sequence_beam(
    *,
    model: Any,
    normalized_token_by_id: list[str],
    index: dict[Any, Counter[str]],
    item: dict[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_positions: list[int],
    alpha: float,
) -> list[list[str]]:
    beams = [(0.0, input_ids.clone(), list(item["context"]), [])]
    for slot_index, (word_position, token_position) in enumerate(
        zip(item["target_positions"], token_positions)
    ):
        ids_batch = torch.stack([beam[1] for beam in beams]).to(DEVICE)
        masks = attention_mask.unsqueeze(0).repeat(len(beams), 1).to(DEVICE)
        with torch.inference_mode():
            logits = model(
                input_ids=ids_batch,
                attention_mask=masks,
            ).logits[:, token_position].cpu()
        expanded = []
        for beam_index, (score, ids, context, predictions) in enumerate(beams):
            left, right = contiguous_context(context, word_position)
            candidates = candidate_rows(
                logits[beam_index],
                normalized_token_by_id,
                index,
                left,
                right,
                alpha,
                limit=CANDIDATE_TOPN,
            )[:BEAM_EXPANSION]
            for candidate, token_id, candidate_score in candidates:
                new_ids = ids.clone()
                new_ids[token_position] = token_id
                new_context = list(context)
                new_context[word_position] = candidate
                expanded.append(
                    (
                        score + candidate_score,
                        new_ids,
                        new_context,
                        predictions + [candidate],
                    )
                )
        beams = sorted(expanded, key=lambda beam: -beam[0])[:BEAM_WIDTH]
    return [beam[3] for beam in beams]


def empty_counts() -> dict[str, int]:
    return {"n": 0, "top1": 0, "top5": 0, "top10": 0, "top20": 0}


def add_rank(counts: dict[str, int], rank: int) -> None:
    counts["n"] += 1
    for cutoff in (1, 5, 10, 20):
        counts[f"top{cutoff}"] += rank < cutoff


def percentages(counts: dict[str, int]) -> dict[str, float | int]:
    total = counts["n"]
    return {
        "n": total,
        **{
            key: 100 * counts[key] / total if total else 0.0
            for key in ("top1", "top5", "top10", "top20")
        },
    }


def evaluate(
    *,
    items: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    normalized_token_by_id: list[str],
    index: dict[Any, Counter[str]],
    alpha: float,
) -> dict[str, Any]:
    cells = {
        method: {
            level: {name: empty_counts() for name in BUCKETS}
            for level in ("slot", "sequence")
        }
        for method in ("baseline", "rag")
    }
    for number, item in enumerate(items, 1):
        encoded = encode_item(item, tokenizer)
        if encoded is None:
            continue
        input_ids, attention_mask, token_positions = encoded
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.unsqueeze(0).to(DEVICE),
                attention_mask=attention_mask.unsqueeze(0).to(DEVICE),
            ).logits[0].cpu()
        name = bucket(item["length"])
        for method, weight in (("baseline", 0.0), ("rag", alpha)):
            for word_position, token_position, gold in zip(
                item["target_positions"], token_positions, item["golds"]
            ):
                left, right = contiguous_context(item["context"], word_position)
                rows = candidate_rows(
                    logits[token_position],
                    normalized_token_by_id,
                    index,
                    left,
                    right,
                    weight,
                )
                add_rank(cells[method]["slot"][name], rank_gold(rows, gold))
            sequences = sequence_beam(
                model=model,
                normalized_token_by_id=normalized_token_by_id,
                index=index,
                item=item,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_positions=token_positions,
                alpha=weight,
            )
            gold_sequence = [rag_normalize(gold) for gold in item["golds"]]
            sequence_rank = next(
                (
                    rank
                    for rank, sequence in enumerate(sequences)
                    if [rag_normalize(word) for word in sequence] == gold_sequence
                ),
                999,
            )
            add_rank(cells[method]["sequence"][name], sequence_rank)
        if number % 10 == 0:
            print(f"evaluated spans: {number}/{len(items)}", flush=True)
    return {
        method: {
            level: {
                name: percentages(counts)
                for name, counts in by_bucket.items()
            }
            for level, by_bucket in levels.items()
        }
        for method, levels in cells.items()
    }


def aggregate_buckets(
    bucket_results: dict[str, dict[str, float | int]],
    names: tuple[str, ...],
) -> dict[str, float | int]:
    total = sum(int(bucket_results[name]["n"]) for name in names)
    result: dict[str, float | int] = {"n": total}
    for metric in ("top1", "top5", "top10", "top20"):
        successes = sum(
            float(bucket_results[name][metric])
            * int(bucket_results[name]["n"])
            / 100
            for name in names
        )
        result[metric] = 100 * successes / total if total else 0.0
    return result


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for level in ("slot", "sequence"):
        for method in ("baseline", "rag"):
            single = report["summary"][method][level]["single_word"]
            multi = report["summary"][method][level]["multiword"]
            rows.append(
                f"| {level.title()} | {method.upper()} | "
                f"{single['n']} | {single['top1']:.1f}% | "
                f"{single['top10']:.1f}% | {multi['n']} | "
                f"{multi['top1']:.1f}% | {multi['top10']:.1f}% |"
            )
    return f"""# Preserved-only RAG by lacuna length

## Held-out results

| Evaluation | Method | Single N | Top-1 | Top-10 | Multiword N | Top-1 | Top-10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Single-word and multi-word scores are reported separately. Slot accuracy asks
whether each editorial word appears in Top-K. Sequence accuracy requires the
entire lacuna to match, in order, in one of the Top-K beams.

The Text-Fabric reconstructions are anonymous editorial evaluation labels, not
physical ground truth. They were excluded from model input, training, and
retrieval. RAG uses only preserved non-biblical training scrolls; alpha
`{report['protocol']['rag_fit']['alpha']}` was selected only on dev scrolls.

No gold character lengths are used. The decoder knows only the number of word
slots in a lacuna.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tf-dir", type=Path, default=DEFAULT_TF_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--per-bucket", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.tf_dir.is_dir():
        raise FileNotFoundError(f"Text-Fabric directory not found: {args.tf_dir}")
    if not 1 <= args.per_bucket <= 500:
        raise ValueError("--per-bucket must be between 1 and 500")
    if not 1 <= args.batch_size <= 128:
        raise ValueError("--batch-size must be between 1 and 128")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(args.model)).to(DEVICE).eval()
    normalized_token_by_id = [
        hebrew_letters(tokenizer.decode([token_id]).strip())
        for token_id in range(len(tokenizer))
    ]
    index, index_metadata = build_token_rag_index()
    rag_fit = fit_alpha(
        model,
        tokenizer,
        normalized_token_by_id,
        index,
        args.batch_size,
    )
    all_items = load_lacunae(args.tf_dir)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_items:
        grouped[bucket(item["length"])].append(item)
    generator = random.Random(42)
    sampled = []
    sampled_counts = {}
    for name in BUCKETS:
        candidates = grouped[name]
        generator.shuffle(candidates)
        chosen = candidates[: args.per_bucket]
        sampled.extend(chosen)
        sampled_counts[name] = len(chosen)
    results = evaluate(
        items=sampled,
        model=model,
        tokenizer=tokenizer,
        normalized_token_by_id=normalized_token_by_id,
        index=index,
        alpha=float(rag_fit["alpha"]),
    )
    summary = {}
    for method, levels in results.items():
        summary[method] = {}
        for level, by_bucket in levels.items():
            summary[method][level] = {
                "single_word": aggregate_buckets(by_bucket, ("1",)),
                "multiword": aggregate_buckets(
                    by_bucket, ("2", "3", "4-5", "6+")
                ),
            }
    report = {
        "protocol": {
            "model": str(args.model),
            "device": DEVICE,
            "corpus": "held-out non-biblical Text-Fabric DSS scrolls",
            "gold": "anonymous editorial reconstruction; evaluation only",
            "known_lacuna_information": "word-slot count only; no gold lengths",
            "sample_seed": 42,
            "sample_per_length_bucket": args.per_bucket,
            "eligible_spans_found": {
                name: len(grouped[name]) for name in BUCKETS
            },
            "sampled_spans": sampled_counts,
            "retrieval_index": index_metadata,
            "rag_fit": rag_fit,
            "heldout_used_for_tuning": False,
        },
        "by_length_bucket": results,
        "summary": summary,
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
