"""Benchmark preserved-only fine-tuning on intact held-out DSS words.

Targets are physically preserved words from non-biblical held-out scrolls.
Editorial reconstructions and explicit unknown material appear only as
unlabelled ``[MASK]`` context slots, never as targets or gold answers.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tf.fabric import Fabric
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import morph_dss
from utils.paths import repo_path
from utils.preserved_corpus import GAP_TOKEN, classify_word, split_scrolls

tlog.set_verbosity_error()

TF_DIR = Path(
    os.environ.get(
        "DSS_TF_DIR",
        "/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0",
    )
)
MODELS = [
    ("MsBERT base", "dicta-il/MsBERT"),
    ("Legacy span-refined", repo_path("ft_msbert_span_refined")),
    (
        "Preserved-only span",
        repo_path("ft_msbert_span_preserved_nonbib"),
    ),
]
WINDOW = 40
SAMPLE_SIZE = 300
TOPN = 20
MIN_CONTEXT_WORDS = 10
SEED = 42
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
device = "mps" if torch.backends.mps.is_available() else "cpu"


def norm(word):
    if not word:
        return ""
    lemma = morph_dss.lemma(word)
    lemma = "".join(FINAL.get(character, character) for character in lemma)
    if lemma in DIVINE:
        return "יהוה"
    if lemma in {"כיא", "כי"}:
        return "כי"
    if lemma in {"לוא", "לא"}:
        return "לא"
    if lemma in {"כול", "כל"}:
        return "כל"
    return lemma


def build_items():
    heldout = split_scrolls("heldout")
    tf = Fabric(locations=str(TF_DIR), silent="deep")
    api = tf.load(
        "otype glyph full rec rem biblical scroll",
        silent="deep",
    )
    if api is False:
        raise RuntimeError(f"Could not load DSS Text-Fabric corpus from {TF_DIR}")
    F, L = api.F, api.L

    items = []
    for scroll_node in F.otype.s("scroll"):
        scroll = F.scroll.v(scroll_node)
        if scroll not in heldout:
            continue
        events = [
            classify_word(F, L, word_node)
            for word_node in L.d(scroll_node, "word")
            if not F.biblical.v(word_node)
        ]
        significant = [
            event for event in events if event["kind"] in {"word", "gap"}
        ]
        for target_index, target in enumerate(significant):
            if target["kind"] != "word" or len(target["token"]) < 2:
                continue
            start = max(0, target_index - WINDOW)
            end = min(len(significant), target_index + WINDOW + 1)
            window = significant[start:end]
            preserved_context = sum(
                event["kind"] == "word" for event in window
            ) - 1
            if preserved_context < MIN_CONTEXT_WORDS:
                continue
            words = [
                event["token"] if event["kind"] == "word" else GAP_TOKEN
                for event in window
            ]
            items.append(
                {
                    "scroll": scroll,
                    "words": words,
                    "target_index": target_index - start,
                    "gold": target["token"],
                    "context_gap_slots": words.count(GAP_TOKEN),
                }
            )
    np.random.default_rng(SEED).shuffle(items)
    return items[:SAMPLE_SIZE]


def summarize(ranks):
    total = len(ranks)
    return {
        "top1": sum(rank == 0 for rank in ranks) / total * 100,
        "top5": sum(rank < 5 for rank in ranks) / total * 100,
        "top10": sum(rank < 10 for rank in ranks) / total * 100,
        "top20": sum(rank < 20 for rank in ranks) / total * 100,
        "total": total,
    }


def evaluate(model_source, items):
    tokenizer = AutoTokenizer.from_pretrained(str(model_source), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_source))
    model = model.to(device).eval()
    records = []
    batch_size = 16

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        model_words = [
            [
                tokenizer.mask_token if word == GAP_TOKEN else word
                for word in item["words"]
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
        target_positions = []
        for row_index, item in enumerate(batch):
            positions = [
                position
                for position, word_id in enumerate(
                    encoding.word_ids(batch_index=row_index)
                )
                if word_id == item["target_index"]
            ]
            target_positions.append(positions)
            for position in positions:
                input_ids[row_index, position] = tokenizer.mask_token_id

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=encoding["attention_mask"].to(device),
            ).logits.cpu()
        for row_index, item in enumerate(batch):
            positions = target_positions[row_index]
            if not positions:
                continue
            log_probs = torch.log_softmax(
                logits[row_index, positions[0]],
                dim=-1,
            )
            top = torch.topk(log_probs, TOPN)
            predictions = [
                tokenizer.decode([token_id]).strip()
                for token_id in top.indices.tolist()
            ]
            records.append((item["gold"], predictions))

    morph_dss.lemmas(
        word
        for gold, predictions in records
        for word in [gold, *predictions]
        if word
    )
    ranks = []
    for gold, predictions in records:
        gold_norm = norm(gold)
        ranks.append(
            next(
                (
                    rank
                    for rank, prediction in enumerate(predictions)
                    if norm(prediction) == gold_norm
                ),
                999,
            )
        )
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return summarize(ranks)


def main():
    items = build_items()
    if len(items) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Only {len(items)} eligible held-out items; expected {SAMPLE_SIZE}"
        )
    results = {}
    for name, source in MODELS:
        if isinstance(source, Path) and not source.is_dir():
            print(f"SKIP: {name}; checkpoint not found at {source}")
            continue
        print(f"Evaluating {name}...")
        results[name] = evaluate(source, items)

    print("\n=== PRESERVED NON-BIBLICAL INTACT-WORD BENCHMARK ===")
    print(f"{'Model':28s} {'Top-1':>8s} {'Top-5':>8s} {'Top-10':>8s} {'Top-20':>8s}")
    for name, metrics in results.items():
        print(
            f"{name:28s} {metrics['top1']:7.1f}% "
            f"{metrics['top5']:7.1f}% {metrics['top10']:7.1f}% "
            f"{metrics['top20']:7.1f}%"
        )

    report = {
        "protocol": {
            "target": "intact preserved words only",
            "context_reconstructions": "redacted as unlabelled [MASK] slots",
            "scroll_split": "preserved-nonbib heldout",
            "sample_size": len(items),
            "sample_seed": SEED,
            "mean_context_gap_slots": float(
                np.mean([item["context_gap_slots"] for item in items])
            ),
            "researcher_comparison": False,
        },
        "results": results,
    }
    report_path = (
        ROOT
        / "analysis"
        / "reports"
        / "preserved_nonbib_intact_benchmark.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"saved -> {report_path}")


if __name__ == "__main__":
    main()
