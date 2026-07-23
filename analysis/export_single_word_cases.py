"""Export all eligible single-word cases with real model predictions.

This is meant to drive the local demo site with full, non-sampled data.
"""
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from tf.fabric import Fabric
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.book_filters import resolve_book_exclusions
from utils.clitic_join import join_likely_clitics
from utils.dss_split import load_partition
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

tlog.set_verbosity_error()

MODEL_NAME = os.environ.get("MODEL_NAME", "ft_msbert_span_refined")
MODEL_REPO = str(repo_path(MODEL_NAME))
SPLIT_MODE = os.environ.get("EVAL_SCROLL_SPLIT", "heldout")
BOOK_FILTER_MODE = os.environ.get("BOOK_FILTER_MODE", "no-aram")
CSV_OUT = os.environ.get("CSV_OUT", str(ROOT / "analysis" / "reports" / "full_single_word_cases.csv"))
WINDOW = int(os.environ.get("WINDOW", "20"))
MIN_PRESERVED = int(os.environ.get("MIN_PRESERVED", "8"))
TOPN = int(os.environ.get("TOPN", "20"))
BEAM = int(os.environ.get("BEAM", "20"))
K = int(os.environ.get("K", "5"))
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))  # <=0 means all

HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
PREFIXES = set("ובכלמשה")
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def beam_words(logits, positions, tok):
    beams = [(0.0, [])]
    for pos in positions:
        logp = torch.log_softmax(logits[pos], -1)
        top = torch.topk(logp, TOPN)
        beams = sorted(
            [(score + delta, seq + [tok_id]) for score, seq in beams for tok_id, delta in zip(top.indices.tolist(), top.values.tolist())],
            key=lambda item: -item[0],
        )[:BEAM]
    out = []
    for _, seq in beams:
        word = tok.decode(seq).replace(" ", "").replace("##", "")
        if word not in out:
            out.append(word)
        if len(out) >= K:
            break
    return out


def collect_items():
    allowed_scrolls, split_label = resolve_scroll_filter(SPLIT_MODE)
    excluded_books, book_filter_label = resolve_book_exclusions(BOOK_FILTER_MODE)
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
        fully_rec = bool(signs) and all(r == 1 for r in recs)
        preserved = bool(signs) and all(r != 1 for r in recs)
        return glyph, fully_rec, preserved

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
        scrolls.setdefault(scroll[0], []).append(winfo(word_node))

    items = []
    for scroll_node, words in scrolls.items():
        scroll_name = F.scroll.v(scroll_node)
        for idx, (glyph, fully_rec, _) in enumerate(words):
            if not fully_rec or len(glyph) < 2 or any(ch not in HEB for ch in glyph):
                continue
            lo, hi = max(0, idx - WINDOW), min(len(words), idx + WINDOW + 1)
            ctx, tpos, preserved = [], None, 0
            for pos in range(lo, hi):
                cur, _, is_preserved = words[pos]
                if len(cur) >= 1 and all(ch in HEB for ch in cur):
                    if pos == idx:
                        tpos = len(ctx)
                    ctx.append(cur)
                    if is_preserved and pos != idx:
                        preserved += 1
            if tpos is not None and preserved >= MIN_PRESERVED:
                items.append({
                    "scroll": scroll_name,
                    "context_words": ctx,
                    "target_pos": tpos,
                    "gold": ctx[tpos],
                })
    if MAX_ITEMS > 0:
        items = items[:MAX_ITEMS]
    return items, split_label, book_filter_label


def fit_frequencies():
    fit_rows = load_partition("fit")
    surf = Counter()
    for row in fit_rows:
        for word in row["text"].split():
            surf[word] += 1
    return surf


def joined_context(words, target_pos):
    placeholder = "__LACUNA__"
    patched = [placeholder if idx == target_pos else word for idx, word in enumerate(words)]
    text, _ = join_likely_clitics(" ".join(patched))
    return text.replace(placeholder, "⬚⬚⬚")


def classify(gold, ranked, gold_freq):
    top1 = ranked[0] if ranked else ""
    if not ranked:
        return "empty", "No candidate decoded."
    if gold == top1:
        return "exact_top1_hit", "The model's first guess is exactly correct."
    if gold in ranked:
        return "exact_top5_hit", "The gold word appears in the model's top-5 list."
    if top1 and len(top1) <= max(1, len(gold) - 2):
        return "particle_or_too_short", "Top guess is much shorter than the gold word."
    if gold_freq <= 1:
        return "rare_or_unseen", "Gold word is very rare or unseen in the fit partition."
    if any(len(candidate) == len(gold) for candidate in ranked):
        return "same_length_semantic", "Model prefers another plausible word of similar length."
    return "other_miss", "Miss without a simple short/rare/same-length explanation."


def main():
    if not Path(MODEL_REPO).is_dir():
        raise RuntimeError(f"Model checkpoint not found: {MODEL_REPO}")
    items, split_label, book_filter_label = collect_items()
    surf = fit_frequencies()
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_REPO).to(dev).eval()

    out_path = Path(CSV_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id", "model", "split", "book_filter", "scroll", "target_word", "target_fit_frequency",
        "target_position", "context_for_reading", "raw_context", "model_top1", "model_top2", "model_top3",
        "model_top4", "model_top5", "all_top5", "case_status", "likely_issue", "reader_note",
    ]

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, item in enumerate(items, start=1):
            enc = tok(item["context_words"], is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
            word_map = {}
            for pos, word_id in enumerate(enc.word_ids(0)):
                if word_id is not None:
                    word_map.setdefault(word_id, []).append(pos)
            positions = word_map.get(item["target_pos"])
            if not positions:
                continue
            ids = enc["input_ids"][0].clone()
            for pos in positions:
                ids[pos] = tok.mask_token_id
            with torch.inference_mode():
                logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()
            ranked = beam_words(logits, positions, tok)
            status, note = classify(item["gold"], ranked, surf[item["gold"]])
            row = {
                "row_id": idx,
                "model": MODEL_NAME,
                "split": split_label,
                "book_filter": book_filter_label,
                "scroll": item["scroll"],
                "target_word": item["gold"],
                "target_fit_frequency": surf[item["gold"]],
                "target_position": item["target_pos"],
                "context_for_reading": joined_context(item["context_words"], item["target_pos"]),
                "raw_context": " ".join("⬚⬚⬚" if i == item["target_pos"] else w for i, w in enumerate(item["context_words"])),
                "model_top1": ranked[0] if len(ranked) > 0 else "",
                "model_top2": ranked[1] if len(ranked) > 1 else "",
                "model_top3": ranked[2] if len(ranked) > 2 else "",
                "model_top4": ranked[3] if len(ranked) > 3 else "",
                "model_top5": ranked[4] if len(ranked) > 4 else "",
                "all_top5": " | ".join(ranked),
                "case_status": "hit" if item["gold"] in ranked else "miss",
                "likely_issue": status,
                "reader_note": note,
            }
            writer.writerow(row)
            if idx % 250 == 0:
                print(f"processed {idx}/{len(items)}", flush=True)

    print(f"wrote {out_path}")
    print(f"rows={len(items)}")


if __name__ == "__main__":
    main()
