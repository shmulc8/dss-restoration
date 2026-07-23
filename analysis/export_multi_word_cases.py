"""Export held-out multi-word lacuna cases with per-slot model predictions.

This mirrors the span benchmark regime:
- detect real reconstructed multi-word runs
- mask the whole run at once
- score each gap word while the rest of the run stays masked

The output drives the local demo site so researchers can inspect real span
failures rather than only single-word cases.
"""
import csv
import json
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

from utils import morph_dss
from utils.book_filters import resolve_book_exclusions
from utils.clitic_join import join_likely_clitics
from utils.dss_split import load_partition
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

tlog.set_verbosity_error()

MODEL_NAME = os.environ.get("MODEL_NAME", "ft_msbert_span_refined")
MODEL_REPO = str(repo_path(MODEL_NAME))
CSV_OUT = os.environ.get(
    "CSV_OUT",
    str(ROOT / "analysis" / "reports" / "full_multi_word_cases_refined_hebrew_only.csv"),
)
WINDOW = int(os.environ.get("WINDOW", "40"))
MIN_PRESERVED = int(os.environ.get("MIN_PRESERVED", "6"))
TOPN = int(os.environ.get("TOPN", "50"))
BEAM = int(os.environ.get("BEAM", "50"))
K = int(os.environ.get("K", "5"))
MAX_GAP_WORDS = int(os.environ.get("MAX_GAP_WORDS", "12"))
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))  # <=0 means all
SPLIT_MODE = os.environ.get("EVAL_SCROLL_SPLIT", "heldout")
BOOK_FILTER_MODE = os.environ.get("BOOK_FILTER_MODE", "no-aram")

HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def norm(word: str) -> str:
    lemma = morph_dss.lemma(word)
    lemma = "".join(FINAL.get(ch, ch) for ch in lemma)
    if lemma in DIVINE:
        return "יהוה"
    if lemma in {"כיא", "כי"}:
        return "כי"
    if lemma in {"לוא", "לא"}:
        return "לא"
    if lemma in {"כול", "כל"}:
        return "כל"
    return lemma


def is_hebrew_word(word: str) -> bool:
    return len(word) >= 2 and all(ch in HEB for ch in word)


def gap_bucket(size: int) -> str:
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size == 3:
        return "3"
    if size <= 5:
        return "4-5"
    return "6+"


def beam_words(logits, positions, tok):
    beams = [(0.0, [])]
    for pos in positions:
        logp = torch.log_softmax(logits[pos], -1)
        top = torch.topk(logp, TOPN)
        new_beams = []
        for score, seq in beams:
            for tok_id, delta in zip(top.indices.tolist(), top.values.tolist()):
                if seq and seq[-1] == tok_id:
                    continue
                new_beams.append((score + delta, seq + [tok_id]))
        beams = sorted(new_beams, key=lambda item: -item[0])[:BEAM]
    out = []
    for _, seq in beams:
        word = tok.decode(seq).replace(" ", "").replace("##", "")
        if word not in out:
            out.append(word)
        if len(out) >= K:
            break
    return out


def beam_autoregressive(model, input_ids, gap_token_positions, tok, dev, beam_width=5):
    beams = [(0.0, input_ids.clone().to(dev), [])]
    for slot_idx, ps in enumerate(gap_token_positions):
        new_beams = []
        for score, current_ids, pred_words in beams:
            with torch.no_grad():
                logits = model(current_ids.unsqueeze(0).to(dev)).logits[0].cpu()
                
            slot_beams = [(0.0, [])]
            for p in ps:
                lp = torch.log_softmax(logits[p], -1)
                top = torch.topk(lp, TOPN)
                slot_beams = sorted(
                    [(s + v, seq + [i]) for s, seq in slot_beams
                     for i, v in zip(top.indices.tolist(), top.values.tolist())],
                    key=lambda x: -x[0]
                )[:beam_width]
                
            for slot_score, seq in slot_beams:
                word = tok.decode(seq).replace(" ", "").replace("##", "")
                new_ids = current_ids.clone()
                for i, p in enumerate(ps):
                    new_ids[p] = seq[i]
                new_beams.append((score + slot_score, new_ids, pred_words + [word]))
                
        beams = sorted(new_beams, key=lambda x: -x[0])[:beam_width]
        
    out = []
    for _, _, pred_words in beams:
        if pred_words not in out:
            out.append(pred_words)
    return out


def fit_frequencies():
    fit_rows = load_partition("fit")
    surf = Counter()
    for row in fit_rows:
        for word in row["text"].split():
            surf[word] += 1
    return surf


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
        fully_reconstructed = bool(signs) and all(r == 1 for r in recs)
        preserved = bool(signs) and all(r != 1 for r in recs)
        return glyph, fully_reconstructed, preserved

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
        scrolls.setdefault(scroll_name, []).append(winfo(word_node))

    items = []
    for scroll_name, words in scrolls.items():
        idx = 0
        while idx < len(words):
            if not words[idx][1]:
                idx += 1
                continue
            end = idx
            while end < len(words) and words[end][1]:
                end += 1
            gap_targets = [pos for pos in range(idx, end) if is_hebrew_word(words[pos][0])]
            gap_size = len(gap_targets)
            if 2 <= gap_size <= MAX_GAP_WORDS:
                lo = max(0, idx - WINDOW)
                hi = min(len(words), end + WINDOW + 1)
                context_words = []
                gap_positions = []
                gold_words = []
                preserved = 0
                for pos in range(lo, hi):
                    glyph, _, is_preserved = words[pos]
                    if idx <= pos < end:
                        if pos in gap_targets:
                            gap_positions.append(len(context_words))
                            gold_words.append(glyph)
                            context_words.append(glyph)
                    elif is_hebrew_word(glyph):
                        context_words.append(glyph)
                        if is_preserved:
                            preserved += 1
                if preserved >= MIN_PRESERVED and gap_positions:
                    items.append({
                        "scroll": scroll_name,
                        "context_words": context_words,
                        "gap_positions": gap_positions,
                        "gold_words": gold_words,
                        "gap_length": gap_size,
                    })
            idx = end
    if MAX_ITEMS > 0:
        items = items[:MAX_ITEMS]
    return items, split_label, book_filter_label


def joined_context(words, gap_positions):
    gap_set = set(gap_positions)
    placeholder = "__LACUNA__"
    patched = [placeholder if idx in gap_set else word for idx, word in enumerate(words)]
    text, _ = join_likely_clitics(" ".join(patched))
    return text.replace(placeholder, "⬚⬚⬚")


def raw_context(words, gap_positions):
    gap_set = set(gap_positions)
    return " ".join("⬚⬚⬚" if idx in gap_set else word for idx, word in enumerate(words))


def classify_slot(gold, ranked, gold_freq):
    top1 = ranked[0] if ranked else ""
    if not ranked:
        return "empty", "No candidate decoded."
    if norm(gold) == norm(top1):
        return "exact_top1_hit", "The model's first guess is exactly correct for this slot."
    if any(norm(candidate) == norm(gold) for candidate in ranked):
        return "exact_top5_hit", "The gold word appears in the model's top-5 list for this slot."
    if top1 and len(top1) <= max(1, len(gold) - 2):
        return "particle_or_too_short", "Top guess is much shorter than the gold word."
    if gold_freq <= 1:
        return "rare_or_unseen", "Gold word is very rare or unseen in the fit data."
    if any(len(candidate) == len(gold) for candidate in ranked):
        return "same_length_semantic", "Model prefers another plausible word of similar length."
    return "other_miss", "Miss without a simple short/rare/same-length explanation."


def classify_case(slot_rows):
    slot_count = len(slot_rows)
    top1_hits = sum(1 for slot in slot_rows if slot["hit_top1"])
    top5_hits = sum(1 for slot in slot_rows if slot["hit_top5"])
    if top1_hits == slot_count:
        return "hit", "All gap words are exact top-1 hits."
    if top5_hits == slot_count:
        return "hit", "All gap words appear in the top-5 lists."
    if top5_hits > 0:
        return "miss", f"{top5_hits}/{slot_count} gap words appear in the top-5 lists."
    return "miss", "None of the gap words appear in the top-5 lists."


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
        "row_id", "model", "split", "book_filter", "scroll",
        "gap_length", "gap_bucket", "target_phrase", "target_words",
        "target_fit_frequencies", "top1_phrase", "top5_phrases",
        "context_for_reading", "raw_context", "case_status", "reader_note",
        "slot_top1_hits", "slot_top5_hits", "slot_details_json",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row_id, item in enumerate(items, start=1):
            enc = tok(item["context_words"], is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
            word_map = {}
            for pos, word_id in enumerate(enc.word_ids(0)):
                if word_id is not None:
                    word_map.setdefault(word_id, []).append(pos)
            gap_token_positions = [word_map.get(gap_pos) for gap_pos in item["gap_positions"]]
            if any(positions is None for positions in gap_token_positions):
                continue

            ids = enc["input_ids"][0].clone()
            for positions in gap_token_positions:
                for pos in positions:
                    ids[pos] = tok.mask_token_id

            # Get joint autoregressive predictions
            ranked_phrases = beam_autoregressive(model, ids, gap_token_positions, tok, dev, beam_width=5)

            slot_rows = []
            for slot_index, (positions, gold) in enumerate(zip(gap_token_positions, item["gold_words"]), start=1):
                j = slot_index - 1
                
                # Decompose autoregressive phrases into candidate words for this slot
                ranked = []
                for phrase in ranked_phrases:
                    if len(phrase) > j:
                        w = phrase[j]
                        if w not in ranked:
                            ranked.append(w)
                ranked = ranked[:K]

                category, note = classify_slot(gold, ranked, surf[gold])
                slot_rows.append({
                    "slot_index": slot_index,
                    "gold_word": gold,
                    "gold_fit_frequency": surf[gold],
                    "top_candidates": ranked,
                    "top1": ranked[0] if ranked else "",
                    "hit_top1": bool(ranked) and norm(ranked[0]) == norm(gold),
                    "hit_top5": any(norm(candidate) == norm(gold) for candidate in ranked),
                    "likely_issue": category,
                    "reader_note": note,
                })

            case_status, reader_note = classify_case(slot_rows)
            top1_phrase = " ".join(ranked_phrases[0]) if ranked_phrases else ""
            top5_phrases = [" ".join(phrase) for phrase in ranked_phrases[:5]]

            writer.writerow({
                "row_id": row_id,
                "model": MODEL_NAME,
                "split": split_label,
                "book_filter": book_filter_label,
                "scroll": item["scroll"],
                "gap_length": item["gap_length"],
                "gap_bucket": gap_bucket(item["gap_length"]),
                "target_phrase": " ".join(item["gold_words"]),
                "target_words": json.dumps(item["gold_words"], ensure_ascii=False),
                "target_fit_frequencies": json.dumps([slot["gold_fit_frequency"] for slot in slot_rows], ensure_ascii=False),
                "top1_phrase": top1_phrase,
                "top5_phrases": json.dumps(top5_phrases[:K], ensure_ascii=False),
                "context_for_reading": joined_context(item["context_words"], item["gap_positions"]),
                "raw_context": raw_context(item["context_words"], item["gap_positions"]),
                "case_status": case_status,
                "reader_note": reader_note,
                "slot_top1_hits": sum(1 for slot in slot_rows if slot["hit_top1"]),
                "slot_top5_hits": sum(1 for slot in slot_rows if slot["hit_top5"]),
                "slot_details_json": json.dumps(slot_rows, ensure_ascii=False),
            })
            if row_id % 100 == 0:
                print(f"processed {row_id}/{len(items)}", flush=True)

    print(f"wrote {out_path}")
    print(f"rows={len(items)}")


if __name__ == "__main__":
    main()
