"""Export non-oracle uncertainty cases from the TF layer.

These are not benchmark items. They are researcher-facing loci where the corpus
signals strong uncertainty or loss, but no gold restoration is available.
"""
import csv
import json
import os
import sys
from pathlib import Path

import torch
from tf.fabric import Fabric
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.book_filters import resolve_book_exclusions
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path
from utils.clitic_join import join_likely_clitics

CSV_OUT = os.environ.get(
    "CSV_OUT",
    str(ROOT / "analysis" / "reports" / "full_unknown_gap_cases_hebrew_only.csv"),
)
WINDOW = int(os.environ.get("WINDOW", "8"))
SPLIT_MODE = os.environ.get("EVAL_SCROLL_SPLIT", "heldout")
BOOK_FILTER_MODE = os.environ.get("BOOK_FILTER_MODE", "no-aram")
MIN_UNC = int(os.environ.get("MIN_UNC", "3"))

MODEL_WINDOW = 40
TOPN = 50
BEAM = 100
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL_MAP = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}


def is_hebrew_word(word: str) -> bool:
    return len(word) >= 2 and all(ch in HEB for ch in word)


def get_pattern(raw_text: str) -> str:
    cleaned = ""
    for ch in raw_text:
        if ch in HEB or ch == "#":
            cleaned += ch
    return cleaned.replace("#", "⬚")


def matches_pattern(candidate: str, pattern: str) -> bool:
    if not pattern:
        return True
    if len(candidate) != len(pattern):
        return False
    for c_char, p_char in zip(candidate, pattern):
        if p_char == "⬚":
            continue
        c_norm = FINAL_MAP.get(c_char, c_char)
        p_norm = FINAL_MAP.get(p_char, p_char)
        if c_norm != p_norm:
            return False
    return True


def beam_words(logits, positions, tok, max_candidates=100):
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
        if len(out) >= max_candidates:
            break
    return out


def beam_autoregressive(model, input_ids, gap_token_positions, tok, dev, patterns=None, beam_width=5):
    beams = [(0.0, input_ids.clone().to(dev), [])]
    for slot_idx, ps in enumerate(gap_token_positions):
        pattern = patterns[slot_idx] if patterns else None
        new_beams = []
        for score, current_ids, pred_words in beams:
            with torch.no_grad():
                logits = model(current_ids.unsqueeze(0).to(dev)).logits[0].cpu()
                
            slot_beams = [(0.0, [])]
            for p in ps:
                lp = torch.log_softmax(logits[p], -1)
                search_topk = 500 if pattern else TOPN
                top = torch.topk(lp, search_topk)
                
                expanded = sorted(
                    [(s + v, seq + [i]) for s, seq in slot_beams
                     for i, v in zip(top.indices.tolist(), top.values.tolist())],
                    key=lambda x: -x[0]
                )
                
                filtered = []
                for s, seq in expanded:
                    word = tok.decode(seq).replace(" ", "").replace("##", "")
                    if pattern:
                        if matches_pattern(word, pattern):
                            filtered.append((s, seq))
                    else:
                        filtered.append((s, seq))
                
                slot_beams = filtered[:beam_width]
                
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


def marker_flags(entry):
    flags = []
    if entry["vac"] is not None:
        flags.append("vacat")
    if entry["rem"] is not None:
        flags.append("removed")
    if entry["unc"] is not None and entry["unc"] >= MIN_UNC:
        flags.append(f"uncertain-{entry['unc']}")
    if not entry["glyph"]:
        flags.append("empty-glyph")
    if "#" in entry["glyph"] or "#" in entry["full"]:
        flags.append("hash-loss")
    return flags


def display_token(entry, in_run):
    token = entry["full"] if in_run else (entry["glyph"] or entry["full"])
    token = token.strip() if token else ""
    return token or "∅"


def classify_run(run_entries):
    flags = set()
    for entry in run_entries:
        flags.update(marker_flags(entry))
    if "empty-glyph" in flags:
        return "Unknown gap"
    if "hash-loss" in flags:
        return "Damaged / missing letters"
    if "vacat" in flags:
        return "Vacat or unwritten space"
    if "removed" in flags:
        return "Removed text"
    return "High uncertainty"


def context_for_display(words, start, end):
    lo = max(0, start - WINDOW)
    hi = min(len(words), end + WINDOW)
    pieces = []
    for idx in range(lo, hi):
        if idx == start:
            pieces.append("⟦")
        pieces.append(display_token(words[idx], start <= idx < end))
        if idx == end - 1:
            pieces.append("⟧")
    return " ".join(piece for piece in pieces if piece)


def context_for_reading(words, start, end):
    tokens = []
    for pos in range(max(0, start - WINDOW), min(len(words), end + WINDOW)):
        if pos == start:
            tokens.append("__GAP__")
        elif start < pos < end:
            continue
        else:
            glyph = words[pos]["glyph"]
            cleaned = "".join(ch for ch in glyph if ch in HEB)
            if cleaned:
                tokens.append(cleaned)
            else:
                tokens.append(glyph)

    joined, _ = join_likely_clitics(" ".join(tokens))

    gap_parts = []
    for pos in range(start, end):
        gw = words[pos]["full"]
        cleaned_gw = "".join(ch for ch in gw if ch in HEB or ch == "#")
        cleaned_gw = cleaned_gw.replace("#", "⬚")
        if cleaned_gw:
            gap_parts.append(cleaned_gw)
        else:
            gap_parts.append(words[pos]["glyph"] or gw)

    gap_text = " ".join(gap_parts)
    bracketed_gap = f"[{gap_text}]"

    return joined.replace("__GAP__", bracketed_gap)


def raw_run_text(run_entries):
    return " ".join(display_token(entry, True) for entry in run_entries)


def collect_items(model, tok, dev):
    allowed_scrolls, split_label = resolve_scroll_filter(SPLIT_MODE)
    excluded_books, book_filter_label = resolve_book_exclusions(BOOK_FILTER_MODE)
    tf_dir = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
    tf = Fabric(locations=str(tf_dir), silent="deep")
    api = tf.load("otype glyph full unc vac rem scroll biblical", silent="deep")
    if api is False:
        raise RuntimeError(f"Could not load cached DSS corpus from {tf_dir}")
    F, L = api.F, api.L

    items = []
    for scroll_node in F.otype.s("scroll"):
        scroll = F.scroll.v(scroll_node)
        if allowed_scrolls is not None and scroll not in allowed_scrolls:
            continue
        if scroll in excluded_books:
            continue
        words = []
        for word_node in L.d(scroll_node, "word"):
            if F.biblical.v(word_node):
                continue
            signs = L.d(word_node, "sign")
            glyph = "".join(F.glyph.v(sign) or "" for sign in signs)
            words.append({
                "node": word_node,
                "glyph": glyph,
                "full": F.full.v(word_node) or glyph,
                "unc": F.unc.v(word_node),
                "vac": F.vac.v(word_node),
                "rem": F.rem.v(word_node),
            })

        idx = 0
        while idx < len(words):
            flags = marker_flags(words[idx])
            if not flags:
                idx += 1
                continue
            end = idx + 1
            while end < len(words) and marker_flags(words[end]):
                end += 1
            run = words[idx:end]

            # Reconstruct context for model
            lo = max(0, idx - MODEL_WINDOW)
            hi = min(len(words), end + MODEL_WINDOW)
            context_words = []
            gap_positions = []
            for pos in range(lo, hi):
                if idx <= pos < end:
                    gap_positions.append(len(context_words))
                    context_words.append("ה")
                else:
                    glyph = words[pos]["glyph"]
                    if is_hebrew_word(glyph):
                        context_words.append(glyph)

            enc = tok(context_words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
            word_map = {}
            for pos_in_tokens, word_id in enumerate(enc.word_ids(0)):
                if word_id is not None:
                    word_map.setdefault(word_id, []).append(pos_in_tokens)

            gap_token_positions = [word_map.get(gap_pos) for gap_pos in gap_positions]

            top1_words = []
            top1_constrained_words = []
            slot_details = []
            if not any(positions is None for positions in gap_token_positions):
                ids = enc["input_ids"][0].clone()
                for positions in gap_token_positions:
                    for pos_in_tokens in positions:
                        ids[pos_in_tokens] = tok.mask_token_id

                with torch.inference_mode():
                    logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()

                # Get raw autoregressive predictions
                ranked_raw = beam_autoregressive(model, ids, gap_token_positions, tok, dev, patterns=None, beam_width=5)
                
                # Get patterns for all slots
                patterns = []
                for slot_index in range(1, len(gap_token_positions) + 1):
                    slot_word = words[idx + slot_index - 1]
                    patterns.append(get_pattern(slot_word["full"]))
                
                # Get constrained autoregressive predictions
                ranked_constrained = beam_autoregressive(model, ids, gap_token_positions, tok, dev, patterns=patterns, beam_width=5)
                
                for slot_index, positions in enumerate(gap_token_positions, start=1):
                    j = slot_index - 1
                    
                    raw_preds = []
                    for phrase in ranked_raw:
                        if len(phrase) > j:
                            w = phrase[j]
                            if w not in raw_preds:
                                raw_preds.append(w)
                    raw_preds = raw_preds[:5]
                    
                    constrained_preds = []
                    for phrase in ranked_constrained:
                        if len(phrase) > j:
                            w = phrase[j]
                            if w not in constrained_preds:
                                constrained_preds.append(w)
                    constrained_preds = constrained_preds[:5]

                    slot_details.append({
                        "slot_index": slot_index,
                        "raw_candidates": raw_preds,
                        "constrained_candidates": constrained_preds,
                        "pattern": patterns[j],
                    })

                    if raw_preds:
                        top1_words.append(raw_preds[0])
                    if constrained_preds:
                        top1_constrained_words.append(constrained_preds[0])
                    elif raw_preds:
                        top1_constrained_words.append(raw_preds[0])

            top1_phrase = " ".join(top1_words)
            top1_constrained_phrase = " ".join(top1_constrained_words)

            items.append({
                "scroll": scroll,
                "start_index": idx,
                "end_index": end,
                "run_length": end - idx,
                "context_for_display": context_for_display(words, idx, end),
                "context_for_reading": context_for_reading(words, idx, end),
                "raw_run_text": raw_run_text(run),
                "flags": sorted({flag for entry in run for flag in marker_flags(entry)}),
                "category": classify_run(run),
                "split": split_label,
                "book_filter": book_filter_label,
                "top1_phrase": top1_phrase,
                "top1_constrained_phrase": top1_constrained_phrase,
                "slot_details": slot_details,
            })
            idx = end

            if len(items) % 500 == 0:
                print(f"Collected and predicted {len(items)} unknown items...", flush=True)
    return items


def main():
    MODEL_NAME = os.environ.get("MODEL_NAME", "ft_msbert_span_refined")
    MODEL_REPO = str(repo_path(MODEL_NAME))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading model and tokenizer from {MODEL_REPO} on {dev}...")
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_REPO).to(dev).eval()

    items = collect_items(model, tok, dev)
    out_path = Path(CSV_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id", "scroll", "start_index", "end_index", "run_length",
        "context_for_display", "context_for_reading", "raw_run_text", "flags", "category",
        "split", "book_filter", "top1_phrase", "top1_constrained_phrase", "slot_details_json",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row_id, item in enumerate(items, start=1):
            writer.writerow({
                "row_id": row_id,
                "scroll": item["scroll"],
                "start_index": item["start_index"],
                "end_index": item["end_index"],
                "run_length": item["run_length"],
                "context_for_display": item["context_for_display"],
                "context_for_reading": item["context_for_reading"],
                "raw_run_text": item["raw_run_text"],
                "flags": " | ".join(item["flags"]),
                "category": item["category"],
                "split": item["split"],
                "book_filter": item["book_filter"],
                "top1_phrase": item["top1_phrase"],
                "top1_constrained_phrase": item["top1_constrained_phrase"],
                "slot_details_json": json.dumps(item["slot_details"], ensure_ascii=False),
            })
    print(f"wrote {out_path}")
    print(f"rows={len(items)}")


if __name__ == "__main__":
    main()
