"""Inspect failed single-word lacuna restorations for a chosen model.

Outputs:
1. Coarse failure-category counts on a sampled evaluation split.
2. Concrete miss examples with context, top-k predictions, and fit-set frequency hints.
3. Optional CSV export for spreadsheet review.
"""
import os
import sys
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tf.fabric import Fabric
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import morph_dss
from utils.dss_split import load_partition
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

tlog.set_verbosity_error()

MODEL_NAME = os.environ.get("MODEL_NAME", "ft_msbert_span_refined")
MODEL_REPO = str(repo_path(MODEL_NAME))
SPLIT_MODE = os.environ.get("EVAL_SCROLL_SPLIT", "heldout")
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "300"))
EXAMPLES = int(os.environ.get("EXAMPLES", "12"))
WINDOW = int(os.environ.get("WINDOW", "20"))
MIN_PRESERVED = int(os.environ.get("MIN_PRESERVED", "8"))
TOPN = int(os.environ.get("TOPN", "50"))
BEAM = int(os.environ.get("BEAM", "50"))
K = int(os.environ.get("K", "10"))
SEED = int(os.environ.get("SEED", "0"))
CSV_OUT = os.environ.get("CSV_OUT", "").strip()

HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
PREFIXES = set("ובכלמשה")
rng = np.random.default_rng(SEED)
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def norm(word: str) -> str:
    lem = morph_dss.lemma(word)
    lem = "".join(FINAL.get(c, c) for c in lem)
    if lem in DIVINE:
        return "יהוה"
    if lem in {"כיא", "כי"}:
        return "כי"
    if lem in {"לוא", "לא"}:
        return "לא"
    if lem in {"כול", "כל"}:
        return "כל"
    return lem


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def beam_words(logits, positions, tok):
    beams = [(0.0, [])]
    for pos in positions:
        logp = torch.log_softmax(logits[pos], -1)
        top = torch.topk(logp, TOPN)
        beams = sorted(
            [
                (score + delta, seq + [tok_id])
                for score, seq in beams
                for tok_id, delta in zip(top.indices.tolist(), top.values.tolist())
            ],
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


def fit_frequencies():
    fit_rows = load_partition("fit")
    surf = Counter()
    lem = Counter()
    for row in fit_rows:
        for word in row["text"].split():
            surf[word] += 1
            lem[norm(word)] += 1
    return surf, lem


def load_items():
    allowed_scrolls, split_label = resolve_scroll_filter(SPLIT_MODE)
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
        scrolls.setdefault(scroll[0], []).append(winfo(word_node))

    items = []
    for words in scrolls.values():
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
                items.append((ctx, tpos, ctx[tpos]))
    if len(items) > MAX_ITEMS:
        sel = rng.choice(len(items), size=MAX_ITEMS, replace=False)
        items = [items[i] for i in sel]
    return items, split_label, len(scrolls)


def classify(gold: str, ranked: list[str], surface_freq: Counter, lemma_freq: Counter) -> tuple[str, str]:
    gold_norm = norm(gold)
    top1 = ranked[0] if ranked else ""
    gold_freq = surface_freq[gold]
    gold_lemma_freq = lemma_freq[gold_norm]

    if not ranked:
        return "empty", "No usable candidate was decoded for this gap."

    if len(top1) <= max(1, len(gold) - 2):
        if gold and gold[0] in PREFIXES:
            return "particle_or_split", "Top prediction is much shorter than the gold word, likely reflecting split clitics in the training data."
        return "too_short", "Top prediction is much shorter than the gold word, suggesting residual short-word bias."

    if gold_freq == 0 and gold_lemma_freq == 0:
        return "unseen_lexeme", "Gold form and lemma are absent from the fit partition, so this looks like a true lexical generalization failure."

    if gold_freq == 0 and gold_lemma_freq > 0:
        return "seen_lemma_unseen_form", "The lemma exists in fit data but this surface form does not, suggesting morphological or orthographic sparsity."

    near_same_len = [cand for cand in ranked if len(cand) == len(gold)]
    if near_same_len:
        best = min(near_same_len, key=lambda cand: edit_distance(cand, gold))
        if edit_distance(best, gold) <= 2:
            return "orthographic_or_morph", "A same-length near-neighbor is ranked, so the miss is probably a fine-grained orthographic or inflectional distinction."

    return "semantic_or_ambiguous", "Candidates have plausible length but not the target lemma, so context may be too weak or the target is semantically underdetermined."


def collect_failures():
    if not Path(MODEL_REPO).is_dir():
        raise RuntimeError(f"Model checkpoint not found: {MODEL_REPO}")

    items, split_label, n_scrolls = load_items()
    surface_freq, lemma_freq = fit_frequencies()
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_REPO).to(dev).eval()

    morph_dss.lemmas([gold for _, _, gold in items])

    failures = []
    category_counts = Counter()
    for ctx, tpos, gold in items:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        word_map = {}
        for pos, word_id in enumerate(enc.word_ids(0)):
            if word_id is not None:
                word_map.setdefault(word_id, []).append(pos)
        positions = word_map.get(tpos)
        if not positions:
            ranked = []
        else:
            ids = enc["input_ids"][0].clone()
            for pos in positions:
                ids[pos] = tok.mask_token_id
            with torch.no_grad():
                logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()
            ranked = beam_words(logits, positions, tok)

        if any(norm(word) == norm(gold) for word in ranked[:10]):
            continue

        category, why = classify(gold, ranked[:10], surface_freq, lemma_freq)
        category_counts[category] += 1
        lo, hi = max(0, tpos - 5), min(len(ctx), tpos + 6)
        snippet = " ".join(ctx[idx] if idx != tpos else "⬚⬚⬚" for idx in range(lo, hi))
        failures.append({
            "split_label": split_label,
            "eligible_scrolls": n_scrolls,
            "gold": gold,
            "gold_norm": norm(gold),
            "category": category,
            "why": why,
            "gold_freq": surface_freq[gold],
            "gold_lemma_freq": lemma_freq[norm(gold)],
            "context": snippet,
            "top10": ranked[:10],
        })

    failures.sort(key=lambda item: (item["category"], item["gold_freq"], item["gold"]))
    return failures, category_counts, split_label, n_scrolls, len(items)


def write_csv(path: str, failures: list[dict]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "split",
        "book_filter_mode",
        "sampled_items",
        "category",
        "gold",
        "gold_norm",
        "gold_fit_freq",
        "gold_lemma_fit_freq",
        "context",
        "top1",
        "top2",
        "top3",
        "top4",
        "top5",
        "top6",
        "top7",
        "top8",
        "top9",
        "top10",
        "top10_joined",
        "why",
    ]
    with target.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in failures:
            row = {
                "model": MODEL_NAME,
                "split": item["split_label"],
                "book_filter_mode": BOOK_FILTER_MODE,
                "sampled_items": MAX_ITEMS,
                "category": item["category"],
                "gold": item["gold"],
                "gold_norm": item["gold_norm"],
                "gold_fit_freq": item["gold_freq"],
                "gold_lemma_fit_freq": item["gold_lemma_freq"],
                "context": item["context"],
                "top10_joined": " | ".join(item["top10"]),
                "why": item["why"],
            }
            for idx in range(10):
                row[f"top{idx+1}"] = item["top10"][idx] if idx < len(item["top10"]) else ""
            writer.writerow(row)


def main():
    failures, category_counts, split_label, n_scrolls, sampled_items = collect_failures()
    if CSV_OUT:
        write_csv(CSV_OUT, failures)

    print(
        f"model={MODEL_NAME} | device={dev} | eval split={split_label} "
        f"| eligible scrolls={n_scrolls} | sampled items={sampled_items}"
    )
    print(f"norm top-10 misses: {len(failures)}")
    for category, count in category_counts.most_common():
        print(f"  {category:22s} {count:3d}  {count / max(1, len(failures)) * 100:4.1f}%")

    print("\n=== Example misses ===")
    for item in failures[:EXAMPLES]:
        topk = ", ".join(item["top10"]) if item["top10"] else "<empty>"
        print(f"\n[{item['category']}]")
        print(f"ctx : …{item['context']}…")
        print(
            f"gold: {item['gold']} | fit freq={item['gold_freq']} "
            f"| fit lemma freq={item['gold_lemma_freq']}"
        )
        print(f"top10: {topk}")
        print(f"why : {item['why']}")


if __name__ == "__main__":
    main()
