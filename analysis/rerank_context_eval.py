"""Test two cheap interventions on a shared single-word lacuna sample.

1. Larger context window (20 -> 40 words on each side).
2. Post-hoc reranking that downweights particle-like / function-word candidates.

The goal is to compare these interventions on exactly the same targets so the
differences are attributable to the intervention, not to a changed sample.
"""
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

tlog.set_verbosity_error()

MODELS = [("dicta-il/MsBERT", "MsBERT base")]
model_dir = repo_path("ft_msbert_span")
if model_dir.is_dir():
    MODELS.append((str(model_dir), "MsBERT+span-ft"))

MAX_ITEMS = 300
MIN_PRESERVED = 8
WINDOWS = [20, 40]
TOPN, BEAM, K = 50, 50, 20
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
FUNCTION = {
    "אשר", "כי", "כיא", "את", "אל", "על", "אם", "לא", "לוא", "כל", "כול",
    "מן", "הוא", "היא", "אני", "אתה", "הם", "זה", "זאת", "לו", "בו", "עד",
    "גם", "או", "כן", "אך", "רק", "יש", "אין", "מה", "מי", "ולא", "ואת", "וכל", "כה",
}
PROCLITIC_LETTERS = {"ו", "ל", "ב", "כ", "ה", "מ", "ש"}
rng = np.random.default_rng(0)
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


def heb(word: str) -> bool:
    return len(word) >= 2 and all(ch in HEB for ch in word)


def rerank_score(word: str, base_score: float) -> float:
    """Cheap heuristic: downweight particle-like, ultra-short predictions."""
    penalty = 0.0
    if len(word) == 1:
        penalty += 2.0
    elif len(word) == 2:
        penalty += 0.35
    if word in FUNCTION:
        penalty += 0.75
    if len(word) <= 2 and word[:1] in PROCLITIC_LETTERS:
        penalty += 0.5
    return base_score - penalty


def beam_candidates(logits, positions, tok):
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

    best_by_word = {}
    for score, seq in beams:
        word = tok.decode(seq).replace(" ", "").replace("##", "")
        if not word:
            continue
        prev = best_by_word.get(word)
        if prev is None or score > prev:
            best_by_word[word] = score

    ranked = sorted(best_by_word.items(), key=lambda item: -item[1])[:K]
    reranked = sorted(best_by_word.items(), key=lambda item: -rerank_score(item[0], item[1]))[:K]
    return ranked, reranked


TF_DIR = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
TF = Fabric(locations=str(TF_DIR), silent="deep")
api = TF.load("otype glyph rec biblical", silent="deep")
if api is False:
    raise RuntimeError(f"Could not load cached DSS corpus from {TF_DIR}")
F, L = api.F, api.L


def winfo(word_node):
    signs = L.d(word_node, "sign")
    glyph = "".join(F.glyph.v(sign) or "" for sign in signs)
    recs = [F.rec.v(sign) for sign in signs]
    return glyph, (bool(signs) and all(r == 1 for r in recs)), (bool(signs) and all(r != 1 for r in recs))


scrolls = {}
for word_node in F.otype.s("word"):
    if F.biblical.v(word_node):
        continue
    scroll = L.u(word_node, "scroll")
    if scroll:
        scrolls.setdefault(scroll[0], []).append(winfo(word_node))


target_ids = []
for scroll_id, words in scrolls.items():
    for idx, (glyph, fully_rec, _) in enumerate(words):
        if fully_rec and heb(glyph):
            target_ids.append((scroll_id, idx))

sample_ids = [target_ids[i] for i in rng.choice(len(target_ids), size=min(len(target_ids), MAX_ITEMS * 3), replace=False)]


def build_items(window: int):
    items = {}
    for scroll_id, idx in sample_ids:
        words = scrolls[scroll_id]
        lo, hi = max(0, idx - window), min(len(words), idx + window + 1)
        ctx, tpos, preserved = [], None, 0
        for pos in range(lo, hi):
            glyph, fully_rec, is_preserved = words[pos]
            if len(glyph) >= 1 and all(ch in HEB for ch in glyph):
                if pos == idx:
                    tpos = len(ctx)
                ctx.append(glyph)
                if is_preserved and pos != idx:
                    preserved += 1
        if tpos is not None and preserved >= MIN_PRESERVED:
            items[(scroll_id, idx)] = (ctx, tpos, ctx[tpos])
    return items


items_by_window = {window: build_items(window) for window in WINDOWS}
shared_ids = set.intersection(*(set(items.keys()) for items in items_by_window.values()))
shared_ids = sorted(shared_ids)
if len(shared_ids) > MAX_ITEMS:
    pick = rng.choice(len(shared_ids), size=MAX_ITEMS, replace=False)
    shared_ids = [shared_ids[i] for i in pick]

shared = {
    window: [items_by_window[window][item_id] for item_id in shared_ids]
    for window in WINDOWS
}

print(f"sampled targets: {len(sample_ids)}")
for window in WINDOWS:
    print(f"window={window:2d} valid targets: {len(items_by_window[window])}")
print(f"shared eval items: {len(shared_ids)}\n")

morph_dss.lemmas([gold for ctx, tpos, gold in shared[WINDOWS[0]]])


def summarize(results):
    n = len(results)
    return dict(
        exact_1=sum(1 for gold, ranked in results if ranked and ranked[0] == gold) / n * 100,
        exact_10=sum(1 for gold, ranked in results if gold in ranked[:10]) / n * 100,
        norm_1=sum(1 for gold, ranked in results if ranked and norm(ranked[0]) == norm(gold)) / n * 100,
        norm_10=sum(1 for gold, ranked in results if any(norm(word) == norm(gold) for word in ranked[:10])) / n * 100,
        norm_20=sum(1 for gold, ranked in results if any(norm(word) == norm(gold) for word in ranked[:20])) / n * 100,
    )


for repo, label in MODELS:
    print(f"=== {label} ===")
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).to(dev).eval()
    mask_id = tok.mask_token_id

    for window in WINDOWS:
        baseline = []
        reranked = []
        short_top1 = short_top1_reranked = 0
        for ctx, tpos, gold in shared[window]:
            enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
            word_map = {}
            for pos, word_id in enumerate(enc.word_ids(0)):
                if word_id is not None:
                    word_map.setdefault(word_id, []).append(pos)
            positions = word_map.get(tpos)
            if not positions:
                continue

            ids = enc["input_ids"][0].clone()
            for pos in positions:
                ids[pos] = mask_id
            with torch.no_grad():
                logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()
            ranked, reranked_scored = beam_candidates(logits, positions, tok)
            ranked_words = [word for word, _ in ranked]
            reranked_words = [word for word, _ in reranked_scored]
            baseline.append((gold, ranked_words))
            reranked.append((gold, reranked_words))
            if ranked_words and len(ranked_words[0]) < len(gold) - 1:
                short_top1 += 1
            if reranked_words and len(reranked_words[0]) < len(gold) - 1:
                short_top1_reranked += 1

        base_metrics = summarize(baseline)
        rerank_metrics = summarize(reranked)
        print(f"window={window:2d} baseline  | EXACT t1 {base_metrics['exact_1']:4.1f}% t10 {base_metrics['exact_10']:4.1f}% | NORM t1 {base_metrics['norm_1']:4.1f}% t10 {base_metrics['norm_10']:4.1f}% t20 {base_metrics['norm_20']:4.1f}%")
        print(f"window={window:2d} reranked  | EXACT t1 {rerank_metrics['exact_1']:4.1f}% t10 {rerank_metrics['exact_10']:4.1f}% | NORM t1 {rerank_metrics['norm_1']:4.1f}% t10 {rerank_metrics['norm_10']:4.1f}% t20 {rerank_metrics['norm_20']:4.1f}%")
        print(f"window={window:2d} short-top1 baseline={short_top1:3d} reranked={short_top1_reranked:3d}")
    print()
