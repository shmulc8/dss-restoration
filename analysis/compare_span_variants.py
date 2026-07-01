"""Compare span-ft variants on single-word lacunae.

Reports:
1. Exact/norm Top-1/Top-10 on a shared sample.
2. Content-word exact/norm metrics.
3. How often the top-1 prediction is suspiciously short.
"""
import sys
import os
from pathlib import Path

import numpy as np
import torch
from tf.fabric import Fabric
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import morph_dss
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

tlog.set_verbosity_error()

MODELS = [("dicta-il/MsBERT", "MsBERT base")]
for dirname, label in [
    ("ft_msbert_span", "MsBERT+span-ft"),
    ("ft_msbert_span_noparticles", "MsBERT+span-ft-no-particles"),
    ("ft_msbert_span_refined", "MsBERT+span-ft-refined"),
    ("ft_msbert_span_softshort", "MsBERT+span-ft-softshort"),
]:
    model_dir = repo_path(dirname)
    if model_dir.is_dir():
        MODELS.append((str(model_dir), label))

WINDOW, MIN_PRESERVED, MAX_ITEMS = 20, 8, 300
TOPN, BEAM, K = 50, 50, 20
SPLIT_MODE = os.environ.get("EVAL_SCROLL_SPLIT", "all")
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
FUNCTION = {
    "אשר", "כי", "כיא", "את", "אל", "על", "אם", "לא", "לוא", "כל", "כול",
    "מן", "הוא", "היא", "אני", "אתה", "הם", "זה", "זאת", "לו", "בו", "עד",
    "גם", "או", "כן", "אך", "רק", "יש", "אין", "מה", "מי", "ולא", "ואת", "וכל", "כה",
}
rng = np.random.default_rng(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
allowed_scrolls, split_label = resolve_scroll_filter(SPLIT_MODE)


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


def is_content(word: str) -> bool:
    return len(word) >= 3 and word not in FUNCTION


TF_DIR = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
TF = Fabric(locations=str(TF_DIR), silent="deep")
api = TF.load("otype glyph rec biblical scroll", silent="deep")
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

sel = rng.choice(len(items), size=min(MAX_ITEMS, len(items)), replace=False)
items = [items[i] for i in sel]
morph_dss.lemmas([gold for _, _, gold in items])
print(f"eval split: {split_label} | eligible scrolls: {len(scrolls)} | sampled items: {len(items)}")


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


for repo, label in MODELS:
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).to(dev).eval()
    results = []
    short_top1 = 0

    for ctx, tpos, gold in items:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        word_map = {}
        for pos, word_id in enumerate(enc.word_ids(0)):
            if word_id is not None:
                word_map.setdefault(word_id, []).append(pos)
        positions = word_map.get(tpos)
        if not positions:
            results.append((gold, []))
            continue
        ids = enc["input_ids"][0].clone()
        for pos in positions:
            ids[pos] = tok.mask_token_id
        with torch.no_grad():
            logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()
        ranked = beam_words(logits, positions, tok)
        if ranked and len(ranked[0]) < len(gold) - 1:
            short_top1 += 1
        results.append((gold, ranked))

    n = len(results)
    exact_1 = sum(1 for gold, ranked in results if ranked and ranked[0] == gold) / n * 100
    exact_10 = sum(1 for gold, ranked in results if gold in ranked[:10]) / n * 100
    norm_1 = sum(1 for gold, ranked in results if ranked and norm(ranked[0]) == norm(gold)) / n * 100
    norm_10 = sum(1 for gold, ranked in results if any(norm(word) == norm(gold) for word in ranked[:10])) / n * 100

    content = [(gold, ranked) for gold, ranked in results if is_content(gold)]
    nc = max(1, len(content))
    cont_exact_1 = sum(1 for gold, ranked in content if ranked and ranked[0] == gold) / nc * 100
    cont_norm_10 = sum(1 for gold, ranked in content if any(norm(word) == norm(gold) for word in ranked[:10])) / nc * 100

    print(f"{label:28s} | EXACT t1 {exact_1:4.1f}% t10 {exact_10:4.1f}% | NORM t1 {norm_1:4.1f}% t10 {norm_10:4.1f}% | content exact t1 {cont_exact_1:4.1f}% norm t10 {cont_norm_10:4.1f}% | short top1 {short_top1:3d}/{n}")
