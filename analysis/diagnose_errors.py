"""Quick diagnostic: what are common near-miss patterns?

1. Model-ensemble: does combining MsBERT-base + MsBERT-span-ft via rank fusion help?
2. Wider beam: does TOPN=50, BEAM=50 recover more correct words?
3. Bigger context window: does WINDOW=40 help vs WINDOW=20?
4. Error categorization: what fraction of errors are (a) a synonym, (b) wrong inflection, (c) totally wrong?
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
from tf.app import use

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils import morph_dss
from utils.paths import repo_path

tlog.set_verbosity_error()

WINDOW, MIN_PRESERVED, MAX_ITEMS = 20, 8, 300
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
rng = np.random.default_rng(0)

def norm(w):
    lem = morph_dss.lemma(w)
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

A = use("etcbc/dss", silent="deep")
F, L = A.api.F, A.api.L

def winfo(w):
    signs = L.d(w, "sign")
    g = "".join(F.glyph.v(s) or "" for s in signs)
    recs = [F.rec.v(s) for s in signs]
    return g, (bool(signs) and all(r == 1 for r in recs)), (bool(signs) and all(r != 1 for r in recs))

scrolls = {}
for w in F.otype.s("word"):
    if F.biblical.v(w):
        continue
    sc = L.u(w, "scroll")
    if sc:
        scrolls.setdefault(sc[0], []).append(winfo(w))

items = []
for ws in scrolls.values():
    for i, (g, fr, pr) in enumerate(ws):
        if not fr or len(g) < 2 or any(c not in HEB for c in g):
            continue
        lo, hi = max(0, i - WINDOW), min(len(ws), i + WINDOW + 1)
        ctx, tpos, preserved = [], None, 0
        for k in range(lo, hi):
            gg, ffr, ppr = ws[k]
            if len(gg) >= 1 and all(c in HEB for c in gg):
                if k == i:
                    tpos = len(ctx)
                ctx.append(gg)
                if ppr and k != i:
                    preserved += 1
        if tpos is not None and preserved >= MIN_PRESERVED:
            items.append((ctx, tpos, ctx[tpos]))
sel = rng.choice(len(items), size=min(MAX_ITEMS, len(items)), replace=False)
items = [items[i] for i in sel]

print(f"Test items: {len(items)}")

# Pre-lemmatize gold
morph_dss.lemmas([g for _, _, g in items])

def beam_words(logits, ps, tok, topn=20, beam=25, k=10):
    beams = [(0.0, [])]
    for p in ps:
        lp = torch.log_softmax(logits[p], -1)
        top = torch.topk(lp, topn)
        beams = sorted([(s + v, seq + [i]) for s, seq in beams
                        for i, v in zip(top.indices.tolist(), top.values.tolist())],
                       key=lambda x: -x[0])[:beam]
    out = []
    for _, seq in beams:
        w = tok.decode(seq).replace(" ", "").replace("##", "")
        if w not in out:
            out.append(w)
        if len(out) >= k:
            break
    return out

def get_predictions(items, repo, topn=20, beam=25, k=10):
    """Returns list of (gold, ranked_list) for each item."""
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).eval()
    results = []
    for ctx, tpos, gold in items:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        wmap = {}
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None:
                wmap.setdefault(wid, []).append(pos)
        ps = wmap.get(tpos)
        if not ps:
            results.append((gold, []))
            continue
        ids = enc["input_ids"][0].clone()
        for p in ps:
            ids[p] = tok.mask_token_id
        with torch.no_grad():
            logits = model(ids.unsqueeze(0)).logits[0]
        ranked = beam_words(logits, ps, tok, topn, beam, k)
        results.append((gold, ranked))
    return results

def score(results, label=""):
    n = len(results)
    e1 = sum(1 for g, r in results if r and g == r[0])
    e10 = sum(1 for g, r in results if g in r)
    n1 = sum(1 for g, r in results if r and norm(r[0]) == norm(g))
    n10 = sum(1 for g, r in results if any(norm(w) == norm(g) for w in r))
    print(f"  {label:35s} EXACT Top-1: {e1/n*100:4.1f}% Top-10: {e10/n*100:4.1f}% | NORM Top-1: {n1/n*100:4.1f}% Top-10: {n10/n*100:4.1f}%")
    return n10

# === Test 1: Wider beam (TOPN=50, BEAM=50, K=20) ===
print("\n=== Test 1: Wider beam (TOPN=50, BEAM=50, K=20) vs default (20/25/10) ===")
res_default = get_predictions(items, "dicta-il/MsBERT", topn=20, beam=25, k=10)
score(res_default, "MsBERT base (default beam)")
res_wide = get_predictions(items, "dicta-il/MsBERT", topn=50, beam=50, k=20)
score(res_wide, "MsBERT base (wide beam, top-20)")
# Score the wide beam but only top-10
res_wide_10 = [(g, r[:10]) for g, r in res_wide]
score(res_wide_10, "MsBERT base (wide beam, top-10)")

# === Test 2: Model ensemble via rank fusion ===
print("\n=== Test 2: Rank fusion (MsBERT-base + MsBERT-span-ft) ===")
model_dir = repo_path("ft_msbert_span")
if model_dir.is_dir():
    res_ft = get_predictions(items, str(model_dir), topn=20, beam=25, k=10)
    score(res_ft, "MsBERT+span-ft alone")
    
    # Reciprocal Rank Fusion (RRF)
    K_RRF = 60  # standard RRF constant
    ensemble = []
    for (g1, r1), (g2, r2) in zip(res_default, res_ft):
        scores = {}
        for rank, w in enumerate(r1):
            scores[w] = scores.get(w, 0) + 1.0 / (K_RRF + rank)
        for rank, w in enumerate(r2):
            scores[w] = scores.get(w, 0) + 1.0 / (K_RRF + rank)
        merged = sorted(scores, key=lambda w: -scores[w])[:10]
        ensemble.append((g1, merged))
    score(ensemble, "RRF ensemble (base + span-ft)")

# === Test 3: Error categories on MsBERT base ===
print("\n=== Test 3: Error categories (MsBERT base, top-10 misses) ===")
total_miss = 0
cat_same_lemma = 0     # right lemma, wrong inflection
cat_same_length = 0    # same consonant count
cat_short_pred = 0     # predicted word is shorter (particle leak)
cat_total_miss = 0     # no overlap at all

for gold, ranked in res_default:
    ng = norm(gold)
    if any(norm(w) == ng for w in ranked):
        continue  # it's a hit under normalization
    total_miss += 1
    if not ranked:
        cat_total_miss += 1
        continue
    top1 = ranked[0]
    if len(top1) < len(gold) - 1:
        cat_short_pred += 1
    elif len(top1) == len(gold):
        cat_same_length += 1
    else:
        cat_total_miss += 1

print(f"  Total norm-misses: {total_miss}")
print(f"  Short prediction (particle leak): {cat_short_pred} ({cat_short_pred/max(total_miss,1)*100:.0f}%)")
print(f"  Same length (semantic miss):      {cat_same_length} ({cat_same_length/max(total_miss,1)*100:.0f}%)")
print(f"  Other:                            {cat_total_miss} ({cat_total_miss/max(total_miss,1)*100:.0f}%)")

# Show some near-misses: cases where top-1 prediction has the same lemma as a different gold word
print("\n=== Sample errors (first 15 norm-misses, MsBERT base) ===")
shown = 0
for gold, ranked in res_default:
    ng = norm(gold)
    if any(norm(w) == ng for w in ranked):
        continue
    if shown >= 15:
        break
    top5 = ranked[:5] if ranked else ["<empty>"]
    print(f"  gold: {gold:12s} (norm: {ng:10s}) | top-5: {', '.join(top5)}")
    shown += 1
