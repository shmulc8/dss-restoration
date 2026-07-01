"""Comparison of different normalization methods on the 300-item test set.

Compares:
1. Exact matching (no normalization).
2. Medial string stripping (vav/yod middle stripping + divine name + final letter normalization).
3. DictaBERT-based lemmatization (dicta-il/dictabert-lex + divine name + spelling normalization of lo/loa, kol/kool, ki/kia without stripping vav/yod).
"""
import os
import sys
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

MODELS = [
    ("dicta-il/MsBERT", "MsBERT base"),
]
for d, nice in [("ft_msbert_span", "MsBERT+DSS-span-ft")]:
    model_dir = repo_path(d)
    if model_dir.is_dir():
        MODELS.append((str(model_dir), nice))

WINDOW, MIN_PRESERVED, MAX_ITEMS = 20, 8, 300
TOPN, BEAM, K = 20, 25, 10
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
rng = np.random.default_rng(0)

# String normalization
def norm_string(w):
    w = "".join(FINAL.get(c, c) for c in w)
    if w in DIVINE:
        return "יהוה"
    if w == "כיא":
        return "כי"
    if len(w) > 2:
        inner = w[1:-1].replace("ו", "").replace("י", "")
        return w[0] + inner + w[-1]
    return w

# Lemmatized normalization
def norm_lemma(w):
    # 1. Get lemma
    lem = morph_dss.lemma(w)
    # 2. Normalize finals
    lem = "".join(FINAL.get(c, c) for c in lem)
    # 3. Standardize spelling of common helper alephs / divine names
    if lem in DIVINE:
        return "יהוה"
    if lem in {"כיא", "כי"}:
        return "כי"
    if lem in {"לוא", "לא"}:
        return "לא"
    if lem in {"כול", "כל"}:
        return "כל"
    return lem

def is_content(w):
    return len(w) >= 3 and w not in {"אשר", "כי", "כיא", "את", "אל", "על", "אם", "לא", "לוא", "כל", "כול"}

A = use("etcbc/dss", silent="deep")
F, L = A.api.F, A.api.L

def winfo(w):
    signs = L.d(w, "sign")
    g = "".join(F.glyph.v(s) or "" for s in signs)
    recs = [F.rec.v(s) for s in signs]
    return g, (bool(signs) and all(r == 1 for r in recs)), (bool(signs) and all(r != 1 for r in recs))

# Find test items
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
        if fr and len(g) >= 2 and all(ch in HEB for ch in g):
            lo, hi = max(0, i - WINDOW), min(len(ws), i + WINDOW)
            ctx, preserved = [], 0
            for k in range(lo, hi):
                cg, cfr, cpr = ws[k]
                if k == i:
                    ctx.append(cg)
                elif len(cg) >= 2 and all(ch in HEB for ch in cg):
                    ctx.append(cg)
                    if cpr:
                        preserved += 1
            if preserved >= MIN_PRESERVED:
                items.append((ctx, ctx.index(g), g))

if len(items) > MAX_ITEMS:
    sel = rng.choice(len(items), size=MAX_ITEMS, replace=False)
    items = [items[i] for i in sel]

print(f"Loaded {len(items)} test items.")

# Pre-lemmatize all gold words in a batch to populate cache and speed up loop
print("Pre-lemmatizing gold words...", flush=True)
morph_dss.lemmas([g for _, _, g in items])

def beam_words(logits, ps, tok):
    beams = [(0.0, [])]
    for p in ps:
        lp = torch.log_softmax(logits[p], -1)
        top = torch.topk(lp, TOPN)
        beams = sorted([(s + v, seq + [i]) for s, seq in beams
                        for i, v in zip(top.indices.tolist(), top.values.tolist())],
                       key=lambda x: -x[0])[:BEAM]
    out = []
    for _, seq in beams:
        w = tok.decode(seq).replace(" ", "").replace("##", "")
        if w not in out:
            out.append(w)
        if len(out) >= K:
            break
    return out

for repo, nice in MODELS:
    print(f"\nEvaluating {nice}...", flush=True)
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).eval()
    MASK = tok.mask_token_id
    
    n_all = n_content = 0
    all_exact_1 = all_exact_10 = 0
    all_str_1 = all_str_10 = 0
    all_lem_1 = all_lem_10 = 0
    
    # loop and evaluate
    for i, (ctx, tpos, gold) in enumerate(items):
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        wmap = {}
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None:
                wmap.setdefault(wid, []).append(pos)
        ps = wmap.get(tpos)
        if not ps:
            continue
        ids = enc["input_ids"][0].clone()
        for p in ps:
            ids[p] = MASK
        with torch.no_grad():
            logits = model(ids.unsqueeze(0)).logits[0]
        ranked = beam_words(logits, ps, tok)
        
        # Pre-lemmatize predicted words to fill cache
        if ranked:
            morph_dss.lemmas(ranked)
            
        n_all += 1
        
        # 1. Exact
        e1 = bool(ranked) and gold == ranked[0]
        e10 = gold in ranked
        all_exact_1 += e1; all_exact_10 += e10
        
        # 2. String Normalization
        ng_str = norm_string(gold)
        s1 = bool(ranked) and norm_string(ranked[0]) == ng_str
        s10 = any(norm_string(r) == ng_str for r in ranked)
        all_str_1 += s1; all_str_10 += s10
        
        # 3. Lemmatized Normalization
        ng_lem = norm_lemma(gold)
        l1 = bool(ranked) and norm_lemma(ranked[0]) == ng_lem
        l10 = any(norm_lemma(r) == ng_lem for r in ranked)
        all_lem_1 += l1; all_lem_10 += l10
        
    print(f"Results for {nice} (n={n_all}):")
    print(f"  EXACT Matching          -> Top-1: {all_exact_1/n_all*100:4.1f}% | Top-10: {all_exact_10/n_all*100:4.1f}%")
    print(f"  STRING Normalization    -> Top-1: {all_str_1/n_all*100:4.1f}% | Top-10: {all_str_10/n_all*100:4.1f}%")
    print(f"  LEMMA Normalization     -> Top-1: {all_lem_1/n_all*100:4.1f}% | Top-10: {all_lem_10/n_all*100:4.1f}%")
