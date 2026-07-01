"""Quick top-20 evaluation for both models."""
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

MODELS = [("dicta-il/MsBERT", "MsBERT base")]
model_dir = repo_path("ft_msbert_span")
if model_dir.is_dir():
    MODELS.append((str(model_dir), "MsBERT+span-ft"))

WINDOW, MIN_PRESERVED, MAX_ITEMS = 20, 8, 300
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
rng = np.random.default_rng(0)

def norm(w):
    lem = morph_dss.lemma(w)
    lem = "".join(FINAL.get(c, c) for c in lem)
    if lem in DIVINE: return "יהוה"
    if lem in {"כיא", "כי"}: return "כי"
    if lem in {"לוא", "לא"}: return "לא"
    if lem in {"כול", "כל"}: return "כל"
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
    if F.biblical.v(w): continue
    sc = L.u(w, "scroll")
    if sc: scrolls.setdefault(sc[0], []).append(winfo(w))

items = []
for ws in scrolls.values():
    for i, (g, fr, pr) in enumerate(ws):
        if not fr or len(g) < 2 or any(c not in HEB for c in g): continue
        lo, hi = max(0, i - WINDOW), min(len(ws), i + WINDOW + 1)
        ctx, tpos, preserved = [], None, 0
        for k in range(lo, hi):
            gg, ffr, ppr = ws[k]
            if len(gg) >= 1 and all(c in HEB for c in gg):
                if k == i: tpos = len(ctx)
                ctx.append(gg)
                if ppr and k != i: preserved += 1
        if tpos is not None and preserved >= MIN_PRESERVED:
            items.append((ctx, tpos, ctx[tpos]))
sel = rng.choice(len(items), size=min(MAX_ITEMS, len(items)), replace=False)
items = [items[i] for i in sel]

morph_dss.lemmas([g for _, _, g in items])

def beam_words(logits, ps, tok, topn=50, beam=50, k=20):
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
        if w not in out: out.append(w)
        if len(out) >= k: break
    return out

print(f"n={len(items)}")
for repo, nice in MODELS:
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).eval()
    results = []
    for ctx, tpos, gold in items:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        wmap = {}
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None: wmap.setdefault(wid, []).append(pos)
        ps = wmap.get(tpos)
        if not ps: results.append((gold, [])); continue
        ids = enc["input_ids"][0].clone()
        for p in ps: ids[p] = tok.mask_token_id
        with torch.no_grad(): logits = model(ids.unsqueeze(0)).logits[0]
        ranked = beam_words(logits, ps, tok)
        morph_dss.lemmas(ranked)
        results.append((gold, ranked))
    
    n = len(results)
    for topk in [1, 10, 20]:
        exact = sum(1 for g, r in results if g in r[:topk])
        normd = sum(1 for g, r in results if any(norm(w) == norm(g) for w in r[:topk]))
        print(f"  {nice:22s} Top-{topk:2d}: EXACT {exact/n*100:4.1f}% | NORM {normd/n*100:4.1f}%")
