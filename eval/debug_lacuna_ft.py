"""Content word vs All word breakdown for base and fine-tuned models.

Scholars care most about content words (nouns, verbs, adjectives). This script
compares all models on the subset of content words versus all words.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
from tf.app import use
tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils import morph_dss
from utils.paths import repo_path

MODELS = [
    ("dicta-il/MsBERT", "MsBERT base"),
    ("dicta-il/BEREL", "BEREL base"),
]
for d, nice in [("ft_msbert_span", "MsBERT+DSS-span-ft")]:
    model_dir = repo_path(d)
    if model_dir.is_dir():
        MODELS.append((str(model_dir), nice))

WINDOW, MIN_PRESERVED, MAX_ITEMS = 20, 8, 300
TOPN, BEAM, K = 20, 25, 10
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
rng = np.random.default_rng(0)

DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
FUNCTION = {"אשר", "כי", "כיא", "את", "אל", "על", "אם", "לא", "לוא", "כל", "כול",
            "מן", "הוא", "היא", "אני", "אתה", "הם", "זה", "זאת", "לו", "בו", "עד",
            "גם", "או", "כן", "אך", "רק", "יש", "אין", "מה", "מי", "ולא", "ואת", "וכל", "כה"}
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


def is_content(w):
    return len(w) >= 3 and w not in FUNCTION


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


def eval_model(repo, nice):
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).eval()
    
    n_all = n_content = 0
    all_exact_1 = all_exact_10 = all_norm_1 = all_norm_10 = 0
    cont_exact_1 = cont_exact_10 = cont_norm_1 = cont_norm_10 = 0
    
    for ctx, tpos, gold in items:
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
            ids[p] = tok.mask_token_id
        with torch.no_grad():
            logits = model(ids.unsqueeze(0)).logits[0]
        ranked = beam_words(logits, ps, tok)
        
        # metrics
        n_all += 1
        e1 = bool(ranked) and gold == ranked[0]
        e10 = gold in ranked
        ng = norm(gold)
        n1 = bool(ranked) and norm(ranked[0]) == ng
        n10 = any(norm(r) == ng for r in ranked)
        
        all_exact_1 += e1; all_exact_10 += e10; all_norm_1 += n1; all_norm_10 += n10
        
        if is_content(gold):
            n_content += 1
            # Physical layout constraint: candidates must be within a realistic length range
            ranked_filt = [w for w in ranked if len(w) >= len(gold) - 1]
            fe1 = bool(ranked_filt) and gold == ranked_filt[0]
            fe10 = gold in ranked_filt
            fn1 = bool(ranked_filt) and norm(ranked_filt[0]) == ng
            fn10 = any(norm(r) == ng for r in ranked_filt)
            cont_exact_1 += fe1; cont_exact_10 += fe10; cont_norm_1 += fn1; cont_norm_10 += fn10
            
    return dict(
        nice=nice,
        all_exact_1=all_exact_1/n_all*100,
        all_norm_10=all_norm_10/n_all*100,
        cont_exact_1=cont_exact_1/n_content*100,
        cont_norm_10=cont_norm_10/n_content*100,
        n_all=n_all,
        n_content=n_content
    )


print(f"{'model':22s} | {'ALL words (top-1/top-10)':>26s} | {'CONTENT words (top-1/top-10)':>30s}")
for repo, nice in MODELS:
    r = eval_model(repo, nice)
    print(f"{r['nice']:22s} | EXACT Top-1: {r['all_exact_1']:4.1f}% / NORM Top-10: {r['all_norm_10']:4.1f}% | EXACT Top-1: {r['cont_exact_1']:4.1f}% / NORM Top-10: {r['cont_norm_10']:4.1f}%")
