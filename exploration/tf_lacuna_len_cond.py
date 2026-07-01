"""REAL-lacuna evaluation with length conditioning.

Simulates the scenario where the scholar knows the physical size of the gap
(represented by the character length of the gold word). We filter candidate
predictions to only keep those of the correct length, and compare against the
unconstrained baseline. Non-biblical only.
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
from utils.paths import repo_path

MODELS = [("dicta-il/MsBERT", "MsBERT base")]
for d, nice in [("ft_msbert", "MsBERT+DSS-ft"), ("ft_msbert_span", "MsBERT+DSS-span-ft")]:
    model_dir = repo_path(d)
    if model_dir.is_dir():
        MODELS.append((str(model_dir), nice))

WINDOW = 20          # words of context each side of the gap
MIN_PRESERVED = 8    # require this many preserved words in the window
MAX_ITEMS = 400
TOPN, BEAM, K = 20, 40, 10  # slightly wider beam to ensure we capture correct lengths
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
rng = np.random.default_rng(42)

A = use("etcbc/dss", silent="deep")
F, L = A.api.F, A.api.L


def word_info(w):
    signs = L.d(w, "sign")
    glyph = "".join(F.glyph.v(s) or "" for s in signs)
    recs = [F.rec.v(s) for s in signs]
    fully_rec = bool(signs) and all(r == 1 for r in recs)
    preserved = bool(signs) and all(r != 1 for r in recs)
    return glyph, fully_rec, preserved


# per-scroll ordered word lists (non-biblical only), with reconstruction flags
scrolls = {}
for w in F.otype.s("word"):
    if F.biblical.v(w):
        continue
    sc = L.u(w, "scroll")
    if not sc:
        continue
    g, fr, pr = word_info(w)
    scrolls.setdefault(sc[0], []).append((g, fr, pr))

# build items
items = []
for sc, ws in scrolls.items():
    for i, (g, fr, pr) in enumerate(ws):
        if not fr or len(g) < 2 or any(ch not in HEB for ch in g):
            continue
        lo, hi = max(0, i - WINDOW), min(len(ws), i + WINDOW + 1)
        win = ws[lo:hi]
        tgt = i - lo
        ctx, tpos = [], None
        for j, (gg, ffr, ppr) in enumerate(win):
            if len(gg) >= 1 and all(ch in HEB for ch in gg):
                if j == tgt:
                    tpos = len(ctx)
                ctx.append(gg)
        if tpos is None or len(ctx) < MIN_PRESERVED + 1:
            continue
        preserved_ctx = sum(1 for j, (gg, ffr, ppr) in enumerate(win)
                            if ppr and len(gg) >= 1 and all(ch in HEB for ch in gg))
        if preserved_ctx < MIN_PRESERVED:
            continue
        items.append((ctx, tpos, ctx[tpos]))

if len(items) > MAX_ITEMS:
    sel = rng.choice(len(items), size=MAX_ITEMS, replace=False)
    items = [items[i] for i in sel]
print(f"Loaded {len(items)} real-lacuna test items.")


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
    MASK = tok.mask_token_id
    
    c_base = [0, 0, 0, 0]  # unconstrained
    c_cond = [0, 0, 0, 0]  # length-conditioned
    
    for ctx, tpos, gold in items:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=512)
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
        
        # 1. Base unconstrained
        rank = ranked.index(gold) if gold in ranked else 999
        c_base[0] += rank == 0; c_base[1] += rank < 5; c_base[2] += rank < 10; c_base[3] += 1
        
        # 2. Length-conditioned (filter by exact gold word character length)
        ranked_cond = [w for w in ranked if len(w) == len(gold)]
        rank_cond = ranked_cond.index(gold) if gold in ranked_cond else 999
        c_cond[0] += rank_cond == 0; c_cond[1] += rank_cond < 5; c_cond[2] += rank_cond < 10; c_cond[3] += 1
        
    return nice, c_base, c_cond


print(f"{'model':22s} | {'unconstrained (top-1/5/10)':>28s} | {'length-conditioned (top-1/5/10)':>32s}")
for repo, nice in MODELS:
    nice, cb, cc = eval_model(repo, nice)
    nb = max(cb[3], 1)
    nc = max(cc[3], 1)
    print(f"{nice:22s} | {cb[0]/nb*100:5.1f}% / {cb[1]/nb*100:5.1f}% / {cb[2]/nb*100:5.1f}% | {cc[0]/nc*100:5.1f}% / {cc[1]/nc*100:5.1f}% / {cc[2]/nc*100:5.1f}%")
