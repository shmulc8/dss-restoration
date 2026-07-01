"""REAL-lacuna restoration eval on etcbc/dss.

Target = a word the editors FULLY reconstructed (all its signs rec=1) — i.e. a
real `[ ]` restoration. We blank it, give the model the surrounding transcription,
and check top-k agreement with the editor's reading. Non-biblical only (canon
recall doesn't count). This is the honest task; synthetic masking was the proxy.
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

MODELS = [("dicta-il/BEREL", "BEREL base"), ("dicta-il/MsBERT", "MsBERT base")]
for d, nice in [("ft_berel", "BEREL+DSS-ft"), ("ft_msbert", "MsBERT+DSS-ft")]:
    model_dir = repo_path(d)
    if model_dir.is_dir():
        MODELS.append((str(model_dir), nice))

WINDOW = 20          # words of context each side of the gap
MIN_PRESERVED = 8    # require this many preserved words in the window
MAX_ITEMS = 500
TOPN, BEAM, K = 20, 25, 10
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
rng = np.random.default_rng(0)

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
    if F.biblical.v(w):                      # skip biblical scrolls
        continue
    sc = L.u(w, "scroll")
    if not sc:
        continue
    g, fr, pr = word_info(w)
    scrolls.setdefault(sc[0], []).append((g, fr, pr))

# build items: (context_words, target_idx, gold_word)
items = []
n_rec_words = n_words = 0
for sc, ws in scrolls.items():
    n_words += len(ws)
    n_rec_words += sum(1 for g, fr, pr in ws if fr)
    for i, (g, fr, pr) in enumerate(ws):
        if not fr or len(g) < 2 or any(ch not in HEB for ch in g):
            continue
        lo, hi = max(0, i - WINDOW), min(len(ws), i + WINDOW + 1)
        win = ws[lo:hi]
        tgt = i - lo
        # render legible words (drop empty-glyph gaps); track target position
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

print(f"non-biblical words: {n_words:,} | fully-reconstructed: {n_rec_words:,} "
      f"({n_rec_words/max(n_words,1)*100:.0f}%)")
if len(items) > MAX_ITEMS:
    sel = rng.choice(len(items), size=MAX_ITEMS, replace=False)
    items = [items[i] for i in sel]
print(f"real-lacuna eval items: {len(items)}\n")


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
    c = [0, 0, 0, 0]
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
        rank = ranked.index(gold) if gold in ranked else 999
        c[0] += rank == 0; c[1] += rank < 5; c[2] += rank < 10; c[3] += 1
    return nice, c


print(f"{'model':16s}{'top-1':>8s}{'top-5':>8s}{'top-10':>8s}{'n':>7s}")
for repo, nice in MODELS:
    nice, c = eval_model(repo, nice)
    n = max(c[3], 1)
    print(f"{nice:16s}{c[0]/n*100:7.1f}%{c[1]/n*100:7.1f}%{c[2]/n*100:7.1f}%{c[3]:7d}")
print("\n(real editorial [ ] reconstructions, non-biblical; gold = editor's reading)")
