"""Length-conditioned REAL-lacuna eval on etcbc/dss.

A real lacuna is a contiguous run of N reconstructed words. We blank the whole
run (model knows it's N words), then restore each gap word with the rest of the
gap still masked, bucketed by gap length N. Shows decay with length and whether
DSS-finetuning helps more on the hard, long gaps. Non-biblical only.

MsBERT is whole-word (1 token/word) so N-word gap = N masks, exact. BEREL is
subword: we mask each gap word's true subtoken count (optimistic for BEREL — it
leaks subtoken length; MsBERT gets only the word-count signal).
"""
import os
import sys
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

MODELS = [("dicta-il/MsBERT", "MsBERT base"), ("dicta-il/BEREL", "BEREL base")]
for d, nice in [("ft_msbert", "MsBERT ft-scatter"), ("ft_berel", "BEREL ft-scatter"),
                ("ft_msbert_span", "MsBERT ft-SPAN"), ("ft_berel_span", "BEREL ft-SPAN")]:
    model_dir = repo_path(d)
    if model_dir.is_dir():
        MODELS.append((str(model_dir), nice))

WINDOW = 20
MIN_PRESERVED = 6
PER_BUCKET = 140          # cap gap-words per length bucket
TOPN, BEAM, K = 50, 50, 20
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
rng = np.random.default_rng(0)
def norm(w):                                   # scholar-lenient: finals, matres, divine name
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


def heb(g):
    return len(g) >= 2 and all(ch in HEB for ch in g)


def bucket(n):
    return "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4-5" if n <= 5 else "6+"


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

# items: (ctx_words, gap_ctx_positions[list], golds[list], N)
items = []
for ws in scrolls.values():
    i = 0
    while i < len(ws):
        if ws[i][1]:                                  # start of a reconstructed run
            j = i
            while j < len(ws) and ws[j][1]:
                j += 1
            gap_targets = [k for k in range(i, j) if heb(ws[k][0])]
            N = len(gap_targets)
            if 1 <= N <= 12:
                lo, hi = max(0, i - WINDOW), min(len(ws), j + WINDOW)
                ctx, gap_pos, golds, preserved = [], [], [], 0
                for k in range(lo, hi):
                    g, fr, pr = ws[k]
                    if i <= k < j:
                        if k in gap_targets:
                            gap_pos.append(len(ctx)); golds.append(g); ctx.append(g)
                    elif heb(g):
                        ctx.append(g)
                        if pr:
                            preserved += 1
                if preserved >= MIN_PRESERVED and gap_pos:
                    items.append((ctx, gap_pos, golds, N))
            i = j
        else:
            i += 1

# balance buckets
by_b = {}
for it in items:
    by_b.setdefault(bucket(it[3]), []).append(it)
print("gap-length buckets (spans found):", {b: len(v) for b, v in sorted(by_b.items())})
sample = []
for b, v in by_b.items():
    idx = rng.choice(len(v), size=min(len(v), PER_BUCKET), replace=False)
    sample += [v[i] for i in idx]
print(f"eval spans: {len(sample)}\n")


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
    cells = {}   # bucket -> [t1,t5,t10,t20,n]
    for ctx, gap_pos, golds, N in sample:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=512)
        wmap = {}
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None:
                wmap.setdefault(wid, []).append(pos)
        gp = [(wmap.get(g), gold) for g, gold in zip(gap_pos, golds)]
        if any(ps is None for ps, _ in gp):
            continue
        ids = enc["input_ids"][0].clone()
        for ps, _ in gp:
            for p in ps:
                ids[p] = MASK
        with torch.no_grad():
            logits = model(ids.unsqueeze(0)).logits[0]
        b = bucket(N)
        c = cells.setdefault(b, [0, 0, 0, 0, 0])
        for ps, gold in gp:
            ranked = beam_words(logits, ps, tok)
            ng = norm(gold)
            rank = next((i for i, r in enumerate(ranked) if norm(r) == ng), 999)
            c[0] += rank == 0; c[1] += rank < 5; c[2] += rank < 10; c[3] += rank < 20; c[4] += 1
    return nice, cells


order = ["1", "2", "3", "4-5", "6+"]
res = [eval_model(r, n) for r, n in MODELS]
for metric, mi in [("top-1", 0), ("top-5", 1), ("top-10", 2), ("top-20", 3)]:
    print(f"=== {metric} by gap length (words) ===")
    print(f"{'model':16s}" + "".join(f"{b:>9s}" for b in order))
    for nice, cells in res:
        row = f"{nice:16s}"
        for b in order:
            c = cells.get(b)
            row += f"{(c[mi]/c[4]*100 if c and c[4] else 0):8.1f}%" if c else f"{'-':>9s}"
        print(row)
    print()
ns = res[0][1]
print("n per bucket:", {b: (ns[b][3] if b in ns else 0) for b in order})
