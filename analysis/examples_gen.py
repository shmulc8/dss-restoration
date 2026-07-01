"""Generate real restoration examples from the best model (MsBERT+span-ft) on
real single-word lacunae: short context with the gap, editor's word, model top-5."""
import sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
from tf.app import use
tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.paths import repo_path

WINDOW, DISP, MIN_PRESERVED = 20, 5, 8
TOPN, BEAM, K = 20, 25, 5
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
rng = np.random.default_rng(3)


def norm(w):
    w = "".join(FINAL.get(c, c) for c in w)
    if w in DIVINE:
        return "יהוה"
    if w == "כיא":
        return "כי"
    if len(w) > 2:
        inner = w[1:-1].replace("ו", "").replace("י", "")
        return w[0] + inner + w[-1]
    return w


A = use("etcbc/dss", silent="deep")
F, L = A.api.F, A.api.L


def winfo(w):
    s = L.d(w, "sign")
    g = "".join(F.glyph.v(x) or "" for x in s)
    r = [F.rec.v(x) for x in s]
    return g, (bool(s) and all(v == 1 for v in r)), (bool(s) and all(v != 1 for v in r))


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
        if not fr or len(g) < 3 or any(c not in HEB for c in g):   # content-ish: len>=3
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
sel = rng.choice(len(items), size=400, replace=False)
items = [items[i] for i in sel]


def beam_words(logits, ps, tok):
    beams = [(0.0, [])]
    for p in ps:
        lp = torch.log_softmax(logits[p], -1)
        t = torch.topk(lp, TOPN)
        beams = sorted([(s + v, seq + [i]) for s, seq in beams
                        for i, v in zip(t.indices.tolist(), t.values.tolist())],
                       key=lambda x: -x[0])[:BEAM]
    out = []
    for _, seq in beams:
        w = tok.decode(seq).replace(" ", "").replace("##", "")
        if w not in out:
            out.append(w)
        if len(out) >= K:
            break
    return out


MODEL_DIR = str(repo_path("ft_msbert_span"))
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR).eval()
for tag in ("EXACT", "IN5", "NORM", "MISS"):
    print(f"\n===== {tag} examples =====")
    shown = 0
    for ctx, tpos, gold in items:
        if shown >= 6:
            break
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
        r = beam_words(logits, ps, tok)
        if not r:
            continue
        this = ("EXACT" if r[0] == gold else "IN5" if gold in r else
                "NORM" if any(norm(x) == norm(gold) for x in r) else "MISS")
        if this != tag:
            continue
        lo, hi = max(0, tpos - DISP), min(len(ctx), tpos + DISP + 1)
        snippet = " ".join(ctx[j] if j != tpos else "⬚⬚⬚" for j in range(lo, hi))
        print(f"  ctx: …{snippet}…")
        print(f"  gold: {gold}    top5: {' / '.join(r)}")
        shown += 1
