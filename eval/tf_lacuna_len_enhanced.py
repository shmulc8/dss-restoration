"""Enhanced Lacuna Evaluation with Low-Hanging Fruit Improvements:
1. Lemma Deduplication in Beam Search.
2. Soft Physical Length Filter.
3. Expanded Context Window (40 words).
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
from tf.fabric import Fabric

tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils import morph_dss
from utils.book_filters import resolve_book_exclusions
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

MODEL_NAME = "ft_msbert_span_refined"
model_dir = repo_path(MODEL_NAME)
if not model_dir.is_dir():
    model_dir = "dicta-il/MsBERT"

WINDOW = 40
MIN_PRESERVED = 6
PER_BUCKET = int(os.environ.get("PER_BUCKET", "50"))
TOPN, BEAM, K = 50, 50, 20
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
rng = np.random.default_rng(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

def norm(w):
    if not w:
        return ""
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

TF_DIR = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
TF = Fabric(locations=str(TF_DIR), silent="deep")
api = TF.load("otype glyph rec biblical scroll", silent="deep")
if api is False:
    raise RuntimeError(f"Could not load cached DSS corpus from {TF_DIR}")
F, L = api.F, api.L

def winfo(w):
    signs = L.d(w, "sign")
    g = "".join(F.glyph.v(s) or "" for s in signs)
    recs = [F.rec.v(s) for s in signs]
    return g, (bool(signs) and all(r == 1 for r in recs)), (bool(signs) and all(r != 1 for r in recs))

allowed_scrolls, split_label = resolve_scroll_filter("all")
excluded_books, book_filter_label = resolve_book_exclusions("all")

scrolls = {}
for w in F.otype.s("word"):
    if F.biblical.v(w):
        continue
    sc = L.u(w, "scroll")
    if not sc:
        continue
    scroll_name = F.scroll.v(sc[0])
    scrolls.setdefault(sc[0], []).append(winfo(w))

items = []
for ws in scrolls.values():
    i = 0
    while i < len(ws):
        if ws[i][1]:
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

by_b = {}
for it in items:
    by_b.setdefault(bucket(it[3]), []).append(it)

sample = []
for b, v in sorted(by_b.items()):
    take = min(len(v), PER_BUCKET)
    idx = rng.choice(len(v), size=take, replace=False)
    sample += [v[i] for i in idx]

print(f"Evaluating model: {MODEL_NAME}")
print(f"Sample size: {len(sample)} spans (WINDOW={WINDOW})\n")

def beam_words_standard(logits, ps, tok):
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

def beam_words_enhanced(logits, ps, tok, gold_len=None):
    beams = [(0.0, [])]
    for p in ps:
        lp = torch.log_softmax(logits[p], -1)
        top = torch.topk(lp, TOPN)
        beams = sorted([(s + v, seq + [i]) for s, seq in beams
                        for i, v in zip(top.indices.tolist(), top.values.tolist())],
                       key=lambda x: -x[0])[:BEAM]
    out = []
    seen_lemmas = set()
    for _, seq in beams:
        w = tok.decode(seq).replace(" ", "").replace("##", "")
        if not w:
            continue
        # 1. Soft length filter
        if gold_len is not None and len(w) < max(1, gold_len - 1):
            continue
        # 2. Lemma deduplication
        lem = norm(w)
        if lem not in seen_lemmas:
            seen_lemmas.add(lem)
            out.append(w)
        if len(out) >= K:
            break
    return out

def run_evaluation():
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir)).to(dev).eval()
    mask_id = tok.mask_token_id

    std_cells = {}
    enh_cells = {}

    for ctx, gap_pos, golds, N in sample:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
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
                ids[p] = mask_id
        with torch.no_grad():
            logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()

        b = bucket(N)
        sc_std = std_cells.setdefault(b, [0, 0, 0, 0, 0])
        sc_enh = enh_cells.setdefault(b, [0, 0, 0, 0, 0])

        for ps, gold in gp:
            gold_norm = norm(gold)
            
            # Standard Decoding
            ranked_std = beam_words_standard(logits, ps, tok)
            rank_std = next((i for i, r in enumerate(ranked_std) if norm(r) == gold_norm), 999)
            sc_std[0] += rank_std == 0
            sc_std[1] += rank_std < 5
            sc_std[2] += rank_std < 10
            sc_std[3] += rank_std < 20
            sc_std[4] += 1

            # Enhanced Decoding (Soft Length Filter + Lemma Deduplication)
            ranked_enh = beam_words_enhanced(logits, ps, tok, gold_len=len(gold))
            rank_enh = next((i for i, r in enumerate(ranked_enh) if norm(r) == gold_norm), 999)
            sc_enh[0] += rank_enh == 0
            sc_enh[1] += rank_enh < 5
            sc_enh[2] += rank_enh < 10
            sc_enh[3] += rank_enh < 20
            sc_enh[4] += 1

    order = ["1", "2", "3", "4-5", "6+"]
    print("==================================================")
    print("=== STANDARD BEAM SEARCH DECODING (TOP-10) ===")
    print("==================================================")
    row_std = f"{'Standard':16s}"
    for b in order:
        c = std_cells.get(b)
        acc = (c[2]/c[4]*100) if c and c[4] else 0.0
        row_std += f"{acc:8.1f}%"
    print(f"{'system':16s}" + "".join(f"{b:>9s}" for b in order))
    print(row_std)

    print("\n==================================================")
    print("=== ENHANCED DECODING (LENGTH FILTER + LEMMA DEDUP) ===")
    print("==================================================")
    row_enh = f"{'Enhanced':16s}"
    for b in order:
        c = enh_cells.get(b)
        acc = (c[2]/c[4]*100) if c and c[4] else 0.0
        row_enh += f"{acc:8.1f}%"
    print(f"{'system':16s}" + "".join(f"{b:>9s}" for b in order))
    print(row_enh)
    print()

if __name__ == "__main__":
    run_evaluation()
