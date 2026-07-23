"""Comparative Benchmark: Intact Text (Synthetic Masking) vs Real Physical Lacunae.

Evaluates MsBERT ft-SPAN-refined on:
1. Intact / Preserved Text (rec != 1) where words are artificially masked.
2. Real Physical Lacunae (rec == 1) where words were reconstructed by human editors.
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
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

MODEL_NAME = "ft_msbert_span_refined"
model_dir = repo_path(MODEL_NAME)
if not model_dir.is_dir():
    model_dir = "dicta-il/MsBERT"

WINDOW = 40
PER_BUCKET = 50
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
rng = np.random.default_rng(42)
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

allowed_scrolls, split_label = resolve_scroll_filter("heldout")

scrolls = {}
for w in F.otype.s("word"):
    if F.biblical.v(w):
        continue
    sc = L.u(w, "scroll")
    if not sc:
        continue
    scroll_name = F.scroll.v(sc[0])
    if allowed_scrolls is not None and scroll_name not in allowed_scrolls:
        continue
    scrolls.setdefault(sc[0], []).append(winfo(w))

# 1. Real Lacunae Spans (rec == 1)
real_lacuna_items = []
# 2. Intact Text Spans (rec != 1)
intact_text_items = []

for ws in scrolls.values():
    # Collect real lacunae
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
                ctx, gap_pos, golds = [], [], []
                for k in range(lo, hi):
                    g, fr, pr = ws[k]
                    if i <= k < j:
                        if k in gap_targets:
                            gap_pos.append(len(ctx)); golds.append(g); ctx.append(g)
                    elif heb(g):
                        ctx.append(g)
                if gap_pos:
                    real_lacuna_items.append((ctx, gap_pos, golds, N))
            i = j
        else:
            i += 1

    # Collect intact text spans (synthetic masking of healthy text)
    i = 0
    while i < len(ws):
        if ws[i][2] and heb(ws[i][0]):
            for span_len in [1, 2, 3, 4, 6]:
                j = i + span_len
                if j <= len(ws) and all(ws[k][2] and heb(ws[k][0]) for k in range(i, j)):
                    lo, hi = max(0, i - WINDOW), min(len(ws), j + WINDOW)
                    ctx, gap_pos, golds = [], [], []
                    for k in range(lo, hi):
                        g, fr, pr = ws[k]
                        if i <= k < j:
                            gap_pos.append(len(ctx)); golds.append(g); ctx.append(g)
                        elif heb(g):
                            ctx.append(g)
                    if gap_pos:
                        intact_text_items.append((ctx, gap_pos, golds, span_len))
            i += 5
        else:
            i += 1

# Sample balanced items
def sample_buckets(items_list):
    by_b = {}
    for it in items_list:
        by_b.setdefault(bucket(it[3]), []).append(it)
    s = []
    for b in ["1", "2", "3", "4-5", "6+"]:
        v = by_b.get(b, [])
        if not v:
            continue
        take = min(len(v), PER_BUCKET)
        idx = rng.choice(len(v), size=take, replace=False)
        s += [v[i] for i in idx]
    return s

sample_lacuna = sample_buckets(real_lacuna_items)
sample_intact = sample_buckets(intact_text_items)

print(f"Sampled {len(sample_lacuna)} real lacuna spans (rec == 1)")
print(f"Sampled {len(sample_intact)} intact text spans (rec != 1)\n")

tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(str(model_dir)).to(dev).eval()
mask_id = tok.mask_token_id

def eval_dataset(dataset_sample, label):
    cells = {}
    for ctx, gap_pos, golds, N in dataset_sample:
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
        sc = cells.setdefault(b, [0, 0, 0, 0, 0])

        for ps, gold in gp:
            gold_norm = norm(gold)
            lp = torch.log_softmax(logits[ps[0]], -1)
            top = torch.topk(lp, 20)
            words = [tok.decode([idx]).strip() for idx in top.indices.tolist()]
            rank = next((i for i, r in enumerate(words) if norm(r) == gold_norm), 999)

            sc[0] += rank == 0
            sc[1] += rank < 5
            sc[2] += rank < 10
            sc[3] += rank < 20
            sc[4] += 1
    return cells

lacuna_results = eval_dataset(sample_lacuna, "Real Lacunae")
intact_results = eval_dataset(sample_intact, "Intact Text")

order = ["1", "2", "3", "4-5", "6+"]
print("==================================================")
print("=== INTACT TEXT (SYNTHETIC) vs REAL LACUNAE TOP-10 ===")
print("==================================================")
print(f"{'Text Type':28s}" + "".join(f"{b:>9s}" for b in order))

row_intact = f"{'Intact Text (rec != 1)':28s}"
for b in order:
    c = intact_results.get(b)
    acc = (c[2]/c[4]*100) if c and c[4] else 0.0
    row_intact += f"{acc:8.1f}%"
print(row_intact)

row_lacuna = f"{'Real Lacunae (rec == 1)':28s}"
for b in order:
    c = lacuna_results.get(b)
    acc = (c[2]/c[4]*100) if c and c[4] else 0.0
    row_lacuna += f"{acc:8.1f}%"
print(row_lacuna)
print()
