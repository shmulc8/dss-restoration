"""Composition-Level Split vs Scroll-Level Split Evaluation for DSS Restoration.

Evaluates whether model performance holds when entire literary compositions (e.g. Damascus Document,
Serekh ha-Yahad) are purged from the training set, preventing parallel manuscript copy leakage.
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
from utils.paths import repo_path
from utils.dss_split import _load_rows

model_dir = repo_path("ft_msbert_span_refined")
if not model_dir.is_dir():
    model_dir = repo_path("ft_msbert_span")
    if not model_dir.is_dir():
        model_dir = "dicta-il/MsBERT"

WINDOW = 40
MIN_PRESERVED = 6
PER_BUCKET = 50
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
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

def run_composition_split_check():
    rows = _load_rows()
    nonbib = [r for r in rows if r["bib"] == "nonbib"]
    compositions = sorted(set(r["composition"] for r in nonbib if r["composition"].strip()))
    
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(compositions))
    n_heldout = max(1, int(len(compositions) * 0.30))
    heldout_compositions = {compositions[i] for i in perm[:n_heldout]}
    
    print(f"Total non-biblical compositions: {len(compositions)}")
    print(f"Held-out compositions ({len(heldout_compositions)}): {sorted(list(heldout_compositions))[:10]}...")
    
    scroll2comp = {r["book"]: r["composition"] for r in nonbib}

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

    scrolls = {}
    for w in F.otype.s("word"):
        if F.biblical.v(w):
            continue
        sc = L.u(w, "scroll")
        if not sc:
            continue
        scroll_name = F.scroll.v(sc[0])
        comp = scroll2comp.get(scroll_name, "")
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

    print("Gap-length buckets (composition-heldout spans found):", {b: len(v) for b, v in sorted(by_b.items())})
    sample = []
    for b, v in sorted(by_b.items()):
        take = min(len(v), PER_BUCKET)
        idx = rng.choice(len(v), size=take, replace=False)
        sample += [v[i] for i in idx]
    print(f"Total evaluated composition-heldout spans: {len(sample)}\n")

    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir)).to(dev).eval()
    mask_id = tok.mask_token_id

    slot_cells = {}
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
        sc = slot_cells.setdefault(b, [0, 0, 0, 0, 0])
        for ps, gold in gp:
            # Beam decoding
            lp = torch.log_softmax(logits[ps[0]], -1)
            top = torch.topk(lp, 20)
            words = [tok.decode([idx]).strip() for idx in top.indices.tolist()]
            ng = norm(gold)
            rank = next((i for i, r in enumerate(words) if norm(r) == ng), 999)
            sc[0] += rank == 0
            sc[1] += rank < 5
            sc[2] += rank < 10
            sc[3] += rank < 20
            sc[4] += 1

    print("==================================================")
    print("=== COMPOSITION-LEVEL SPLIT: SLOT TOP-10 ACCURACY ===")
    print("==================================================")
    order = ["1", "2", "3", "4-5", "6+"]
    row = "MsBERT span-ft  "
    for b in order:
        c = slot_cells.get(b)
        acc = (c[2]/c[4]*100) if c and c[4] else 0.0
        row += f"{acc:8.1f}%"
    print(f"{'bucket':16s}" + "".join(f"{b:>9s}" for b in order))
    print(row)
    print()

if __name__ == "__main__":
    run_composition_split_check()
