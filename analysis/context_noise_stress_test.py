"""Context Degradation Ablation (Stress Test) on DSS heldout split.

Simulates Pythia-style manuscript damage by randomly masking 10%, 25%, and 40%
of the surrounding context words, measuring how accuracy decays under context loss.
"""
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils import morph_dss
from utils.book_filters import resolve_book_exclusions
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

MODEL_DIR = repo_path("ft_msbert_span_refined")
if not MODEL_DIR.is_dir():
    MODEL_DIR = "dicta-il/MsBERT"

WINDOW = 40
MIN_PRESERVED = 6
TOPN, BEAM, K = 50, 5, 5
SPLIT_MODE = "heldout"
BOOK_FILTER_MODE = "no-aram"

HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
rng = np.random.default_rng(42)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

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

def heb(g):
    return len(g) >= 2 and all(ch in HEB for ch in g)

def bucket(n):
    return "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4-5" if n <= 5 else "6+"

# Load TF
from tf.fabric import Fabric
TF_DIR = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
TF = Fabric(locations=str(TF_DIR), silent="deep")
api = TF.load("otype glyph rec biblical scroll", silent="deep")
F, L = api.F, api.L

def winfo(w):
    signs = L.d(w, "sign")
    g = "".join(F.glyph.v(s) or "" for s in signs)
    recs = [F.rec.v(s) for s in signs]
    return g, (bool(signs) and all(r == 1 for r in recs)), (bool(signs) and all(r != 1 for r in recs))

# Load scrolls
allowed_scrolls, _ = resolve_scroll_filter(SPLIT_MODE)
excluded_books, _ = resolve_book_exclusions(BOOK_FILTER_MODE)

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
    if scroll_name in excluded_books:
        continue
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
            if 1 <= N <= 5:
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

# Sample 30 cases per bucket to run fast
by_b = {}
for it in items:
    by_b.setdefault(bucket(it[3]), []).append(it)

sampled_items = []
for b, v in by_b.items():
    take = min(len(v), 30)
    idx = rng.choice(len(v), size=take, replace=False)
    sampled_items += [v[i] for i in idx]

print(f"Loaded {len(sampled_items)} test items for degradation stress test.")

# Load Model
print(f"Loading model from {MODEL_DIR}...")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR)).to(dev).eval()

def beam_autoregressive(model, input_ids, gap_token_positions, tok, beam_width=5):
    beams = [(0.0, input_ids.clone().to(dev), [])]
    for slot_idx, ps in enumerate(gap_token_positions):
        new_beams = []
        for score, current_ids, pred_words in beams:
            with torch.no_grad():
                logits = model(current_ids.unsqueeze(0).to(dev)).logits[0].cpu()
            slot_beams = [(0.0, [])]
            for p in ps:
                lp = torch.log_softmax(logits[p], -1)
                top = torch.topk(lp, TOPN)
                slot_beams = sorted(
                    [(s + v, seq + [i]) for s, seq in slot_beams
                     for i, v in zip(top.indices.tolist(), top.values.tolist())],
                    key=lambda x: -x[0]
                )[:beam_width]
            for slot_score, seq in slot_beams:
                word = tok.decode(seq).replace(" ", "").replace("##", "")
                new_ids = current_ids.clone()
                for i, p in enumerate(ps):
                    new_ids[p] = seq[i]
                new_beams.append((score + slot_score, new_ids, pred_words + [word]))
        beams = sorted(new_beams, key=lambda x: -x[0])[:beam_width]
    return beams[0][2] if beams else []

# Stress Test Loop
noise_levels = [0.0, 0.10, 0.25, 0.40]
results = {}

for noise in noise_levels:
    print(f"Running evaluation with {int(noise*100)}% context noise...", flush=True)
    slot_hits, slot_total = 0, 0
    seq_hits, seq_total = 0, 0
    
    for ctx_words, gap_pos_idx, golds, N in sampled_items:
        # Construct input sequence
        # We need to tokenize with word mapping
        enc = tok(ctx_words, is_split_into_words=True, return_tensors="pt")
        word_map = {}
        for pos, word_id in enumerate(enc.word_ids(0)):
            if word_id is not None:
                word_map.setdefault(word_id, []).append(pos)
                
        # Define slots
        gap_token_positions = [word_map.get(gp) for gp in gap_pos_idx]
        if any(positions is None for positions in gap_token_positions):
            continue
            
        ids = enc["input_ids"][0].clone()
        
        # Mask target gap
        for positions in gap_token_positions:
            for pos in positions:
                ids[pos] = tok.mask_token_id
                
        # Apply noise: randomly mask other context words with probability `noise`
        for w_idx in range(len(ctx_words)):
            if w_idx not in gap_pos_idx:
                if rng.random() < noise:
                    positions = word_map.get(w_idx)
                    if positions:
                        for pos in positions:
                            ids[pos] = tok.mask_token_id
                            
        # Predict
        preds = beam_autoregressive(model, ids, gap_token_positions, tok, beam_width=5)
        
        # Eval
        is_seq_hit = True
        for j, (gold, pred) in enumerate(zip(golds, preds)):
            is_hit = norm(pred) == norm(gold)
            if is_hit:
                slot_hits += 1
            else:
                is_seq_hit = False
            slot_total += 1
            
        if is_seq_hit:
            seq_hits += 1
        seq_total += 1
        
    results[noise] = {
        "slot_acc": (slot_hits / slot_total) * 100 if slot_total else 0,
        "seq_acc": (seq_hits / seq_total) * 100 if seq_total else 0,
    }

lines = [
    "# Context Degradation Ablation (Pythia-style stress test)",
    "",
    f"Randomly masks a fraction of the surrounding context words on {len(sampled_items)} "
    f"held-out non-biblical cases (≤30 per gap-length bucket), then measures restoration accuracy.",
    "",
    "| Context Noise Level | Slot-Level Top-1 | Sequence-Level Top-1 |",
    "| ------------------- | ---------------- | -------------------- |",
]
for noise in noise_levels:
    res = results[noise]
    lines.append(f"| {int(noise*100)}% noise | {res['slot_acc']:.1f}% | {res['seq_acc']:.1f}% |")

report = "\n".join(lines) + "\n"
out = ROOT / "analysis" / "reports" / "context_noise_ablation.md"
out.write_text(report, encoding="utf-8")
print("\n" + report)
print(f"Written to {out}")
