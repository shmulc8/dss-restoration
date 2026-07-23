"""Evaluate TavBERT (tau/tavbert-he) on DSS lacuna restoration.

TavBERT is Omri Keren et al.'s character-level Masked Language Model for Hebrew.
This script tests TavBERT on held-out non-biblical DSS lacunae spans.
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

MODEL_ID = "tau/tavbert-he"
WINDOW = 40
MIN_PRESERVED = 6
PER_BUCKET = int(os.environ.get("PER_BUCKET", "50"))
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
    take = len(v) if PER_BUCKET <= 0 else min(len(v), PER_BUCKET)
    if take > 0:
        idx = rng.choice(len(v), size=take, replace=False)
        sample += [v[i] for i in idx]

print(f"Evaluating TavBERT ({MODEL_ID}) on heldout split ({split_label})")
print(f"Sample size: {len(sample)} test spans (PER_BUCKET={PER_BUCKET})\n")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(dev).eval()
mask_id = tok.mask_token_id

slot_cells = {}

for span_idx, (ctx, gap_pos, golds, N) in enumerate(sample, start=1):
    # TavBERT processes full sentence text as character sequence
    # Reconstruct text with target words replaced by [MASK] characters
    full_text_tokens = []
    gold_char_spans = []

    for idx, word in enumerate(ctx):
        if idx in gap_pos:
            gold_word = ctx[idx]
            # Replace each character of the gold word with a [MASK] token
            mask_char_len = len(gold_word)
            start_tok_idx = len(full_text_tokens)
            for _ in range(mask_char_len):
                full_text_tokens.append("[MASK]")
            end_tok_idx = len(full_text_tokens)
            gold_char_spans.append((start_tok_idx, end_tok_idx, gold_word))
        else:
            # Tokenize regular context word into characters
            chars = tok.tokenize(word)
            full_text_tokens.extend(chars)
        full_text_tokens.append(" ") # space separator

    # Encode full character sequence
    input_ids = tok.convert_tokens_to_ids(full_text_tokens)
    if len(input_ids) > 512:
        input_ids = input_ids[:512]

    tensor_ids = torch.tensor([input_ids]).to(dev)

    with torch.no_grad():
        logits = model(tensor_ids).logits[0].cpu()

    b = bucket(N)
    sc = slot_cells.setdefault(b, [0, 0, 0, 0, 0])

    for start_p, end_p, gold in gold_char_spans:
        if end_p >= len(input_ids):
            continue

        # Predict top character beam for this word span
        char_positions = list(range(start_p, end_p))
        beams = [(0.0, [])]
        for pos in char_positions:
            lp = torch.log_softmax(logits[pos], -1)
            top = torch.topk(lp, 20)
            beams = sorted([(s + v, seq + [i]) for s, seq in beams
                            for i, v in zip(top.indices.tolist(), top.values.tolist())],
                           key=lambda x: -x[0])[:50]
        
        candidates = []
        for _, seq in beams:
            cand_chars = tok.convert_ids_to_tokens(seq)
            cand_word = "".join(cand_chars).replace(" ", "")
            if cand_word not in candidates:
                candidates.append(cand_word)
            if len(candidates) >= 20:
                break

        gold_norm = norm(gold)
        rank = next((i for i, r in enumerate(candidates) if norm(r) == gold_norm), 999)

        sc[0] += rank == 0
        sc[1] += rank < 5
        sc[2] += rank < 10
        sc[3] += rank < 20
        sc[4] += 1

    if span_idx % 25 == 0:
        print(f"Processed {span_idx}/{len(sample)} test spans...", flush=True)

order = ["1", "2", "3", "4-5", "6+"]
print("\n==================================================")
print("=== TavBERT (tau/tavbert-he) SLOT-LEVEL ACCURACY ===")
print("==================================================")
for metric_idx, metric_name in enumerate(["Top-1", "Top-5", "Top-10", "Top-20"]):
    row = f"{metric_name:16s}"
    for b in order:
        c = slot_cells.get(b)
        acc = (c[metric_idx]/c[4]*100) if c and c[4] else 0.0
        row += f"{acc:8.1f}%"
    print(f"{'bucket':16s}" + "".join(f"{b:>9s}" for b in order))
    print(row)
    print()
