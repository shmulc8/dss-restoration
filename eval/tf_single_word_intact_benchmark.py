"""Single-Word Intact Text Lacuna Completion Benchmark.

Evaluates MsBERT ft-SPAN-refined on single-word masks on intact preserved text (rec != 1).
"""
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog

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

allowed_scrolls, _ = resolve_scroll_filter("heldout")

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

single_word_intact_items = []
for ws in scrolls.values():
    for i in range(len(ws)):
        g, is_rec, is_pr = ws[i]
        if is_pr and heb(g):
            lo, hi = max(0, i - WINDOW), min(len(ws), i + WINDOW + 1)
            ctx, gap_pos = [], None
            for k in range(lo, hi):
                word_str, _, pr = ws[k]
                if k == i:
                    gap_pos = len(ctx)
                    ctx.append(word_str)
                elif heb(word_str):
                    ctx.append(word_str)
            if gap_pos is not None:
                single_word_intact_items.append((ctx, gap_pos, g))

# Sample 500 single-word intact test items
sample_size = min(len(single_word_intact_items), 500)
idx = rng.choice(len(single_word_intact_items), size=sample_size, replace=False)
sampled_items = [single_word_intact_items[i] for i in idx]

tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(str(model_dir)).to(dev).eval()
mask_id = tok.mask_token_id

top1, top5, top10, top20, total = 0, 0, 0, 0, 0

for ctx, gap_pos, gold in sampled_items:
    enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
    wmap = {}
    for pos, wid in enumerate(enc.word_ids(0)):
        if wid is not None:
            wmap.setdefault(wid, []).append(pos)
            
    target_positions = wmap.get(gap_pos)
    if not target_positions:
        continue
        
    ids = enc["input_ids"][0].clone()
    for p in target_positions:
        ids[p] = mask_id
        
    with torch.no_grad():
        logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()
        
    lp = torch.log_softmax(logits[target_positions[0]], -1)
    top = torch.topk(lp, 20)
    preds = [tok.decode([idx]).strip() for idx in top.indices.tolist()]
    
    gold_norm = norm(gold)
    rank = next((i for i, r in enumerate(preds) if norm(r) == gold_norm), 999)
    
    top1 += (rank == 0)
    top5 += (rank < 5)
    top10 += (rank < 10)
    top20 += (rank < 20)
    total += 1

print("\n==================================================")
print("=== SINGLE-WORD INTACT TEXT RESTORATION (rec != 1) ===")
print("==================================================")
print(f"Total Single-Word Test Cases: {total}")
print(f"Top-1  Accuracy: {top1/total*100:.1f}%")
print(f"Top-5  Accuracy: {top5/total*100:.1f}%")
print(f"Top-10 Accuracy: {top10/total*100:.1f}%")
print(f"Top-20 Accuracy: {top20/total*100:.1f}%")
print("==================================================")
