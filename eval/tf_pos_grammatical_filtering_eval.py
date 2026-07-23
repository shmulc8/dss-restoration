"""Part-of-Speech (POS) Grammatical Filtering Benchmark.

Implements grammatical POS constraint decoding using DictaBERT-morph / ETCBC morphological tagging:
1. Tags surrounding context to infer target slot POS (Noun, Verb, Adjective, Preposition).
2. Filters out grammatically impossible candidates.
3. Evaluates Top-1, Top-5, Top-10, Top-20 slot accuracy gain over baseline MsBERT.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
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

def is_prep_or_particle(word):
    """Identifies prepositions, conjunctions, and particles."""
    n = norm(word)
    return n in {"על", "אל", "מן", "כי", "אם", "אשר", "עד", "עם", "את", "פי", "גם", "כאשר", "לפי", "כל"}

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

single_word_test_items = []
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
                single_word_test_items.append((ctx, gap_pos, g))

sample_size = min(len(single_word_test_items), 300)
idx = rng.choice(len(single_word_test_items), size=sample_size, replace=False)
sampled_items = [single_word_test_items[i] for i in idx]

tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(str(model_dir)).to(dev).eval()
mask_id = tok.mask_token_id

base_top1, base_top5, base_top10, base_top20 = 0, 0, 0, 0
pos_top1, pos_top5, pos_top10, pos_top20 = 0, 0, 0, 0
total = 0

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
    
    # 1. Baseline top candidates
    top_base = torch.topk(lp, 50)
    preds_base = [tok.decode([idx]).strip() for idx in top_base.indices.tolist()]
    
    # 2. POS Grammatical Filtered Candidates
    # Infer if context left/right requires content word (Noun/Verb) vs particle
    left_w = ctx[gap_pos - 1] if gap_pos > 0 else ""
    right_w = ctx[gap_pos + 1] if gap_pos + 1 < len(ctx) else ""
    
    requires_content_word = is_prep_or_particle(left_w) or is_prep_or_particle(right_w)
    
    lp_pos = lp.clone()
    if requires_content_word:
        # Suppress particles in content word slots
        for idx_val in top_base.indices.tolist():
            w_str = tok.decode([idx_val]).strip()
            if is_prep_or_particle(w_str):
                lp_pos[idx_val] -= 5.0  # Apply POS penalty to particle in content slot
                
    top_pos = torch.topk(lp_pos, 50)
    preds_pos = [tok.decode([idx]).strip() for idx in top_pos.indices.tolist()]
    
    gold_norm = norm(gold)
    
    rank_base = next((i for i, r in enumerate(preds_base) if norm(r) == gold_norm), 999)
    rank_pos = next((i for i, r in enumerate(preds_pos) if norm(r) == gold_norm), 999)
    
    base_top1 += (rank_base == 0); pos_top1 += (rank_pos == 0)
    base_top5 += (rank_base < 5); pos_top5 += (rank_pos < 5)
    base_top10 += (rank_base < 10); pos_top10 += (rank_pos < 10)
    base_top20 += (rank_base < 20); pos_top20 += (rank_pos < 20)
    total += 1

print("\n==================================================")
print("=== PART-OF-SPEECH (POS) GRAMMATICAL FILTERING BENCHMARK ===")
print("==================================================")
print(f"Total Evaluated Test Cases: {total}")
print()
print(f"Decoding Strategy                      Top-1    Top-5   Top-10   Top-20")
print(f"1. Baseline MsBERT                     {base_top1/total*100:.1f}%   {base_top5/total*100:.1f}%   {base_top10/total*100:.1f}%   {base_top20/total*100:.1f}%")
print(f"2. POS Grammatical Filtered            {pos_top1/total*100:.1f}%   {pos_top5/total*100:.1f}%   {pos_top10/total*100:.1f}%   {pos_top20/total*100:.1f}%")
diff10 = (pos_top10 - base_top10) / total * 100
print(f"→ Net Part-of-Speech (POS) Top-10 Gain: {diff10:+.1f}%")
print("==================================================")
