"""Test predictions of MsBERT base vs MsBERT+DSS-span-ft on a specific sectarian scroll sentence.
We mask a key theological or sectarian content word and compare predictions.
"""
import os, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
from tf.app import use
tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.paths import repo_path

A = use("etcbc/dss", silent="deep")
F, L, T = A.api.F, A.api.L, A.api.T

# Find a scroll like 1QS (often named 1QS or 1Q28 in etcbc/dss)
target_scroll = None
for sc in F.otype.s("scroll"):
    name = F.scroll.v(sc)
    if name == "1QS":
        target_scroll = sc
        break

if not target_scroll:
    # Look for any scroll containing "1QS" in name
    for sc in F.otype.s("scroll"):
        name = F.scroll.v(sc)
        if "1QS" in name or "1Q28" in name:
            target_scroll = sc
            break

if not target_scroll:
    print("Could not find 1QS scroll in database!")
    sys.exit(1)

print(f"Found scroll: {F.scroll.v(target_scroll)}")

# Print a few lines/words from 1QS to find a good sentence
words = L.d(target_scroll, "word")
print(f"Total words in scroll: {len(words)}")

# Reconstruct sentences from 1QS words
# We look for a sentence with a key theological word like "סרך" or "עצה" or "יחד"
# Let's rebuild the first 300 words
text_words = []
for w in words[:400]:
    signs = L.d(w, "sign")
    g = "".join(F.glyph.v(s) or "" for s in signs)
    text_words.append((w, g))

print("\n--- First few reconstructed sentences/phrases in 1QS ---")
for i in range(0, len(text_words), 15):
    phrase = " ".join(g for w, g in text_words[i:i+15])
    print(f"[{i}]: {phrase}")

# Let's pick a very famous sentence from 1QS Column 1, lines 1-3:
# "למשכיל ... לדורש אל ב[כול ... ] לעשות הטוב והישר לפניו כאשר צוה ביד משה וביד כול עבדיו הנביאים"
# Let's find words in our list that correspond to:
# "לעשות הטוב והישר לפניו כאשר צוה ביד משה"
# In our reconstructed words, let's search for "משה" or "עבדיו" or "הנביאים"
moshe_idx = None
for idx, (w, g) in enumerate(text_words):
    if g == "משה":
        moshe_idx = idx
        break

if moshe_idx is None:
    # let's search for another keyword like "הטוב" or "הישר" or "עבדיו"
    for idx, (w, g) in enumerate(text_words):
        if g in ("הטוב", "הישר", "הנביאים", "צוה"):
            moshe_idx = idx
            break

if moshe_idx is None:
    print("Could not find key sentence words!")
    sys.exit(1)

# Let's build a context around moshe_idx
lo, hi = max(0, moshe_idx - 8), min(len(text_words), moshe_idx + 8)
context_slice = text_words[lo:hi]
sentence = " ".join(g for w, g in context_slice)
target_word_idx = moshe_idx - lo
target_word_gold = context_slice[target_word_idx][1]

print(f"\nTarget sentence: '{sentence}'")
print(f"Masking word: '{target_word_gold}' (index {target_word_idx} in context)")

# Prepare input context
ctx = [g for w, g in context_slice]

# Evaluate models
def eval_sentence(repo, nice):
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).eval()
    
    enc = tok(ctx, is_split_into_words=True, return_tensors="pt")
    wmap = {}
    for pos, wid in enumerate(enc.word_ids(0)):
        if wid is not None:
            wmap.setdefault(wid, []).append(pos)
            
    ps = wmap.get(target_word_idx)
    if not ps:
        print(f"Error mapping target word to token positions in {nice}!")
        return
        
    ids = enc["input_ids"][0].clone()
    for p in ps:
        ids[p] = tok.mask_token_id
        
    with torch.no_grad():
        logits = model(ids.unsqueeze(0)).logits[0]
        
    # Unconstrained predictions
    beams = [(0.0, [])]
    for p in ps:
        lp = torch.log_softmax(logits[p], -1)
        top = torch.topk(lp, 20)
        beams = sorted([(s + v, seq + [i]) for s, seq in beams
                        for i, v in zip(top.indices.tolist(), top.values.tolist())],
                       key=lambda x: -x[0])[:25]
    
    out = []
    for _, seq in beams:
        w = tok.decode(seq).replace(" ", "").replace("##", "")
        if w not in out:
            out.append(w)
        if len(out) >= 5:
            break
            
    print(f"\n### {nice}")
    print(f"  Top-5 predictions: {' / '.join(out)}")
    
    # Let's show character length constrained predictions (simulating physical gap size)
    out_cond = [w for w in out if len(w) == len(target_word_gold)]
    print(f"  Length-constrained (len={len(target_word_gold)}): {' / '.join(out_cond)}")

eval_sentence("dicta-il/MsBERT", "MsBERT base")
model_dir = repo_path("ft_msbert_span")
if model_dir.is_dir():
    eval_sentence(str(model_dir), "MsBERT+DSS-span-ft")
