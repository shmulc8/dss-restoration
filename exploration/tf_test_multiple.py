"""Test predictions of MsBERT base vs MsBERT+DSS-span-ft on several famous sectarian words in 1QS.
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
F, L = A.api.F, A.api.L

# Find 1QS scroll
target_scroll = None
for sc in F.otype.s("scroll"):
    name = F.scroll.v(sc)
    if name == "1QS":
        target_scroll = sc
        break

if not target_scroll:
    for sc in F.otype.s("scroll"):
        name = F.scroll.v(sc)
        if "1QS" in name or "1Q28" in name:
            target_scroll = sc
            break

words = L.d(target_scroll, "word")
text_words = []
for w in words[:500]:
    signs = L.d(w, "sign")
    g = "".join(F.glyph.v(s) or "" for s in signs)
    text_words.append(g)

# We will search for specific target words in 1QS
targets = ["צדקה", "משפט", "אמת", "ברית", "חסד", "יחד", "עצת"]

# Let's load the models
print("Loading models...")
tok_base = AutoTokenizer.from_pretrained("dicta-il/MsBERT", use_fast=True)
model_base = AutoModelForMaskedLM.from_pretrained("dicta-il/MsBERT").eval()

MODEL_DIR = str(repo_path("ft_msbert_span"))
tok_ft = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model_ft = AutoModelForMaskedLM.from_pretrained(MODEL_DIR).eval()

def test_word(target_gold):
    try:
        idx = text_words.index(target_gold)
    except ValueError:
        print(f"Could not find word '{target_gold}' in the first 500 words of 1QS.")
        return
        
    lo, hi = max(0, idx - 7), min(len(text_words), idx + 8)
    ctx = text_words[lo:hi]
    target_idx = idx - lo
    
    snippet = " ".join(ctx[j] if j != target_idx else "⬚" for j in range(len(ctx)))
    print(f"\nTarget context: '... {snippet} ...'")
    print(f"Gold word: '{target_gold}'")
    
    # Predict for both models
    for tok, model, name in [(tok_base, model_base, "MsBERT base"), (tok_ft, model_ft, "MsBERT+DSS-span-ft")]:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt")
        wmap = {}
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None:
                wmap.setdefault(wid, []).append(pos)
        ps = wmap.get(target_idx)
        if not ps:
            continue
        ids = enc["input_ids"][0].clone()
        for p in ps:
            ids[p] = tok.mask_token_id
            
        with torch.no_grad():
            logits = model(ids.unsqueeze(0)).logits[0]
            
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
        print(f"  {name:20s}: {' / '.join(out)}")
        
        # Length-constrained
        out_cond = [w for w in out if len(w) == len(target_gold)]
        if out_cond:
            print(f"  {' (length-constrained)':20s}: {' / '.join(out_cond)}")

for t in targets:
    test_word(t)
