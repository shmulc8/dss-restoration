"""Cross-Epoch Generalization Benchmark: Evaluating MsBERT ft-SPAN-refined on Historical Hebrew Text.

Tests how well our Qumran-finetuned model restores lacunae on historical Hebrew texts.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog

tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import morph_dss
from utils.paths import repo_path

MODEL_NAME = "ft_msbert_span_refined"
model_dir = repo_path(MODEL_NAME)
if not model_dir.is_dir():
    model_dir = "dicta-il/MsBERT"

dev = "mps" if torch.backends.mps.is_available() else "cpu"
rng = np.random.default_rng(42)

print(f"Loading historical Hebrew dataset from HF...")
ds = load_dataset("dicta-il/hebrew-space-restoration-corpus", split="test[:100]")

tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(str(model_dir)).to(dev).eval()
mask_id = tok.mask_token_id

top1_hits, top5_hits, top10_hits, total = 0, 0, 0, 0

for item in ds:
    clean_text = item["output"]
    words = clean_text.split()
    if len(words) < 10:
        continue
    
    # Pick a random middle target word to mask
    target_idx = rng.integers(3, len(words) - 3)
    gold_word = words[target_idx]
    
    # Construct context window with [MASK] at target
    ctx_words = list(words)
    ctx_words[target_idx] = "[MASK]"
    
    enc = tok(ctx_words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
    wmap = {}
    for pos, wid in enumerate(enc.word_ids(0)):
        if wid is not None:
            wmap.setdefault(wid, []).append(pos)
            
    mask_positions = wmap.get(target_idx)
    if not mask_positions:
        continue
        
    ids = enc["input_ids"][0].clone()
    for p in mask_positions:
        ids[p] = mask_id
        
    with torch.no_grad():
        logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()
        
    lp = torch.log_softmax(logits[mask_positions[0]], -1)
    top = torch.topk(lp, 20)
    preds = [tok.decode([idx]).strip() for idx in top.indices.tolist()]
    
    gold_norm = morph_dss.lemma(gold_word)
    rank = next((i for i, r in enumerate(preds) if morph_dss.lemma(r) == gold_norm), 999)
    
    top1_hits += (rank == 0)
    top5_hits += (rank < 5)
    top10_hits += (rank < 10)
    total += 1

print("\n==================================================")
print("=== CROSS-EPOCH HISTORICAL HEBREW EVALUATION ===")
print("==================================================")
print(f"Total Test Sentences: {total}")
print(f"Top-1  Slot Accuracy: {top1_hits/total*100:.1f}%")
print(f"Top-5  Slot Accuracy: {top5_hits/total*100:.1f}%")
print(f"Top-10 Slot Accuracy: {top10_hits/total*100:.1f}%")
print("==================================================")
