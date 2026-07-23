"""Automated Comparative Epigraphic Scorer.

Scoring and ranking competing human scholar conjectures (e.g. DJD vs. Qimron QTD vs. SQE)
using pseudo-log-likelihood scores from MsBERT ft-SPAN-refined.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog

tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import repo_path

MODEL_NAME = "ft_msbert_span_refined"
model_dir = repo_path(MODEL_NAME)
if not model_dir.is_dir():
    model_dir = "dicta-il/MsBERT"

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(str(model_dir)).to(dev).eval()

def score_conjecture(left_context, conjecture_words, right_context):
    """Calculates pseudo-log-likelihood score for a proposed scholar conjecture."""
    full_sentence = left_context + conjecture_words + right_context
    enc = tok(full_sentence, is_split_into_words=True, return_tensors="pt")
    
    wmap = {}
    for pos, wid in enumerate(enc.word_ids(0)):
        if wid is not None:
            wmap.setdefault(wid, []).append(pos)
            
    start_conj_idx = len(left_context)
    end_conj_idx = start_conj_idx + len(conjecture_words)
    conj_word_indices = list(range(start_conj_idx, end_conj_idx))
    
    total_log_prob = 0.0
    word_scores = []
    
    for w_idx in conj_word_indices:
        target_word = full_sentence[w_idx]
        target_positions = wmap.get(w_idx)
        if not target_positions:
            continue
            
        masked_ids = enc["input_ids"][0].clone()
        for p in target_positions:
            masked_ids[p] = tok.mask_token_id
            
        with torch.no_grad():
            logits = model(masked_ids.unsqueeze(0).to(dev)).logits[0].cpu()
            
        log_probs = F.log_softmax(logits[target_positions[0]], dim=-1)
        target_token_id = enc["input_ids"][0][target_positions[0]]
        word_log_prob = log_probs[target_token_id].item()
        
        total_log_prob += word_log_prob
        word_scores.append((target_word, math_exp_safe(word_log_prob)))
        
    avg_log_prob = total_log_prob / max(1, len(conjecture_words))
    perplexity = math_exp_safe(-avg_log_prob)
    return {
        "conjecture": " ".join(conjecture_words),
        "joint_log_prob": round(total_log_prob, 3),
        "avg_log_prob": round(avg_log_prob, 3),
        "perplexity": round(perplexity, 2),
        "word_scores": word_scores
    }

def math_exp_safe(val):
    try:
        import math
        return math.exp(val)
    except:
        return 0.0

if __name__ == "__main__":
    print("==================================================")
    print("=== AUTOMATED COMPARATIVE EPIGRAPHIC SCORER ===")
    print("==================================================")
    
    # Real 1QS Case Study: Competing Scholar Reconstructions
    left_ctx = ["לעשות", "אמת", "ו"]
    right_ctx = ["ו", "משפט", "ב", "ארץ"]
    
    conjecture_A = ["צדקה"]          # DJD Standard Edition
    conjecture_B = ["חורב"]          # Alternative Epigraphic Conjecture
    conjecture_C = ["חסד"]           # Synonymous Biblical Conjecture
    
    res_A = score_conjecture(left_ctx, conjecture_A, right_ctx)
    res_B = score_conjecture(left_ctx, conjecture_B, right_ctx)
    res_C = score_conjecture(left_ctx, conjecture_C, right_ctx)
    
    results = sorted([res_A, res_B, res_C], key=lambda x: -x["avg_log_prob"])
    
    print("\nCompeting Scholar Reconstructions for: לעשות אמת ו [ ? ] ו משפט ב ארץ\n")
    for rank, res in enumerate(results, start=1):
        print(f"Rank {rank}: '{res['conjecture']}'")
        print(f"  └─ Avg Log-Likelihood: {res['avg_log_prob']} | Perplexity: {res['perplexity']}")
        print(f"  └─ Word Likelihood Probability: {res['word_scores'][0][1]*100:.2f}%")
        print()
