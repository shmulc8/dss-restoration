"""Compute final-layer attention saliency map for slots in demo data.

For each slot, runs a forward pass to extract attention weights paid by the slot's [MASK] token
to all context words, saving the normalized word-level weights to the JSON datasets.
"""
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ft_msbert_span_refined"
if not MODEL_DIR.is_dir():
    MODEL_DIR = "dicta-il/MsBERT"

dev = "mps" if torch.backends.mps.is_available() else "cpu"
DEMO_DATA = ROOT / "demo" / "data"

print(f"Loading model from {MODEL_DIR}...")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
# eager attention required: sdpa (the default) returns an empty attentions tuple,
# so output_attentions=True yields nothing to visualize.
model = AutoModelForMaskedLM.from_pretrained(
    str(MODEL_DIR), attn_implementation="eager"
).to(dev).eval()

def compute_word_attentions(ctx_words, gap_positions):
    """
    ctx_words: list of string tokens at word level
    gap_positions: list of int indices at word level pointing to gaps
    """
    # Tokenize
    enc = tok(ctx_words, is_split_into_words=True, return_tensors="pt")
    input_ids = enc["input_ids"][0].clone()
    
    # Map word index to token indices
    word_map = {}
    for pos, word_id in enumerate(enc.word_ids(0)):
        if word_id is not None:
            word_map.setdefault(word_id, []).append(pos)
            
    # Mask target slots
    gap_token_positions = []
    for gp in gap_positions:
        positions = word_map.get(gp)
        if positions:
            gap_token_positions.append(positions)
            for p in positions:
                input_ids[p] = tok.mask_token_id
        else:
            gap_token_positions.append([])
            
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0).to(dev), output_attentions=True)
        
    # Get attentions from last layer
    # attentions tuple: num_layers of (batch, heads, seq_len, seq_len)
    last_attention = outputs.attentions[-1][0].cpu() # (heads, seq_len, seq_len)
    avg_attention = last_attention.mean(dim=0) # (seq_len, seq_len)
    
    # For each gap slot, extract attention paid to all word tokens
    slot_word_attentions = []
    for slot_idx, slot_tokens in enumerate(gap_token_positions):
        if not slot_tokens:
            slot_word_attentions.append([0.0] * len(ctx_words))
            continue
            
        # Sum attention paid by slot token(s) to other tokens
        token_att = torch.zeros(len(input_ids))
        for t_pos in slot_tokens:
            token_att += avg_attention[t_pos]
            
        # Aggregate token attention to word levels
        word_att = []
        for w_idx in range(len(ctx_words)):
            w_tokens = word_map.get(w_idx, [])
            if w_tokens:
                # Sum or average attention paid to the word's sub-tokens
                score = sum(token_att[tp].item() for tp in w_tokens)
                word_att.append(score)
            else:
                word_att.append(0.0)
                
        # Normalize weights so max is 1.0 (to make visual highlight clear)
        max_v = max(word_att) if word_att else 0
        if max_v > 0:
            word_att = [round(v / max_v, 4) for v in word_att]
        else:
            word_att = [0.0] * len(ctx_words)
            
        slot_word_attentions.append(word_att)
        
    return slot_word_attentions

def process_spans():
    path = DEMO_DATA / "spans.json"
    if not path.exists():
        print("spans.json not found")
        return
    print("Processing spans attentions...")
    spans = json.loads(path.read_text(encoding="utf-8"))
    
    for idx, row in enumerate(spans):
        # We need the context words and gap word indices
        # In spans.json, we can derive them from context_for_reading
        ctx_for_reading = row.get("context_for_reading")
        if not ctx_for_reading:
            continue
        ctx_words = ctx_for_reading.split()
        gap_pos = [i for i, w in enumerate(ctx_words) if "⬚" in w]
        if not gap_pos:
            continue
            
        # Run forward attention pass
        try:
            attentions_list = compute_word_attentions(ctx_words, gap_pos)
            for j, slot in enumerate(row["slot_details"]):
                if j < len(attentions_list):
                    slot["attentions"] = attentions_list[j]
        except Exception as e:
            print(f"  ! span row {idx} attention failed: {e}")
            for slot in row["slot_details"]:
                slot["attentions"] = [0.0] * len(ctx_words)
                
        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1}/{len(spans)} spans...")
            
    path.write_text(json.dumps(spans, ensure_ascii=False, indent=2), encoding="utf-8")
    print("spans.json updated.")

def process_failures():
    path = DEMO_DATA / "failures.json"
    if not path.exists():
        print("failures.json not found")
        return
    print("Processing failures attentions...")
    failures = json.loads(path.read_text(encoding="utf-8"))
    
    for idx, row in enumerate(failures):
        ctx_for_reading = row.get("context_for_reading")
        if not ctx_for_reading:
            continue
        ctx_words = ctx_for_reading.split()
        gap_pos = [i for i, w in enumerate(ctx_words) if "⬚" in w]
        if not gap_pos:
            continue

        try:
            attentions_list = compute_word_attentions(ctx_words, gap_pos)
            if attentions_list:
                row["attentions"] = attentions_list[0]
        except Exception as e:
            print(f"  ! failure row {idx} attention failed: {e}")
            row["attentions"] = [0.0] * len(ctx_words)
            
        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1}/{len(failures)} failures...")
            
    path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print("failures.json updated.")

if __name__ == "__main__":
    process_spans()
    process_failures()
    print("All done!")
