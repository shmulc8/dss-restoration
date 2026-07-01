"""Validity and leakage checks on the evaluation setup.

1. Verifies that train and test splits are completely disjoint at the scroll/book level.
2. Verifies that the gold word is completely masked and does not leak into the model input.
3. Checks for any subtoken length leakage for MsBERT.
"""
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.dss_split import load_split

def check_split_leakage():
    print("=== Check 1: Split Leakage ===")
    train, test, bib = load_split()
    train_books = set(r["book"] for r in train)
    test_books = set(r["book"] for r in test)
    intersection = train_books.intersection(test_books)
    print(f"Train scrolls: {len(train_books)}")
    print(f"Test scrolls: {len(test_books)}")
    print(f"Intersection: {intersection}")
    assert len(intersection) == 0, "ERROR: Scroll overlap between train and test sets!"
    print("SUCCESS: Train and test sets are completely disjoint at the scroll level.\n")

def check_masking_leakage():
    print("=== Check 2: Masking Leakage ===")
    # Load MsBERT tokenizer
    repo = "dicta-il/MsBERT"
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    MASK = tok.mask_token_id

    # Test sentence from sectarian scroll
    ctx = ["לעשות", "ענוה", "וצדקה", "ומשפט", "ואהבת", "חסד"]
    # Mask the word "וצדקה" (index 2)
    tpos = 2
    gold = ctx[tpos]

    enc = tok(ctx, is_split_into_words=True, return_tensors="pt")
    wmap = {}
    for pos, wid in enumerate(enc.word_ids(0)):
        if wid is not None:
            wmap.setdefault(wid, []).append(pos)
    
    ps = wmap.get(tpos)
    assert ps is not None, "ERROR: Target word position not found in word_ids!"

    # Clone and mask
    ids = enc["input_ids"][0].clone()
    for p in ps:
        ids[p] = MASK

    # Decode target positions
    decoded_target = tok.decode([ids[p] for p in ps]).strip()
    decoded_full = tok.decode(ids)

    print(f"Context: {' '.join(ctx)}")
    print(f"Gold: {gold}")
    print(f"Masked positions: {ps}")
    print(f"Decoded target position: '{decoded_target}'")
    print(f"Decoded full context: '{decoded_full}'")

    assert decoded_target == "[MASK]", f"ERROR: Target not fully masked! Found: {decoded_target}"
    assert gold not in decoded_full, "ERROR: Gold word leaked into the decoded input!"
    print("SUCCESS: Target is completely masked and no gold word content leaked.\n")

def check_subtoken_leakage():
    print("=== Check 3: Subtoken Length Leakage ===")
    repo = "dicta-il/MsBERT"
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    
    # Check if MsBERT is indeed whole-word (no subword splitting for standard words)
    words = ["ומשפט", "וצדקה", "בארץ", "הכוהנים", "יכין"]
    for w in words:
        tokens = tok.tokenize(w)
        print(f"Word: {w:10s} -> Tokens: {tokens}")
        assert len(tokens) == 1, f"ERROR: MsBERT split '{w}' into multiple subtokens: {tokens}"
    print("SUCCESS: MsBERT tokenizes all standard words as a single token (no subtoken length leak).\n")

if __name__ == "__main__":
    check_split_leakage()
    check_masking_leakage()
    check_subtoken_leakage()
    print("ALL VALIDITY CHECKS PASSED SUCCESSFULLY.")
