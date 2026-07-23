"""Track B: Expanded Corpus Fine-Tuning with Augmentation.

Augments the DSS training corpus with:
1. Synonym substitution from a small hand-written DSS-oriented map
2. Context window shifting
3. Word dropout
4. Likely-clitic joining

Fine-tunes ft_msbert_span_refined on the combined expanded corpus.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.clitic_join import join_likely_clitics
from utils.dss_split import load_partition
from utils.paths import repo_path

tlog.set_verbosity_error()

BASE_REPO = str(repo_path(os.environ.get("BASE_REPO", "ft_msbert_span_refined")))
OUTDIR = repo_path(os.environ.get("OUTDIR_NAME", "ft_msbert_span_refined_expanded"))
MAX_LEN = 160
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH = int(os.environ.get("BATCH", "16"))
LR = float(os.environ.get("LR", "1.5e-5"))
MASK_FRAC, SPAN_P, SPAN_MAX = 0.15, 0.3, 10
rng = np.random.default_rng(42)

dev = "mps" if torch.backends.mps.is_available() else "cpu"

HEB = set(chr(c) for c in range(0x05D0, 0x05EB))

# ── DSS Synonym Pairs (Period-Appropriate Second Temple Hebrew) ──
SYNONYM_PAIRS = [
    ("אמת", "צדקה"), ("אמת", "צדק"),
    ("משפט", "דין"), ("חסד", "רחמים"),
    ("רשע", "חטא"), ("תורה", "חוק"),
    ("עולם", "נצח"), ("ברית", "עדות"),
    ("כבוד", "הדר"), ("גבורה", "עוז"),
    ("דעת", "בינה"), ("חכמה", "שכל"),
    ("נפש", "רוח"), ("לב", "נפש"),
    ("מעשה", "פעולה"), ("דבר", "מלה"),
    ("ארץ", "אדמה"), ("שמים", "מרום"),
    ("אור", "נגה"), ("חושך", "אפלה"),
    ("טוב", "ישר"), ("רע", "חטא"),
    ("קדוש", "טהור"), ("טמא", "חלל"),
    ("עם", "גוי"), ("איש", "אדם"),
    ("מלך", "שר"), ("כהן", "לוי"),
    ("מלחמה", "קרב"), ("שלום", "מנוחה"),
]

# Build bidirectional synonym map
SYNONYM_MAP = {}
for a, b in SYNONYM_PAIRS:
    SYNONYM_MAP.setdefault(a, set()).add(b)
    SYNONYM_MAP.setdefault(b, set()).add(a)

def heb_word(w):
    return len(w) >= 2 and all(ch in HEB for ch in w)

# ── Load Base Training Data ──
train = load_partition("train")
base_texts = [row["text"].strip() for row in train]
print(f"Base training chunks: {len(base_texts)}")

# ── Augmentation Strategy 1: Synonym Substitution ──
def synonym_augment(text, prob=0.15):
    """Replace words with period-appropriate synonyms at random."""
    words = text.split()
    augmented = []
    changed = False
    for w in words:
        if w in SYNONYM_MAP and rng.random() < prob:
            synonyms = list(SYNONYM_MAP[w])
            replacement = rng.choice(synonyms)
            augmented.append(replacement)
            changed = True
        else:
            augmented.append(w)
    if changed:
        return " ".join(augmented)
    return None

# ── Augmentation Strategy 2: Context Window Shuffling ──
def window_shift_augment(text, shift_range=5):
    """Create new training sample by shifting context window."""
    words = text.split()
    if len(words) < 30:
        return None
    shift = int(rng.integers(-shift_range, shift_range + 1))
    if shift == 0:
        return None
    if shift > 0:
        shifted = words[shift:]
    else:
        shifted = words[:shift]
    if len(shifted) >= 20:
        return " ".join(shifted)
    return None

# ── Augmentation Strategy 3: Word Dropout (Noise Injection) ──
def word_dropout_augment(text, drop_prob=0.05):
    """Randomly drop words to simulate damaged/missing context."""
    words = text.split()
    kept = [w for w in words if rng.random() > drop_prob]
    if len(kept) >= 20 and len(kept) < len(words):
        return " ".join(kept)
    return None

# ── Augmentation Strategy 4: Clitic Join Augmentation ──
def clitic_augment(text):
    """Join likely clitics (reusing existing clitic joiner)."""
    joined, merges = join_likely_clitics(text, prefixes="ובכלמשה", min_next_len=3)
    if merges > 0:
        return joined
    return None

# ── Generate All Augmented Texts ──
aug_synonym = []
aug_window = []
aug_dropout = []
aug_clitic = []

for text in base_texts:
    # Synonym substitution (2 passes for diversity)
    for _ in range(2):
        result = synonym_augment(text, prob=0.12)
        if result:
            aug_synonym.append(result)

    # Context window shifting
    result = window_shift_augment(text, shift_range=8)
    if result:
        aug_window.append(result)

    # Word dropout
    result = word_dropout_augment(text, drop_prob=0.06)
    if result:
        aug_dropout.append(result)

    # Clitic join
    result = clitic_augment(text)
    if result:
        aug_clitic.append(result)

print(f"Synonym augmented chunks: {len(aug_synonym)}")
print(f"Window shift augmented chunks: {len(aug_window)}")
print(f"Word dropout augmented chunks: {len(aug_dropout)}")
print(f"Clitic join augmented chunks: {len(aug_clitic)}")

# Combine all
all_texts = base_texts + aug_synonym + aug_window + aug_dropout + aug_clitic
print(f"Total combined training chunks: {len(all_texts)}")

# ── Training Loop (identical structure to finetune_span_continue_cliticaug.py) ──
def span_words(nwords):
    target = max(1, round(nwords * MASK_FRAC))
    chosen, tries = set(), 0
    while len(chosen) < target and tries < 50:
        tries += 1
        span_len = min(int(rng.geometric(SPAN_P)), SPAN_MAX, nwords)
        start = int(rng.integers(0, max(1, nwords - span_len + 1)))
        span = set(range(start, min(start + span_len, nwords)))
        if span & chosen:
            continue
        chosen |= span
    return chosen


def make_batch(batch_texts, tok, mask_id, vocab_size):
    encoded = []
    for text in batch_texts:
        words = text.split()
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN)
        encoded.append((enc["input_ids"], enc.word_ids()))

    max_len = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), max_len), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(encoded), max_len), dtype=torch.long)
    labels = torch.full((len(encoded), max_len), -100, dtype=torch.long)

    for batch_idx, (ids, wids) in enumerate(encoded):
        input_ids[batch_idx, :len(ids)] = torch.tensor(ids)
        attn[batch_idx, :len(ids)] = 1
        groups = {}
        for pos, word_id in enumerate(wids):
            if word_id is not None:
                groups.setdefault(word_id, []).append(pos)
        words = list(groups)
        if not words:
            continue
        chosen = span_words(len(words))
        for word_idx in chosen:
            for pos in groups[words[word_idx]]:
                labels[batch_idx, pos] = ids[pos]
                draw = rng.random()
                if draw < 0.8:
                    input_ids[batch_idx, pos] = mask_id
                elif draw < 0.9:
                    input_ids[batch_idx, pos] = int(rng.integers(vocab_size))

    return input_ids.to(dev), attn.to(dev), labels.to(dev)


def finetune():
    if not Path(BASE_REPO).is_dir():
        raise RuntimeError(f"Base checkpoint not found: {BASE_REPO}")

    print(f"\n=== {Path(BASE_REPO).name} -> {OUTDIR} ===")
    print(f"device={dev} | epochs={EPOCHS} | lr={LR:g} | batch={BATCH} | chunks={len(all_texts)}")
    tok = AutoTokenizer.from_pretrained(BASE_REPO, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(BASE_REPO).to(dev).train()
    mask_id, vocab_size = tok.mask_token_id, model.config.vocab_size
    opt = AdamW(model.parameters(), lr=LR)
    steps_per_epoch = math.ceil(len(all_texts) / BATCH)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(all_texts))
        total_loss = 0.0
        for step in range(steps_per_epoch):
            batch = [all_texts[i] for i in order[step * BATCH:(step + 1) * BATCH]]
            input_ids, attn, labels = make_batch(batch, tok, mask_id, vocab_size)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += out.loss.item()
        print(f"  epoch {epoch+1}/{EPOCHS}  loss={total_loss/steps_per_epoch:.3f}", flush=True)

    model.save_pretrained(str(OUTDIR))
    tok.save_pretrained(str(OUTDIR))
    print(f"  saved -> {OUTDIR}\n", flush=True)


finetune()
print("EXPANDED CORPUS FINE-TUNING COMPLETE")
