"""Continue MsBERT span-ft training with context-noise augmentation.

Same recipe as finetune_span_continue_noparticles.py (continue from ft_msbert_span,
exclude 1-char targets), plus Pythia-style context degradation: a fraction of the
NON-target words is additionally replaced with [MASK] as corrupted input (label -100,
never predicted). This simulates the damaged surrounding text of real fragments and
discourages the model from leaning on a single adjacent word.
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

from utils.dss_split import load_partition
from utils.paths import repo_path

tlog.set_verbosity_error()

BASE_REPO = str(repo_path("ft_msbert_span"))
OUTDIR = repo_path(os.environ.get("OUTDIR_NAME", "ft_msbert_span_refined_noiseaug"))
MAX_LEN = 160
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH = int(os.environ.get("BATCH", "16"))
LR = float(os.environ.get("LR", "2e-5"))
MASK_FRAC, SPAN_P, SPAN_MAX = 0.15, 0.3, 10
MIN_TARGET_LEN = 2
NOISE_FRAC = float(os.environ.get("NOISE_FRAC", "0.15"))  # fraction of non-target words masked as noise
TRAIN_PARTITION = os.environ.get("TRAIN_PARTITION", "fit")
rng = np.random.default_rng(0)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
train = load_partition(TRAIN_PARTITION)
texts = [row["text"].strip() for row in train]
print(
    f"device={dev} | span-ft-continue-noiseaug | partition={TRAIN_PARTITION} "
    f"| epochs={EPOCHS} | lr={LR:g} | noise_frac={NOISE_FRAC:g} | train chunks={len(texts)}\n"
)


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
        encoded.append((enc["input_ids"], enc.word_ids(), words))

    max_len = max(len(ids) for ids, _, _ in encoded)
    input_ids = torch.full((len(encoded), max_len), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(encoded), max_len), dtype=torch.long)
    labels = torch.full((len(encoded), max_len), -100, dtype=torch.long)

    for batch_idx, (ids, wids, words) in enumerate(encoded):
        input_ids[batch_idx, :len(ids)] = torch.tensor(ids)
        attn[batch_idx, :len(ids)] = 1
        groups = {}
        for pos, word_id in enumerate(wids):
            if word_id is not None and word_id < len(words):
                groups.setdefault(word_id, []).append(pos)
        eligible = [word_id for word_id in groups if len(words[word_id]) >= MIN_TARGET_LEN]
        if not eligible:
            continue

        chosen = span_words(len(eligible))
        chosen_word_ids = {eligible[idx] for idx in chosen}
        for word_id in chosen_word_ids:
            for pos in groups[word_id]:
                labels[batch_idx, pos] = ids[pos]
                draw = rng.random()
                if draw < 0.8:
                    input_ids[batch_idx, pos] = mask_id
                elif draw < 0.9:
                    input_ids[batch_idx, pos] = int(rng.integers(vocab_size))

        # Context-noise augmentation: corrupt non-target words to [MASK] (label stays -100).
        for word_id, positions in groups.items():
            if word_id in chosen_word_ids:
                continue
            if rng.random() < NOISE_FRAC:
                for pos in positions:
                    input_ids[batch_idx, pos] = mask_id

    return input_ids.to(dev), attn.to(dev), labels.to(dev)


def finetune():
    if not Path(BASE_REPO).is_dir():
        raise RuntimeError(f"Base checkpoint not found: {BASE_REPO}")

    print(f"=== {BASE_REPO} -> {OUTDIR} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_REPO, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(BASE_REPO).to(dev).train()
    mask_id, vocab_size = tok.mask_token_id, model.config.vocab_size
    opt = AdamW(model.parameters(), lr=LR)
    steps_per_epoch = math.ceil(len(texts) / BATCH)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(texts))
        total_loss = 0.0
        for step in range(steps_per_epoch):
            batch = [texts[i] for i in order[step * BATCH:(step + 1) * BATCH]]
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
print("ALL DONE")
