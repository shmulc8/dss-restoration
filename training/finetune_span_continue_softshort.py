"""Continue MsBERT span-ft training with a softer penalty for 1-character targets.

Instead of excluding one-letter words from the objective, keep them in the
training signal but downweight their contribution to the masked-LM loss.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dss_split import load_partition
from utils.paths import repo_path

tlog.set_verbosity_error()

BASE_REPO = str(repo_path("ft_msbert_span"))
OUTDIR = repo_path(os.environ.get("OUTDIR_NAME", "ft_msbert_span_softshort"))
MAX_LEN = 160
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH = int(os.environ.get("BATCH", "16"))
LR = float(os.environ.get("LR", "2e-5"))
MASK_FRAC, SPAN_P, SPAN_MAX = 0.15, 0.3, 10
SHORT_LEN = 1
SHORT_WEIGHT = float(os.environ.get("SHORT_WEIGHT", "0.25"))
TRAIN_PARTITION = os.environ.get("TRAIN_PARTITION", "fit")
rng = np.random.default_rng(0)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
train = load_partition(TRAIN_PARTITION)
texts = [row["text"].strip() for row in train]
print(
    f"device={dev} | span-ft-continue-softshort | short_len<={SHORT_LEN} "
    f"weight={SHORT_WEIGHT:.2f} | partition={TRAIN_PARTITION} "
    f"| epochs={EPOCHS} | lr={LR:g} | train chunks={len(texts)}\n"
)


def count_word_lengths():
    total_words = short_words = 0
    for text in texts:
        words = text.split()
        total_words += len(words)
        short_words += sum(1 for word in words if len(word) <= SHORT_LEN)
    print(
        f"short target words: {short_words}/{total_words} "
        f"({short_words/max(total_words, 1)*100:.1f}%)"
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
    weights = torch.zeros((len(encoded), max_len), dtype=torch.float32)

    for batch_idx, (ids, wids, words) in enumerate(encoded):
        input_ids[batch_idx, :len(ids)] = torch.tensor(ids)
        attn[batch_idx, :len(ids)] = 1
        groups = {}
        for pos, word_id in enumerate(wids):
            if word_id is not None and word_id < len(words):
                groups.setdefault(word_id, []).append(pos)
        if not groups:
            continue

        ordered_word_ids = sorted(groups)
        chosen_indices = span_words(len(ordered_word_ids))
        chosen_word_ids = [ordered_word_ids[idx] for idx in chosen_indices]
        for word_id in chosen_word_ids:
            weight = SHORT_WEIGHT if len(words[word_id]) <= SHORT_LEN else 1.0
            for pos in groups[word_id]:
                labels[batch_idx, pos] = ids[pos]
                weights[batch_idx, pos] = weight
                draw = rng.random()
                if draw < 0.8:
                    input_ids[batch_idx, pos] = mask_id
                elif draw < 0.9:
                    input_ids[batch_idx, pos] = int(rng.integers(vocab_size))

    return input_ids.to(dev), attn.to(dev), labels.to(dev), weights.to(dev)


def weighted_mlm_loss(logits, labels, weights):
    vocab = logits.shape[-1]
    losses = F.cross_entropy(
        logits.view(-1, vocab),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(labels)
    active = labels.ne(-100)
    total_weight = weights[active].sum()
    if total_weight.item() == 0:
        return losses.sum() * 0
    return (losses[active] * weights[active]).sum() / total_weight


def finetune():
    if not Path(BASE_REPO).is_dir():
        raise RuntimeError(f"Base checkpoint not found: {BASE_REPO}")

    print(f"=== {BASE_REPO} -> {OUTDIR} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_REPO, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(BASE_REPO).to(dev).train()
    mask_id, vocab_size = tok.mask_token_id, model.config.vocab_size
    count_word_lengths()
    opt = AdamW(model.parameters(), lr=LR)
    steps_per_epoch = math.ceil(len(texts) / BATCH)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(texts))
        total_loss = 0.0
        for step in range(steps_per_epoch):
            batch = [texts[i] for i in order[step * BATCH:(step + 1) * BATCH]]
            input_ids, attn, labels, weights = make_batch(batch, tok, mask_id, vocab_size)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            loss = weighted_mlm_loss(logits, labels, weights)
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        print(f"  epoch {epoch+1}/{EPOCHS}  loss={total_loss/steps_per_epoch:.3f}", flush=True)

    model.save_pretrained(str(OUTDIR))
    tok.save_pretrained(str(OUTDIR))
    print(f"  saved -> {OUTDIR}\n", flush=True)


finetune()
print("ALL DONE")
