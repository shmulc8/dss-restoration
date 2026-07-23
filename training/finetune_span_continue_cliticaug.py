"""Continue span-ft training on original plus clitic-joined augmented texts."""
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
OUTDIR = repo_path(os.environ.get("OUTDIR_NAME", "ft_msbert_span_refined_cliticaug"))
MAX_LEN = 160
EPOCHS = int(os.environ.get("EPOCHS", "1"))
BATCH = int(os.environ.get("BATCH", "16"))
LR = float(os.environ.get("LR", "2e-5"))
MASK_FRAC, SPAN_P, SPAN_MAX = 0.15, 0.3, 10
TRAIN_PARTITION = os.environ.get("TRAIN_PARTITION", "train")
PREFIXES = os.environ.get("CLITIC_PREFIXES", "ובכלמשה")
MIN_NEXT_LEN = int(os.environ.get("MIN_NEXT_LEN", "3"))
MODE = os.environ.get("AUG_MODE", "append")  # append or replace
rng = np.random.default_rng(0)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
train = load_partition(TRAIN_PARTITION)
base_texts = [row["text"].strip() for row in train]
aug_texts = []
changed = 0
merge_total = 0
for text in base_texts:
    joined, merges = join_likely_clitics(text, prefixes=PREFIXES, min_next_len=MIN_NEXT_LEN)
    if merges:
        changed += 1
        merge_total += merges
        aug_texts.append(joined)

if MODE == "replace":
    texts = aug_texts or base_texts
else:
    texts = base_texts + aug_texts

print(
    f"device={dev} | span-ft-cliticaug | base={Path(BASE_REPO).name} "
    f"| partition={TRAIN_PARTITION} | epochs={EPOCHS} | lr={LR:g} "
    f"| mode={MODE} | prefixes={PREFIXES} | min_next_len={MIN_NEXT_LEN}\n"
)
print(
    f"base chunks={len(base_texts)} | changed chunks={changed} | augmented chunks={len(aug_texts)} "
    f"| total merges={merge_total} | train chunks used={len(texts)}\n"
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
