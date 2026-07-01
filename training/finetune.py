"""Finetune a Hebrew masked-LM on non-biblical DSS text (whole-word masking),
to test whether a DSS-specialized model beats the ~17%/31%@10 wall.

Book-level train split (shared with the evaluator) — never touches test scrolls.
Whole-word masking so training matches the whole-word restoration metric.
Saves to ./ft_berel and ./ft_msbert; the evaluator auto-picks them up.
"""
import os, sys, math
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.dss_split import load_split
from utils.paths import repo_path
tlog.set_verbosity_error()

BASES = [("dicta-il/BEREL", repo_path("ft_berel")), ("dicta-il/MsBERT", repo_path("ft_msbert"))]
MAX_LEN = 160
EPOCHS = 4
BATCH = 16
LR = 3e-5
MLM_PROB = 0.15
rng = np.random.default_rng(0)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
train, _, _ = load_split()
texts = [r["text"].strip() for r in train]
print(f"device={dev} | train chunks={len(texts)} | ~{sum(len(t.split()) for t in texts):,} words\n")


def encode_all(tok):
    out = []
    for t in texts:
        e = tok(t, truncation=True, max_length=MAX_LEN)
        wids = e.word_ids()
        out.append((e["input_ids"], wids))
    return out


def make_batch(batch, tok, MASK, vocab):
    L = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    for bi, (ids, wids) in enumerate(batch):
        input_ids[bi, :len(ids)] = torch.tensor(ids)
        attn[bi, :len(ids)] = 1
        groups = {}
        for pos, w in enumerate(wids):
            if w is not None:
                groups.setdefault(w, []).append(pos)
        words = list(groups)
        if not words:
            continue
        n = max(1, int(round(len(words) * MLM_PROB)))
        for w in rng.choice(words, size=min(n, len(words)), replace=False):
            for pos in groups[w]:
                labels[bi, pos] = ids[pos]
                r = rng.random()
                if r < 0.8:
                    input_ids[bi, pos] = MASK
                elif r < 0.9:
                    input_ids[bi, pos] = int(rng.integers(vocab))
    return input_ids.to(dev), attn.to(dev), labels.to(dev)


def finetune(repo, outdir):
    print(f"=== {repo} -> {outdir} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).to(dev).train()
    MASK, vocab = tok.mask_token_id, model.config.vocab_size
    data = encode_all(tok)
    opt = AdamW(model.parameters(), lr=LR)
    steps_per_epoch = math.ceil(len(data) / BATCH)
    for ep in range(EPOCHS):
        order = rng.permutation(len(data))
        tot = 0.0
        for si in range(steps_per_epoch):
            idx = order[si * BATCH:(si + 1) * BATCH]
            batch = [data[i] for i in idx]
            input_ids, attn, labels = make_batch(batch, tok, MASK, vocab)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            tot += out.loss.item()
        print(f"  epoch {ep+1}/{EPOCHS}  loss={tot/steps_per_epoch:.3f}", flush=True)
    model.save_pretrained(str(outdir))
    tok.save_pretrained(str(outdir))
    print(f"  saved -> {outdir}\n", flush=True)


for repo, outdir in BASES:
    finetune(repo, outdir)
print("ALL DONE")
