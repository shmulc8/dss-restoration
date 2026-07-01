"""Like finetune.py but with CONTIGUOUS SPAN masking (whole multi-word runs),
matching real lacunae, instead of scattered 15% single-word masks. Saves to
./ft_berel_span and ./ft_msbert_span."""
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

BASES = [("dicta-il/BEREL", repo_path("ft_berel_span")), ("dicta-il/MsBERT", repo_path("ft_msbert_span"))]
MAX_LEN, EPOCHS, BATCH, LR = 160, 4, 16, 3e-5
MASK_FRAC, SPAN_P, SPAN_MAX = 0.15, 0.3, 10   # geometric mean ~3.3 words, clip 10
rng = np.random.default_rng(0)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
train, _, _ = load_split()
texts = [r["text"].strip() for r in train]
print(f"device={dev} | span-mask | train chunks={len(texts)}\n")


def encode_all(tok):
    return [(e["input_ids"], e.word_ids())
            for e in (tok(t, truncation=True, max_length=MAX_LEN) for t in texts)]


def span_words(nwords):
    """set of word indices covered by non-overlapping contiguous spans (~MASK_FRAC)."""
    target = max(1, round(nwords * MASK_FRAC))
    chosen, tries = set(), 0
    while len(chosen) < target and tries < 50:
        tries += 1
        L = min(int(rng.geometric(SPAN_P)), SPAN_MAX, nwords)
        start = int(rng.integers(0, max(1, nwords - L + 1)))
        span = set(range(start, min(start + L, nwords)))
        if span & chosen:
            continue
        chosen |= span
    return chosen


def make_batch(batch, tok, MASK, vocab):
    Lmax = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), Lmax), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(batch), Lmax), dtype=torch.long)
    labels = torch.full((len(batch), Lmax), -100, dtype=torch.long)
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
        wset = list(span_words(len(words)))
        for wi in wset:
            for pos in groups[words[wi]]:
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
    spe = math.ceil(len(data) / BATCH)
    for ep in range(EPOCHS):
        order = rng.permutation(len(data))
        tot = 0.0
        for si in range(spe):
            batch = [data[i] for i in order[si * BATCH:(si + 1) * BATCH]]
            input_ids, attn, labels = make_batch(batch, tok, MASK, vocab)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward(); opt.step(); opt.zero_grad()
            tot += out.loss.item()
        print(f"  epoch {ep+1}/{EPOCHS}  loss={tot/spe:.3f}", flush=True)
    model.save_pretrained(str(outdir)); tok.save_pretrained(str(outdir))
    print(f"  saved -> {outdir}\n", flush=True)


for repo, outdir in BASES:
    finetune(repo, outdir)
print("ALL DONE")
