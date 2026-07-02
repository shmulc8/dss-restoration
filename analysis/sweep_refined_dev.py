"""Small in-memory sweep for the refined MsBERT span-ft objective.

This avoids writing a full checkpoint for every candidate configuration.
Each run:
1. Continues training from ft_msbert_span on a chosen book partition.
2. Evaluates on a chosen scroll split using the single-word lacuna probe.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tf.fabric import Fabric
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import morph_dss
from utils.dss_split import load_partition
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

tlog.set_verbosity_error()

BASE_REPO = str(repo_path("ft_msbert_span"))
MAX_LEN, BATCH = 160, 16
MASK_FRAC, SPAN_P, SPAN_MAX = 0.15, 0.3, 10
MIN_TARGET_LEN = 2
WINDOW, MIN_PRESERVED, MAX_ITEMS = 20, 8, 300
TOPN, BEAM, K = 50, 50, 20
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
FUNCTION = {
    "אשר", "כי", "כיא", "את", "אל", "על", "אם", "לא", "לוא", "כל", "כול",
    "מן", "הוא", "היא", "אני", "אתה", "הם", "זה", "זאת", "לו", "בו", "עד",
    "גם", "או", "כן", "אך", "רק", "יש", "אין", "מה", "מי", "ולא", "ואת", "וכל", "כה",
}
CONFIGS = [
    ("e1_lr2e5", 1, 2e-5),
    ("e2_lr1e5", 2, 1e-5),
    ("e2_lr2e5", 2, 2e-5),
    ("e3_lr2e5", 3, 2e-5),
]
ONLY_CONFIG = os.environ.get("ONLY_CONFIG")
TRAIN_PARTITION = os.environ.get("TRAIN_PARTITION", "train")
EVAL_SCROLL_SPLIT = os.environ.get("EVAL_SCROLL_SPLIT", "dev")

dev = "mps" if torch.backends.mps.is_available() else "cpu"
rng = np.random.default_rng(0)
train_rows = load_partition(TRAIN_PARTITION)
train_texts = [row["text"].strip() for row in train_rows]


def norm(word: str) -> str:
    lem = morph_dss.lemma(word)
    lem = "".join(FINAL.get(c, c) for c in lem)
    if lem in DIVINE:
        return "יהוה"
    if lem in {"כיא", "כי"}:
        return "כי"
    if lem in {"לוא", "לא"}:
        return "לא"
    if lem in {"כול", "כל"}:
        return "כל"
    return lem


def is_content(word: str) -> bool:
    return len(word) >= 3 and word not in FUNCTION


def span_words(nwords, local_rng):
    target = max(1, round(nwords * MASK_FRAC))
    chosen, tries = set(), 0
    while len(chosen) < target and tries < 50:
        tries += 1
        span_len = min(int(local_rng.geometric(SPAN_P)), SPAN_MAX, nwords)
        start = int(local_rng.integers(0, max(1, nwords - span_len + 1)))
        span = set(range(start, min(start + span_len, nwords)))
        if span & chosen:
            continue
        chosen |= span
    return chosen


def make_batch(batch_texts, tok, mask_id, vocab_size, local_rng):
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

        chosen = span_words(len(eligible), local_rng)
        chosen_word_ids = [eligible[idx] for idx in chosen]
        for word_id in chosen_word_ids:
            for pos in groups[word_id]:
                labels[batch_idx, pos] = ids[pos]
                draw = local_rng.random()
                if draw < 0.8:
                    input_ids[batch_idx, pos] = mask_id
                elif draw < 0.9:
                    input_ids[batch_idx, pos] = int(local_rng.integers(vocab_size))

    return input_ids.to(dev), attn.to(dev), labels.to(dev)


TF_DIR = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
TF = Fabric(locations=str(TF_DIR), silent="deep")
api = TF.load("otype glyph rec biblical scroll", silent="deep")
if api is False:
    raise RuntimeError(f"Could not load cached DSS corpus from {TF_DIR}")
F, L = api.F, api.L
allowed_scrolls, split_label = resolve_scroll_filter(EVAL_SCROLL_SPLIT)


def winfo(word_node):
    signs = L.d(word_node, "sign")
    glyph = "".join(F.glyph.v(sign) or "" for sign in signs)
    recs = [F.rec.v(sign) for sign in signs]
    return glyph, (bool(signs) and all(r == 1 for r in recs)), (bool(signs) and all(r != 1 for r in recs))


scrolls = {}
for word_node in F.otype.s("word"):
    if F.biblical.v(word_node):
        continue
    scroll = L.u(word_node, "scroll")
    if not scroll:
        continue
    scroll_name = F.scroll.v(scroll[0])
    if allowed_scrolls is not None and scroll_name not in allowed_scrolls:
        continue
    scrolls.setdefault(scroll[0], []).append(winfo(word_node))

items = []
for words in scrolls.values():
    for idx, (glyph, fully_rec, _) in enumerate(words):
        if not fully_rec or len(glyph) < 2 or any(ch not in HEB for ch in glyph):
            continue
        lo, hi = max(0, idx - WINDOW), min(len(words), idx + WINDOW + 1)
        ctx, tpos, preserved = [], None, 0
        for pos in range(lo, hi):
            cur, _, is_preserved = words[pos]
            if len(cur) >= 1 and all(ch in HEB for ch in cur):
                if pos == idx:
                    tpos = len(ctx)
                ctx.append(cur)
                if is_preserved and pos != idx:
                    preserved += 1
        if tpos is not None and preserved >= MIN_PRESERVED:
            items.append((ctx, tpos, ctx[tpos]))

sel = rng.choice(len(items), size=min(MAX_ITEMS, len(items)), replace=False)
items = [items[i] for i in sel]
morph_dss.lemmas([gold for _, _, gold in items])


def beam_words(logits, positions, tok):
    beams = [(0.0, [])]
    for pos in positions:
        logp = torch.log_softmax(logits[pos], -1)
        top = torch.topk(logp, TOPN)
        beams = sorted(
            [
                (score + delta, seq + [tok_id])
                for score, seq in beams
                for tok_id, delta in zip(top.indices.tolist(), top.values.tolist())
            ],
            key=lambda item: -item[0],
        )[:BEAM]
    out = []
    for _, seq in beams:
        word = tok.decode(seq).replace(" ", "").replace("##", "")
        if word not in out:
            out.append(word)
        if len(out) >= K:
            break
    return out


def eval_model(model, tok):
    results = []
    short_top1 = 0
    for ctx, tpos, gold in items:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        word_map = {}
        for pos, word_id in enumerate(enc.word_ids(0)):
            if word_id is not None:
                word_map.setdefault(word_id, []).append(pos)
        positions = word_map.get(tpos)
        if not positions:
            results.append((gold, []))
            continue
        ids = enc["input_ids"][0].clone()
        for pos in positions:
            ids[pos] = tok.mask_token_id
        with torch.no_grad():
            logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()
        ranked = beam_words(logits, positions, tok)
        if ranked and len(ranked[0]) < len(gold) - 1:
            short_top1 += 1
        results.append((gold, ranked))

    n = len(results)
    exact_1 = sum(1 for gold, ranked in results if ranked and ranked[0] == gold) / n * 100
    exact_10 = sum(1 for gold, ranked in results if gold in ranked[:10]) / n * 100
    norm_1 = sum(1 for gold, ranked in results if ranked and norm(ranked[0]) == norm(gold)) / n * 100
    norm_10 = sum(1 for gold, ranked in results if any(norm(word) == norm(gold) for word in ranked[:10])) / n * 100

    content = [(gold, ranked) for gold, ranked in results if is_content(gold)]
    nc = max(1, len(content))
    cont_exact_1 = sum(1 for gold, ranked in content if ranked and ranked[0] == gold) / nc * 100
    cont_norm_10 = sum(1 for gold, ranked in content if any(norm(word) == norm(gold) for word in ranked[:10])) / nc * 100
    return exact_1, exact_10, norm_1, norm_10, cont_exact_1, cont_norm_10, short_top1, n


print(
    f"device={dev} | refined objective sweep | train_partition={TRAIN_PARTITION} "
    f"| train chunks={len(train_texts)} "
    f"| {split_label} sampled items={len(items)}"
)
print("name           | EXACT t1 t10 | NORM t1 t10 | content exact t1 norm t10 | short top1")

configs = [cfg for cfg in CONFIGS if ONLY_CONFIG in {None, "", cfg[0]}]
if not configs:
    raise ValueError(f"ONLY_CONFIG={ONLY_CONFIG!r} did not match any config names")

for offset, (name, epochs, lr) in enumerate(configs):
    local_rng = np.random.default_rng(100 + offset)
    tok = AutoTokenizer.from_pretrained(BASE_REPO, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(BASE_REPO).to(dev).train()
    mask_id, vocab_size = tok.mask_token_id, model.config.vocab_size
    opt = AdamW(model.parameters(), lr=lr)
    steps_per_epoch = math.ceil(len(train_texts) / BATCH)

    for _ in range(epochs):
        order = local_rng.permutation(len(train_texts))
        for step in range(steps_per_epoch):
            batch = [train_texts[i] for i in order[step * BATCH:(step + 1) * BATCH]]
            input_ids, attn, labels = make_batch(batch, tok, mask_id, vocab_size, local_rng)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()

    model.eval()
    exact_1, exact_10, norm_1, norm_10, cont_exact_1, cont_norm_10, short_top1, n = eval_model(model, tok)
    print(
        f"{name:14s} | {exact_1:5.1f}% {exact_10:5.1f}% | {norm_1:5.1f}% {norm_10:5.1f}% | "
        f"{cont_exact_1:5.1f}% {cont_norm_10:5.1f}% | {short_top1:3d}/{n}"
    )
    del model
    if dev == "mps":
        torch.mps.empty_cache()
