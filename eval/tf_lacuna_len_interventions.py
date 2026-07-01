"""Gap-length evaluation with two cheap interventions.

Compares:
1. Baseline ranking.
2. Post-hoc reranking that downweights particle-like / very short candidates.
3. Two context windows (20 and 40 words each side).

This keeps the original gap-length setup but focuses on the main two models:
MsBERT base and MsBERT+span-ft.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from tf.fabric import Fabric
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as tlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import morph_dss
from utils.paths import repo_path

tlog.set_verbosity_error()

MODELS = [("dicta-il/MsBERT", "MsBERT base")]
model_dir = repo_path("ft_msbert_span")
if model_dir.is_dir():
    MODELS.append((str(model_dir), "MsBERT ft-SPAN"))

WINDOWS = [20, 40]
MIN_PRESERVED = 6
PER_BUCKET = 140
TOPN, BEAM, K = 50, 50, 20
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
FUNCTION = {
    "אשר", "כי", "כיא", "את", "אל", "על", "אם", "לא", "לוא", "כל", "כול",
    "מן", "הוא", "היא", "אני", "אתה", "הם", "זה", "זאת", "לו", "בו", "עד",
    "גם", "או", "כן", "אך", "רק", "יש", "אין", "מה", "מי", "ולא", "ואת", "וכל", "כה",
}
PROCLITIC_LETTERS = {"ו", "ל", "ב", "כ", "ה", "מ", "ש"}
rng = np.random.default_rng(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"


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


def heb(word: str) -> bool:
    return len(word) >= 2 and all(ch in HEB for ch in word)


def bucket(n: int) -> str:
    return "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4-5" if n <= 5 else "6+"


def rerank_score(word: str, base_score: float) -> float:
    penalty = 0.0
    if len(word) == 1:
        penalty += 2.0
    elif len(word) == 2:
        penalty += 0.35
    if word in FUNCTION:
        penalty += 0.75
    if len(word) <= 2 and word[:1] in PROCLITIC_LETTERS:
        penalty += 0.5
    return base_score - penalty


def beam_candidates(logits, positions, tok):
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

    best_by_word = {}
    for score, seq in beams:
        word = tok.decode(seq).replace(" ", "").replace("##", "")
        if not word:
            continue
        prev = best_by_word.get(word)
        if prev is None or score > prev:
            best_by_word[word] = score

    ranked = sorted(best_by_word.items(), key=lambda item: -item[1])[:K]
    reranked = sorted(best_by_word.items(), key=lambda item: -rerank_score(item[0], item[1]))[:K]
    return [word for word, _ in ranked], [word for word, _ in reranked]


TF_DIR = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
TF = Fabric(locations=str(TF_DIR), silent="deep")
api = TF.load("otype glyph rec biblical scroll", silent="deep")
if api is False:
    raise RuntimeError(f"Could not load cached DSS corpus from {TF_DIR}")
F, L = api.F, api.L


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
    if scroll:
        scrolls.setdefault(scroll[0], []).append(winfo(word_node))


def collect_items(window: int):
    items = []
    for words in scrolls.values():
        idx = 0
        while idx < len(words):
            if words[idx][1]:
                end = idx
                while end < len(words) and words[end][1]:
                    end += 1
                gap_targets = [pos for pos in range(idx, end) if heb(words[pos][0])]
                gap_len = len(gap_targets)
                if 1 <= gap_len <= 12:
                    lo, hi = max(0, idx - window), min(len(words), end + window)
                    ctx, gap_pos, golds, preserved = [], [], [], 0
                    for pos in range(lo, hi):
                        glyph, _, is_preserved = words[pos]
                        if idx <= pos < end:
                            if pos in gap_targets:
                                gap_pos.append(len(ctx))
                                golds.append(glyph)
                                ctx.append(glyph)
                        elif heb(glyph):
                            ctx.append(glyph)
                            if is_preserved:
                                preserved += 1
                    if preserved >= MIN_PRESERVED and gap_pos:
                        items.append((ctx, gap_pos, golds, gap_len))
                idx = end
            else:
                idx += 1
    return items


def balanced_sample(items):
    by_bucket = {}
    for item in items:
        by_bucket.setdefault(bucket(item[3]), []).append(item)
    sample = []
    counts = {}
    for name, rows in by_bucket.items():
        pick = rng.choice(len(rows), size=min(len(rows), PER_BUCKET), replace=False)
        chosen = [rows[i] for i in pick]
        sample.extend(chosen)
        counts[name] = len(chosen)
    return sample, counts


order = ["1", "2", "3", "4-5", "6+"]
all_samples = {}
for window in WINDOWS:
    items = collect_items(window)
    sample, counts = balanced_sample(items)
    all_samples[window] = sample
    print(f"window={window}: spans found={len(items)} eval spans={len(sample)} counts={counts}")
print()


def evaluate_model(repo: str, label: str, sample):
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).to(dev).eval()
    mask_id = tok.mask_token_id
    base_cells = {}
    rerank_cells = {}

    for ctx, gap_pos, golds, gap_len in sample:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        word_map = {}
        for pos, word_id in enumerate(enc.word_ids(0)):
            if word_id is not None:
                word_map.setdefault(word_id, []).append(pos)
        gap_positions = [(word_map.get(target), gold) for target, gold in zip(gap_pos, golds)]
        if any(pos_list is None for pos_list, _ in gap_positions):
            continue

        ids = enc["input_ids"][0].clone()
        for pos_list, _ in gap_positions:
            for pos in pos_list:
                ids[pos] = mask_id
        with torch.no_grad():
            logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()

        b = bucket(gap_len)
        base_cell = base_cells.setdefault(b, [0, 0, 0, 0, 0])
        rerank_cell = rerank_cells.setdefault(b, [0, 0, 0, 0, 0])
        for pos_list, gold in gap_positions:
            ranked, reranked = beam_candidates(logits, pos_list, tok)
            gold_norm = norm(gold)
            base_rank = next((i for i, word in enumerate(ranked) if norm(word) == gold_norm), 999)
            rerank_rank = next((i for i, word in enumerate(reranked) if norm(word) == gold_norm), 999)
            base_cell[0] += base_rank == 0
            base_cell[1] += base_rank < 5
            base_cell[2] += base_rank < 10
            base_cell[3] += base_rank < 20
            base_cell[4] += 1
            rerank_cell[0] += rerank_rank == 0
            rerank_cell[1] += rerank_rank < 5
            rerank_cell[2] += rerank_rank < 10
            rerank_cell[3] += rerank_rank < 20
            rerank_cell[4] += 1

    return label, base_cells, rerank_cells


for window in WINDOWS:
    print(f"=== WINDOW {window} ===")
    results = [evaluate_model(repo, label, all_samples[window]) for repo, label in MODELS]
    for metric, idx in [("top-1", 0), ("top-5", 1), ("top-10", 2), ("top-20", 3)]:
        print(f"--- baseline {metric} ---")
        print(f"{'model':16s}" + "".join(f"{b:>9s}" for b in order))
        for label, base_cells, _ in results:
            row = f"{label:16s}"
            for b in order:
                c = base_cells.get(b)
                row += f"{(c[idx]/c[4]*100 if c and c[4] else 0):8.1f}%" if c else f"{'-':>9s}"
            print(row)
        print()

        print(f"--- reranked {metric} ---")
        print(f"{'model':16s}" + "".join(f"{b:>9s}" for b in order))
        for label, _, rerank_cells in results:
            row = f"{label:16s}"
            for b in order:
                c = rerank_cells.get(b)
                row += f"{(c[idx]/c[4]*100 if c and c[4] else 0):8.1f}%" if c else f"{'-':>9s}"
            print(row)
        print()
