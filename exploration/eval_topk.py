"""Top-1 / top-5 / top-10 whole-word restoration on the held-out DSS test set.

Restoration tools give scholars a RANKED short-list, so top-k is the honest
metric. For multi-subtoken words we beam-combine per-position predictions into
ranked whole-word candidates (positions are independent given all are masked,
so the joint is the product of per-position probabilities).

Evaluates any base models plus local finetuned dirs (./ft_*) if present, on the
IDENTICAL masked words. Same held-out test set the finetuner never trains on.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.dss_split import load_split, SECT
from utils.paths import repo_path
tlog.set_verbosity_error()

MODELS = [
    ("dicta-il/BEREL", "BEREL (base)"),
    ("dicta-il/MsBERT", "MsBERT (base)"),
]
for d, nice in [("ft_berel", "BEREL + DSS-ft (ours)"), ("ft_msbert", "MsBERT + DSS-ft (ours)")]:
    model_dir = repo_path(d)
    if model_dir.is_dir():
        MODELS.append((str(model_dir), nice))

N_TEST = 220         # test chunks sampled for eval
WORDS_PER_CHUNK = 3
MIN_WORDLEN = 2
TOPN = 20            # per-position candidates fed to the beam
BEAM = 25            # beam width
K = 10               # report up to top-10
rng = np.random.default_rng(1)

train, test, bib = load_split()
sel = rng.choice(len(test), size=min(N_TEST, len(test)), replace=False)
test = [test[i] for i in sel]

# fixed masking decisions, shared across all models
tasks = []  # (text, words, mask_indices, group)
for r, grp in [(x, "sect" if SECT.get(x["section"]) == "sect" else "nonbib_other") for x in test] + \
              [(x, "bib") for x in bib]:
    words = r["text"].strip().split()
    cand = [j for j, w in enumerate(words) if len(w) >= MIN_WORDLEN]
    if not cand:
        continue
    pick = sorted(rng.choice(cand, size=min(WORDS_PER_CHUNK, len(cand)), replace=False).tolist())
    tasks.append((r["text"].strip(), words, pick, grp))
print(f"{len(tasks)} test chunks, {sum(len(t[2]) for t in tasks)} masked words "
      f"(fixed across models)\n")


def beam_words(logits, ps, tok):
    """ranked unique whole-word candidates for masked positions ps."""
    beams = [(0.0, [])]
    for p in ps:
        lp = torch.log_softmax(logits[p], -1)
        top = torch.topk(lp, TOPN)
        ids, vals = top.indices.tolist(), top.values.tolist()
        beams = sorted(
            [(s + v, seq + [i]) for s, seq in beams for i, v in zip(ids, vals)],
            key=lambda x: -x[0])[:BEAM]
    out = []
    for _, seq in beams:
        w = tok.decode(seq).replace(" ", "").replace("##", "")
        if w not in out:
            out.append(w)
        if len(out) >= K:
            break
    return out


def eval_model(repo, nice):
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).eval()
    MASK = tok.mask_token_id
    # per group: [top1, top5, top10, n]
    cell = {g: [0, 0, 0, 0] for g in ["sect", "nonbib_other", "bib"]}
    for text, words, pick, grp in tasks:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        ids = enc["input_ids"][0]
        wmap = {}
        for pos, w in enumerate(enc.word_ids(0)):
            if w is not None:
                wmap.setdefault(w, []).append(pos)
        for wi in pick:
            ps = wmap.get(wi)
            if not ps:
                continue
            masked = ids.clone()
            for p in ps:
                masked[p] = MASK
            with torch.no_grad():
                logits = model(masked.unsqueeze(0)).logits[0]
            ranked = beam_words(logits, ps, tok)
            gold = words[wi]
            rank = ranked.index(gold) if gold in ranked else 999
            c = cell[grp]
            c[0] += int(rank == 0); c[1] += int(rank < 5); c[2] += int(rank < 10); c[3] += 1
    return dict(nice=nice, cell=cell)


def nonbib(cell):
    """combine sect + nonbib_other into a single non-biblical figure."""
    s, o = cell["sect"], cell["nonbib_other"]
    return [s[i] + o[i] for i in range(4)]


results = []
for repo, nice in MODELS:
    print(f"loading {repo} ...", flush=True)
    try:
        results.append(eval_model(repo, nice))
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:120]}")


def fmt(c):
    n = max(c[3], 1)
    return f"{c[0]/n*100:5.1f} /{c[1]/n*100:5.1f} /{c[2]/n*100:5.1f}"


print(f"\n{'':26s}{'NON-BIBLICAL':>22s}{'SECTARIAN only':>22s}{'biblical(recall)':>22s}")
print(f"{'model':26s}{'t1 / t5 / t10':>22s}{'t1 / t5 / t10':>22s}{'t1 / t5 / t10':>22s}")
for r in results:
    print(f"{r['nice']:26s}{fmt(nonbib(r['cell'])):>22s}"
          f"{fmt(r['cell']['sect']):>22s}{fmt(r['cell']['bib']):>22s}")
print(f"\n(% word-exact @ top-1/5/10; sectarian n={results[0]['cell']['sect'][3] if results else 0})")
