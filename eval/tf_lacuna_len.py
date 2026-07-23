"""Length-conditioned REAL-lacuna eval on etcbc/dss.

A real lacuna is a contiguous run of N reconstructed words. We blank the whole
run (model knows it's N words), then restore each gap word with the rest of the
gap still masked, bucketed by gap length N. Shows decay with length and whether
DSS-finetuning helps more on the hard, long gaps. Non-biblical only.

MsBERT is whole-word (1 token/word) so N-word gap = N masks, exact. BEREL is
subword: we mask each gap word's true subtoken count (optimistic for BEREL — it
leaks subtoken length; MsBERT gets only the word-count signal).
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
from tf.fabric import Fabric
tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils import morph_dss
from utils.book_filters import resolve_book_exclusions
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

MODELS = [("dicta-il/MsBERT", "MsBERT base"), ("dicta-il/BEREL", "BEREL base")]
for d, nice in [("ft_msbert", "MsBERT ft-scatter"), ("ft_berel", "BEREL ft-scatter"),
                ("ft_msbert_span", "MsBERT ft-SPAN"), ("ft_berel_span", "BEREL ft-SPAN"),
                ("ft_msbert_span_noparticles", "MsBERT ft-SPAN-no-particles"),
                ("ft_msbert_span_refined", "MsBERT ft-SPAN-refined"),
                ("ft_msbert_span_softshort", "MsBERT ft-SPAN-softshort")]:
    model_dir = repo_path(d)
    if model_dir.is_dir():
        MODELS.append((str(model_dir), nice))
for spec in filter(None, os.environ.get("EXTRA_MODELS", "").split(",")):
    dirname, label = spec.split(":", 1)
    model_dir = repo_path(dirname)
    if model_dir.is_dir():
        MODELS.append((str(model_dir), label))
MODEL_FILTERS = [part.strip().lower() for part in os.environ.get("MODEL_FILTER", "").split(",") if part.strip()]

WINDOW = 40
MIN_PRESERVED = 6
PER_BUCKET = int(os.environ.get("PER_BUCKET", "140"))          # <=0 means use all spans
TOPN, BEAM, K = 50, 50, 20
SPLIT_MODE = os.environ.get("EVAL_SCROLL_SPLIT", "all")
BOOK_FILTER_MODE = os.environ.get("BOOK_FILTER_MODE", "all")
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
rng = np.random.default_rng(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
allowed_scrolls, split_label = resolve_scroll_filter(SPLIT_MODE)
excluded_books, book_filter_label = resolve_book_exclusions(BOOK_FILTER_MODE)

if MODEL_FILTERS:
    MODELS = [
        (repo, label)
        for repo, label in MODELS
        if any(token in label.lower() or token in repo.lower() for token in MODEL_FILTERS)
    ]

def norm(w):                                   # scholar-lenient: finals, matres, divine name
    lem = morph_dss.lemma(w)
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


def heb(g):
    return len(g) >= 2 and all(ch in HEB for ch in g)


def bucket(n):
    return "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4-5" if n <= 5 else "6+"


TF_DIR = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
TF = Fabric(locations=str(TF_DIR), silent="deep")
api = TF.load("otype glyph rec biblical scroll", silent="deep")
if api is False:
    raise RuntimeError(f"Could not load cached DSS corpus from {TF_DIR}")
F, L = api.F, api.L


def winfo(w):
    signs = L.d(w, "sign")
    g = "".join(F.glyph.v(s) or "" for s in signs)
    recs = [F.rec.v(s) for s in signs]
    return g, (bool(signs) and all(r == 1 for r in recs)), (bool(signs) and all(r != 1 for r in recs))


EVAL_BIBLICAL = os.environ.get("EVAL_BIBLICAL", "").strip() == "1"

scrolls = {}
for w in F.otype.s("word"):
    is_bib = bool(F.biblical.v(w))
    if EVAL_BIBLICAL:
        if not is_bib:
            continue
    else:
        if is_bib:
            continue
    sc = L.u(w, "scroll")
    if not sc:
        continue
    scroll_name = F.scroll.v(sc[0])
    if not EVAL_BIBLICAL:
        if allowed_scrolls is not None and scroll_name not in allowed_scrolls:
            continue
        if scroll_name in excluded_books:
            continue
    scrolls.setdefault(sc[0], []).append(winfo(w))

# items: (ctx_words, gap_ctx_positions[list], golds[list], N)
items = []
for ws in scrolls.values():
    i = 0
    while i < len(ws):
        if ws[i][1]:                                  # start of a reconstructed run
            j = i
            while j < len(ws) and ws[j][1]:
                j += 1
            gap_targets = [k for k in range(i, j) if heb(ws[k][0])]
            N = len(gap_targets)
            if 1 <= N <= 12:
                lo, hi = max(0, i - WINDOW), min(len(ws), j + WINDOW)
                ctx, gap_pos, golds, preserved = [], [], [], 0
                for k in range(lo, hi):
                    g, fr, pr = ws[k]
                    if i <= k < j:
                        if k in gap_targets:
                            gap_pos.append(len(ctx)); golds.append(g); ctx.append(g)
                    elif heb(g):
                        ctx.append(g)
                        if pr:
                            preserved += 1
                if preserved >= MIN_PRESERVED and gap_pos:
                    items.append((ctx, gap_pos, golds, N))
            i = j
        else:
            i += 1

# balance buckets
by_b = {}
for it in items:
    by_b.setdefault(bucket(it[3]), []).append(it)
print(f"eval split: {'biblical-scrolls' if EVAL_BIBLICAL else split_label}")
print(f"book filter: {book_filter_label}")
print(f"eligible scrolls: {len(scrolls)}")
print("gap-length buckets (spans found):", {b: len(v) for b, v in sorted(by_b.items())})
sample = []
for b, v in by_b.items():
    take = len(v) if PER_BUCKET <= 0 else min(len(v), PER_BUCKET)
    idx = rng.choice(len(v), size=take, replace=False)
    sample += [v[i] for i in idx]
print(f"eval spans: {len(sample)}\n")


def beam_words(logits, ps, tok):
    beams = [(0.0, [])]
    for p in ps:
        lp = torch.log_softmax(logits[p], -1)
        top = torch.topk(lp, TOPN)
        beams = sorted([(s + v, seq + [i]) for s, seq in beams
                        for i, v in zip(top.indices.tolist(), top.values.tolist())],
                       key=lambda x: -x[0])[:BEAM]
    out = []
    for _, seq in beams:
        w = tok.decode(seq).replace(" ", "").replace("##", "")
        if w not in out:
            out.append(w)
        if len(out) >= K:
            break
    return out


def beam_autoregressive(model, input_ids, gp, tok, topn=50, beam_width=5):
    MASK = tok.mask_token_id
    beams = [(0.0, input_ids.clone().to(dev), [])]
    
    for slot_idx, (ps, _) in enumerate(gp):
        new_beams = []
        for score, current_ids, pred_words in beams:
            with torch.no_grad():
                logits = model(current_ids.unsqueeze(0)).logits[0].cpu()
                
            slot_beams = [(0.0, [])]
            for p in ps:
                lp = torch.log_softmax(logits[p], -1)
                top = torch.topk(lp, topn)
                slot_beams = sorted(
                    [(s + v, seq + [i]) for s, seq in slot_beams
                     for i, v in zip(top.indices.tolist(), top.values.tolist())],
                    key=lambda x: -x[0]
                )[:beam_width]
                
            for slot_score, seq in slot_beams:
                word = tok.decode(seq).replace(" ", "").replace("##", "")
                
                new_ids = current_ids.clone()
                for i, p in enumerate(ps):
                    new_ids[p] = seq[i]
                    
                new_beams.append((score + slot_score, new_ids, pred_words + [word]))
                
        beams = sorted(new_beams, key=lambda x: -x[0])[:beam_width]
        
    out = []
    for _, _, pred_words in beams:
        if pred_words not in out:
            out.append(pred_words)
    return out


def eval_model(repo, nice):
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).to(dev).eval()
    MASK = tok.mask_token_id
    
    slot_cells = {}   # bucket -> [t1,t5,t10,t20,n_slots]
    seq_cells = {}    # bucket -> [t1,t5,t10,t20,n_seqs]
    
    for ctx, gap_pos, golds, N in sample:
        enc = tok(ctx, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=512)
        wmap = {}
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None:
                wmap.setdefault(wid, []).append(pos)
        gp = [(wmap.get(g), gold) for g, gold in zip(gap_pos, golds)]
        if any(ps is None for ps, _ in gp):
            continue
        ids = enc["input_ids"][0].clone()
        for ps, _ in gp:
            for p in ps:
                ids[p] = MASK
        with torch.no_grad():
            logits = model(ids.unsqueeze(0).to(dev)).logits[0].cpu()
            
        b = bucket(N)
        sc = slot_cells.setdefault(b, [0, 0, 0, 0, 0])
        sq = seq_cells.setdefault(b, [0, 0, 0, 0, 0])
        
        # 1. Evaluate individual slots
        for ps, gold in gp:
            ranked = beam_words(logits, ps, tok)
            ng = norm(gold)
            rank = next((i for i, r in enumerate(ranked) if norm(r) == ng), 999)
            sc[0] += rank == 0
            sc[1] += rank < 5
            sc[2] += rank < 10
            sc[3] += rank < 20
            sc[4] += 1
            
        # 2. Evaluate entire sequence (רצף)
        ranked_seqs = beam_autoregressive(model, ids, gp, tok, beam_width=5)
        gold_seq = [norm(g) for _, g in gp]
        
        rank_seq = 999
        for i, cand_seq in enumerate(ranked_seqs):
            if len(cand_seq) == len(gold_seq) and all(norm(c) == g for c, g in zip(cand_seq, gold_seq)):
                rank_seq = i
                break
                
        sq[0] += rank_seq == 0
        sq[1] += rank_seq < 5
        sq[2] += rank_seq < 10
        sq[3] += rank_seq < 20
        sq[4] += 1
        
    return nice, slot_cells, seq_cells


order = ["1", "2", "3", "4-5", "6+"]
res = [eval_model(r, n) for r, n in MODELS]

# Print Slot-Level Accuracy Table
print("==================================================")
print("=== SLOT-LEVEL ACCURACY (Individual Words) ===")
print("==================================================")
for metric, mi in [("top-1", 0), ("top-5", 1), ("top-10", 2), ("top-20", 3)]:
    print(f"--- {metric} ---")
    print(f"{'model':16s}" + "".join(f"{b:>9s}" for b in order))
    for nice, slot_cells, _ in res:
        row = f"{nice:16s}"
        for b in order:
            c = slot_cells.get(b)
            row += f"{(c[mi]/c[4]*100 if c and c[4] else 0):8.1f}%" if c else f"{'-':>9s}"
        print(row)
    print()

# Print Sequence-Level Accuracy Table (רצף)
print("==================================================")
print("=== SEQUENCE-LEVEL ACCURACY (Phrase Matching) ===")
print("==================================================")
for metric, mi in [("top-1", 0), ("top-5", 1), ("top-10", 2), ("top-20", 3)]:
    print(f"--- {metric} ---")
    print(f"{'model':16s}" + "".join(f"{b:>9s}" for b in order))
    for nice, _, seq_cells in res:
        row = f"{nice:16s}"
        for b in order:
            c = seq_cells.get(b)
            row += f"{(c[mi]/c[4]*100 if c and c[4] else 0):8.1f}%" if c else f"{'-':>9s}"
        print(row)
    print()

ns_slot = res[0][1]
ns_seq = res[0][2]
print("n slots per bucket:", {b: (ns_slot[b][4] if b in ns_slot else 0) for b in order})
print("n sequences per bucket:", {b: (ns_seq[b][4] if b in ns_seq else 0) for b in order})
