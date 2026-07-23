"""Evaluation of MsBERT models comparing Left-to-Right decoding vs Confidence-Based (Easy-First) decoding.

We test on the best-performing model so far: MsBERT + span-ft-refined, comparing it
against the MsBERT base model.
We evaluate both slot-level and sequence-level accuracy across gap lengths.
Results are saved to the Obsidian vault.
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

# We evaluate: MsBERT base and MsBERT ft-SPAN-refined (our best method so far)
MODELS = [
    ("dicta-il/MsBERT", "MsBERT base"),
]
refined_dir = repo_path("ft_msbert_span_refined")
if refined_dir.is_dir():
    MODELS.append((str(refined_dir), "MsBERT ft-SPAN-refined"))
else:
    # Try any other ft_msbert_span if refined is missing
    alt_dir = repo_path("ft_msbert_span")
    if alt_dir.is_dir():
        MODELS.append((str(alt_dir), "MsBERT ft-SPAN"))

WINDOW = 40
MIN_PRESERVED = 6
PER_BUCKET = int(os.environ.get("PER_BUCKET", "50"))
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

def norm(w):
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

items = []
for ws in scrolls.values():
    i = 0
    while i < len(ws):
        if ws[i][1]:
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

def beam_autoregressive_left_to_right(model, input_ids, gp, tok, topn=50, beam_width=5):
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

def beam_autoregressive_easy_first(model, input_ids, gp, tok, topn=50, beam_width=5):
    """Autoregressively decodes the easiest (highest confidence) slot first."""
    beams = [(0.0, input_ids.clone().to(dev), set(), [None] * len(gp))]
    
    for step in range(len(gp)):
        new_beams = []
        for score, current_ids, filled, pred_words in beams:
            if len(filled) == len(gp):
                new_beams.append((score, current_ids, filled, pred_words))
                continue
                
            with torch.no_grad():
                logits = model(current_ids.unsqueeze(0)).logits[0].cpu()
                
            # Find unfilled slot with highest confidence (average top-1 log prob of its subtokens)
            best_slot_idx = None
            best_confidence = -1e9
            
            for slot_idx, (ps, _) in enumerate(gp):
                if slot_idx in filled:
                    continue
                slot_conf = 0.0
                for p in ps:
                    lp = torch.log_softmax(logits[p], -1)
                    slot_conf += torch.topk(lp, 1).values[0].item()
                slot_conf /= len(ps)
                
                if slot_conf > best_confidence:
                    best_confidence = slot_conf
                    best_slot_idx = slot_idx
                    
            if best_slot_idx is None:
                new_beams.append((score, current_ids, filled, pred_words))
                continue
                
            ps = gp[best_slot_idx][0]
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
                new_filled = set(filled)
                new_filled.add(best_slot_idx)
                new_pred_words = list(pred_words)
                new_pred_words[best_slot_idx] = word
                new_beams.append((score + slot_score, new_ids, new_filled, new_pred_words))
                
        beams = sorted(new_beams, key=lambda x: -x[0])[:beam_width]
        
    out = []
    for _, _, _, pred_words in beams:
        if pred_words not in out:
            out.append(pred_words)
    return out

def eval_model(repo, nice):
    print(f"Evaluating {nice}...", flush=True)
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).to(dev).eval()
    MASK = tok.mask_token_id
    
    # We track sequence-level results for both Left-to-Right and Easy-First decoders
    l2r_cells = {}   # bucket -> [top1, top5, top10, top20, count]
    easy_cells = {}  # bucket -> [top1, top5, top10, top20, count]
    
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
                
        b = bucket(N)
        l2r_res = l2r_cells.setdefault(b, [0, 0, 0, 0, 0])
        easy_res = easy_cells.setdefault(b, [0, 0, 0, 0, 0])
        
        gold_seq = [norm(g) for _, g in gp]
        
        # 1. Left-to-Right decoding
        ranked_l2r = beam_autoregressive_left_to_right(model, ids, gp, tok, beam_width=5)
        rank_l2r = 999
        for i, cand in enumerate(ranked_l2r):
            if len(cand) == len(gold_seq) and all(norm(c) == g for c, g in zip(cand, gold_seq)):
                rank_l2r = i
                break
        l2r_res[0] += rank_l2r == 0
        l2r_res[1] += rank_l2r < 5
        l2r_res[2] += rank_l2r < 10
        l2r_res[3] += rank_l2r < 20
        l2r_res[4] += 1
        
        # 2. Easy-First decoding
        ranked_easy = beam_autoregressive_easy_first(model, ids, gp, tok, beam_width=5)
        rank_easy = 999
        for i, cand in enumerate(ranked_easy):
            if len(cand) == len(gold_seq) and all(norm(c) == g for c, g in zip(cand, gold_seq)):
                rank_easy = i
                break
        easy_res[0] += rank_easy == 0
        easy_res[1] += rank_easy < 5
        easy_res[2] += rank_easy < 10
        easy_res[3] += rank_easy < 20
        easy_res[4] += 1
        
    return nice, l2r_cells, easy_cells

order = ["1", "2", "3", "4-5", "6+"]
res = [eval_model(r, n) for r, n in MODELS]

# Save Results to Obsidian
print("Saving results to Obsidian...", flush=True)
dest_dir = Path("/Users/shmulc/Stuff/Obsidian/Obsidian/Research/dss_restoration")
dest_dir.mkdir(parents=True, exist_ok=True)
dest_file = dest_dir / "easy_first_results.md"

with open(dest_file, "w", encoding="utf-8") as f:
    f.write("# Dead Sea Scrolls Restoration: Left-to-Right vs. Easy-First Decoding\n\n")
    f.write("Comparative evaluation of **Left-to-Right Sequential Beam Search** vs. our new **Confidence-Based (Easy-First) Beam Search**.\n\n")
    f.write(f"- **Sample size per bucket:** {PER_BUCKET} spans\n")
    f.write(f"- **Evaluation Split:** {split_label}\n")
    f.write(f"- **Book Filter:** {book_filter_label}\n\n")
    
    for nice, l2r_cells, easy_cells in res:
        f.write(f"## Model: {nice}\n\n")
        
        for metric, mi in [("Top-1 Sequence Accuracy", 0), ("Top-10 Sequence Accuracy", 2)]:
            f.write(f"### {metric}\n\n")
            f.write("| Decoder | " + " | ".join(order) + " |\n")
            f.write("| --- | " + " | ".join(["---"] * len(order)) + " |\n")
            
            # Row for Left-to-Right
            row_l2r = "| **Left-to-Right** | "
            cells_l2r = []
            for b in order:
                c = l2r_cells.get(b)
                cells_l2r.append(f"{(c[mi]/c[4]*100 if c and c[4] else 0):.1f}%" if c else "-")
            row_l2r += " | ".join(cells_l2r) + " |\n"
            f.write(row_l2r)
            
            # Row for Easy-First
            row_easy = "| **Easy-First (Confidence-Based)** | "
            cells_easy = []
            for b in order:
                c = easy_cells.get(b)
                cells_easy.append(f"{(c[mi]/c[4]*100 if c and c[4] else 0):.1f}%" if c else "-")
            row_easy += " | ".join(cells_easy) + " |\n"
            f.write(row_easy)
            f.write("\n")
            
            # Show relative change
            row_diff = "| *Relative Change* | "
            cells_diff = []
            for b in order:
                cl = l2r_cells.get(b)
                ce = easy_cells.get(b)
                if cl and ce and cl[4] and ce[4]:
                    pct_l = cl[mi]/cl[4]*100
                    pct_e = ce[mi]/ce[4]*100
                    diff = pct_e - pct_l
                    cells_diff.append(f"{diff:+.1f}%")
                else:
                    cells_diff.append("-")
            row_diff += " | ".join(cells_diff) + " |\n"
            f.write(row_diff)
            f.write("\n")

print("All done!")
