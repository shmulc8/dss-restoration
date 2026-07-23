"""Length-conditioned REAL-lacuna eval on etcbc/dss with DYNAMIC decoding (no subtoken leakage) for BEREL models.

This evaluates BEREL base and BEREL ft-SPAN under realistic conditions:
without leaking the subtoken count of each word.
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

# We evaluate: BEREL base and BEREL ft-SPAN
MODELS = [
    ("dicta-il/BEREL", "BEREL base"),
]
span_dir = repo_path("ft_berel_span")
if span_dir.is_dir():
    MODELS.append((str(span_dir), "BEREL ft-SPAN"))

WINDOW = 40
MIN_PRESERVED = 6
PER_BUCKET = int(os.environ.get("PER_BUCKET", "50"))
TOPN, BEAM, K = 50, 5, 20
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

def beam_autoregressive_dynamic(model, ids, N, tok, beam_width=5):
    MASK = tok.mask_token_id
    word_tokens = [[] for _ in range(N)]
    beams = [(0.0, list(ids), 0, word_tokens)]
    
    max_steps = N * 5
    for step in range(max_steps):
        if all(w_cnt >= N for _, _, w_cnt, _ in beams):
            break
            
        new_beams = []
        for score, current_ids, w_cnt, w_toks in beams:
            if w_cnt >= N:
                new_beams.append((score, current_ids, w_cnt, w_toks))
                continue
                
            try:
                mask_pos = current_ids.index(MASK)
            except ValueError:
                new_beams.append((score, current_ids, N, w_toks))
                continue
                
            with torch.no_grad():
                input_tensor = torch.tensor([current_ids], dtype=torch.long, device=dev)
                logits = model(input_tensor).logits[0, mask_pos].cpu()
                
            lp = torch.log_softmax(logits, -1)
            top = torch.topk(lp, 50)
            
            for token_id, val in zip(top.indices.tolist(), top.values.tolist()):
                token_str = tok.decode([token_id])
                is_subword = token_str.startswith("##")
                
                word_add = 0 if is_subword else 1
                new_w_cnt = w_cnt + word_add
                
                new_w_toks = [list(x) for x in w_toks]
                
                if new_w_cnt > N:
                    # Discard token and remove mask slot
                    new_ids = current_ids[:mask_pos] + current_ids[mask_pos+1:]
                    new_beams.append((score + val, new_ids, N, new_w_toks))
                else:
                    new_ids = list(current_ids)
                    new_ids[mask_pos] = token_id
                    
                    target_idx = max(0, w_cnt - 1 if is_subword else w_cnt)
                    new_w_toks[target_idx].append(token_id)
                    
                    if is_subword:
                        new_ids.insert(mask_pos + 1, MASK)
                        
                    new_beams.append((score + val, new_ids, new_w_cnt, new_w_toks))
                    
        beams = sorted(new_beams, key=lambda x: -x[0])[:beam_width]
        
    out = []
    for _, _, _, w_toks in beams:
        pred_words = []
        for wt in w_toks:
            w_str = tok.decode(wt).replace(" ", "").replace("##", "")
            pred_words.append(w_str)
        if pred_words not in out:
            out.append(pred_words)
    return out

def eval_model(repo, nice):
    print(f"Evaluating {nice}...", flush=True)
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(repo).to(dev).eval()
    MASK = tok.mask_token_id
    
    slot_cells = {}   # bucket -> [t1,t5,t10,t20,n_slots]
    seq_cells = {}    # bucket -> [t1,t5,t10,t20,n_seqs]
    
    for ctx, gap_pos, golds, N in sample:
        masked_ctx = list(ctx)
        for idx in gap_pos:
            masked_ctx[idx] = tok.mask_token
            
        enc = tok(masked_ctx, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=512)
        ids = enc["input_ids"][0]
        
        num_masks = (ids == MASK).sum().item()
        if num_masks != N:
            continue
            
        b = bucket(N)
        sc = slot_cells.setdefault(b, [0, 0, 0, 0, 0])
        sq = seq_cells.setdefault(b, [0, 0, 0, 0, 0])
        
        # Evaluate sequence dynamically
        ranked_seqs = beam_autoregressive_dynamic(model, ids, N, tok, beam_width=5)
        gold_seq = [norm(g) for g in golds]
        
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
        
        # Evaluate individual slots dynamically
        for target_idx, gold in enumerate(golds):
            enc_orig = tok(ctx, is_split_into_words=True, return_tensors="pt",
                            truncation=True, max_length=512)
            wmap = {}
            for pos, wid in enumerate(enc_orig.word_ids(0)):
                if wid is not None:
                    wmap.setdefault(wid, []).append(pos)
                    
            gp_orig = [(wmap.get(g), g_word) for g, g_word in zip(gap_pos, golds)]
            if any(ps is None for ps, _ in gp_orig):
                continue
                
            ids_slot = enc_orig["input_ids"][0].clone()
            
            # Mask other slots as UNK
            for other_idx, (other_ps, _) in enumerate(gp_orig):
                if other_idx != target_idx:
                    for p in other_ps:
                        ids_slot[p] = tok.unk_token_id
                        
            # Replace target slot with exactly 1 MASK token
            target_ps, _ = gp_orig[target_idx]
            target_start = target_ps[0]
            ids_slot = torch.cat([ids_slot[:target_start], torch.tensor([MASK]), ids_slot[target_ps[-1]+1:]])
            
            ranked_slot = beam_autoregressive_dynamic(model, ids_slot, 1, tok, beam_width=5)
            ng = norm(gold)
            
            flat_ranked_slot = [cand[0] if cand else "" for cand in ranked_slot]
            rank = next((i for i, r in enumerate(flat_ranked_slot) if norm(r) == ng), 999)
            
            sc[0] += rank == 0
            sc[1] += rank < 5
            sc[2] += rank < 10
            sc[3] += rank < 20
            sc[4] += 1
            
    return nice, slot_cells, seq_cells

order = ["1", "2", "3", "4-5", "6+"]
res = [eval_model(r, n) for r, n in MODELS]

# Write results
print("Writing results to Obsidian...", flush=True)
results_dir = Path("/Users/shmulc/Stuff/Obsidian/Obsidian/Research/dss_restoration")
results_dir.mkdir(parents=True, exist_ok=True)
results_file = results_dir / "dynamic_berel_results.md"

with open(results_file, "w", encoding="utf-8") as f:
    f.write("# BEREL Model Evaluation (Dynamic Decoding without Subtoken Leakage)\n\n")
    f.write("This table shows the realistic performance of `BEREL base` and `BEREL ft-SPAN` when evaluated **without leaking subtoken length** (using dynamic decoding).\n\n")
    f.write(f"- **Evaluated Spans per Bucket:** {PER_BUCKET}\n")
    f.write(f"- **Target Split:** {split_label}\n")
    f.write(f"- **Book Filter:** {book_filter_label}\n\n")
    
    f.write("## 1. SLOT-LEVEL ACCURACY (Individual Words)\n\n")
    for metric, mi in [("top-10", 2), ("top-20", 3)]:
        f.write(f"### {metric.upper()}\n\n")
        f.write("| Model | " + " | ".join(order) + " |\n")
        f.write("| --- | " + " | ".join(["---"] * len(order)) + " |\n")
        for nice, slot_cells, _ in res:
            row = f"| **{nice}** | "
            cells = []
            for b in order:
                c = slot_cells.get(b)
                cells.append(f"{(c[mi]/c[4]*100 if c and c[4] else 0):.1f}%" if c else "-")
            row += " | ".join(cells) + " |\n"
            f.write(row)
        f.write("\n")
        
    f.write("## 2. SEQUENCE-LEVEL ACCURACY (Phrase Matching)\n\n")
    for metric, mi in [("top-10", 2), ("top-20", 3)]:
        f.write(f"### {metric.upper()}\n\n")
        f.write("| Model | " + " | ".join(order) + " |\n")
        f.write("| --- | " + " | ".join(["---"] * len(order)) + " |\n")
        for nice, _, seq_cells in res:
            row = f"| **{nice}** | "
            cells = []
            for b in order:
                c = seq_cells.get(b)
                cells.append(f"{(c[mi]/c[4]*100 if c and c[4] else 0):.1f}%" if c else "-")
            row += " | ".join(cells) + " |\n"
            f.write(row)
        f.write("\n")

print("All done!")
