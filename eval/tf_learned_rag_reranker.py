"""Track A: Learned Weighted RAG Reranker.

Replaces the fixed-weight RAG boost (count * 2.0) with a logistic regression
model trained on dev-set features to optimally combine:
  1. MsBERT log-probability score
  2. RAG n-gram match length (3/4/5/6)
  3. RAG hit count (log-scaled)

Trains on dev partition, evaluates on heldout partition.
"""
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog
from sklearn.linear_model import LogisticRegression

tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import morph_dss
from utils.dss_split import load_partition
from utils.paths import repo_path

MODEL_NAME = "ft_msbert_span_refined"
model_dir = repo_path(MODEL_NAME)
if not model_dir.is_dir():
    model_dir = "dicta-il/MsBERT"

WINDOW = 40
TOPN = 50
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
dev_device = "mps" if torch.backends.mps.is_available() else "cpu"

def norm(w):
    if not w:
        return ""
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

# ── Build Parallel Witness N-gram DB (from training partition only) ──
train_rows = load_partition("train")
train_texts = [r["text"].strip().split() for r in train_rows]
train_vocab = sorted({word for words in train_texts for word in words if heb(word)})
morph_dss.lemmas(train_vocab)

# Store n-grams with their match lengths for feature extraction
ngram_db = {}  # prefix -> Counter of target words
ngram_len_db = {}  # prefix -> n-gram length
for words in train_texts:
    clean = [w for w in words if heb(w)]
    for n in range(3, 7):
        for i in range(len(clean) - n + 1):
            gram = tuple(norm(w) for w in clean[i:i+n])
            prefix = gram[:-1]
            target = gram[-1]
            ngram_db.setdefault(prefix, Counter())[target] += 1
            ngram_len_db[prefix] = n

print(f"RAG DB built: {len(ngram_db)} n-gram contexts")

# ── Load DSS Corpus from Text-Fabric ──
from tf.fabric import Fabric
TF_DIR = Path("/Users/shmulc/text-fabric-data/github/ETCBC/dss/tf/2.0")
TF = Fabric(locations=str(TF_DIR), silent="deep")
api = TF.load("otype glyph rec biblical scroll", silent="deep")
F, L = api.F, api.L

def winfo(w):
    signs = L.d(w, "sign")
    g = "".join(F.glyph.v(s) or "" for s in signs)
    recs = [F.rec.v(s) for s in signs]
    return g, (bool(signs) and all(r == 1 for r in recs)), (bool(signs) and all(r != 1 for r in recs))

from utils.eval_split import resolve_scroll_filter

scrolls = {}
for w in F.otype.s("word"):
    if F.biblical.v(w):
        continue
    sc = L.u(w, "scroll")
    if not sc:
        continue
    scroll_name = F.scroll.v(sc[0])
    scrolls.setdefault(scroll_name, []).append(winfo(w))

def build_single_word_items(split):
    """Build intact single-word probes from one protocol scroll partition."""
    allowed_scrolls, split_label = resolve_scroll_filter(split)
    items = []
    for scroll_name, ws in scrolls.items():
        if allowed_scrolls is not None and scroll_name not in allowed_scrolls:
            continue
        for i, (g, _is_rec, is_pr) in enumerate(ws):
            if not (is_pr and heb(g)):
                continue
            lo, hi = max(0, i - WINDOW), min(len(ws), i + WINDOW + 1)
            ctx, gap_pos = [], None
            for k in range(lo, hi):
                word_str, _, _ = ws[k]
                if k == i:
                    gap_pos = len(ctx)
                    ctx.append(word_str)
                elif heb(word_str):
                    ctx.append(word_str)
            if gap_pos is not None:
                items.append((ctx, gap_pos, g))
    return items, split_label


# Learn only on development scrolls; touch held-out scrolls only for evaluation.
dev_items, dev_split_label = build_single_word_items("dev")
test_items, test_split_label = build_single_word_items("heldout")
np.random.default_rng(42).shuffle(dev_items)
np.random.default_rng(42).shuffle(test_items)

# Subsample for speed
dev_sample = dev_items[:200]
test_sample = test_items[:300]

print(
    f"Dev items: {len(dev_sample)} ({dev_split_label}), "
    f"Test items: {len(test_sample)} ({test_split_label})"
)

# ── Load model ──
tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(str(model_dir)).to(dev_device).eval()
mask_id = tok.mask_token_id

def get_rag_features(ctx_words, candidate_norm):
    """Extract RAG features for a candidate word given left context."""
    best_ngram_len = 0
    best_hit_count = 0
    total_hits = 0

    if len(ctx_words) >= 2:
        left_normed = [norm(w) for w in ctx_words[-6:]]
        for pref_len in range(min(len(left_normed), 6), 1, -1):
            sub_pref = tuple(left_normed[-pref_len:])
            if sub_pref in ngram_db:
                count = ngram_db[sub_pref].get(candidate_norm, 0)
                if count > 0:
                    best_ngram_len = max(best_ngram_len, pref_len + 1)
                    best_hit_count = max(best_hit_count, count)
                    total_hits += count
    return best_ngram_len, best_hit_count, total_hits

def extract_features_for_items(items, label=""):
    """Extract (features, labels) for logistic regression training/evaluation."""
    all_features = []
    all_labels = []
    all_meta = []  # (gold_norm, item_idx) for evaluation
    records = []

    # Batched MLM inference; defer feature normalization so the lemmatizer can
    # process all previously unseen words in large batches rather than one call
    # per candidate.
    inference_batch = 16
    for start in range(0, len(items), inference_batch):
        batch_items = items[start:start + inference_batch]
        enc = tok(
            [ctx for ctx, _, _ in batch_items],
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        ids = enc["input_ids"].clone()
        target_positions_by_row = []
        for row_idx, (_, gap_pos, _) in enumerate(batch_items):
            positions = [
                pos
                for pos, word_id in enumerate(enc.word_ids(batch_index=row_idx))
                if word_id == gap_pos
            ]
            target_positions_by_row.append(positions)
            for pos in positions:
                ids[row_idx, pos] = mask_id
        with torch.no_grad():
            logits = model(
                input_ids=ids.to(dev_device),
                attention_mask=enc["attention_mask"].to(dev_device),
            ).logits.cpu()
        for row_idx, (ctx, gap_pos, gold) in enumerate(batch_items):
            target_positions = target_positions_by_row[row_idx]
            if not target_positions:
                continue
            lp = torch.log_softmax(logits[row_idx, target_positions[0]], -1)
            top = torch.topk(lp, TOPN)
            predictions = [
                tok.decode([tok_id]).strip() for tok_id in top.indices.tolist()
            ]
            records.append(
                (
                    start + row_idx,
                    ctx[:gap_pos],
                    gold,
                    predictions,
                    top.values.tolist(),
                )
            )

    morph_dss.lemmas(
        word
        for _, left_context, gold, predictions, _ in records
        for word in [*left_context, gold, *predictions]
        if word
    )

    for item_idx, ctx_left, gold, predictions, mlm_scores in records:
        gold_norm = norm(gold)
        for w_str, mlm_score in zip(predictions, mlm_scores):
            w_norm = norm(w_str)

            rag_ngram_len, rag_hit_count, rag_total = get_rag_features(ctx_left, w_norm)

            features = [
                mlm_score,                          # MsBERT log-prob
                rag_ngram_len,                      # Best matching n-gram length (0-6)
                np.log1p(rag_hit_count),            # Log-scaled hit count
                1.0 if rag_hit_count > 0 else 0.0,  # Binary RAG match flag
            ]
            label_val = 1 if w_norm == gold_norm else 0

            all_features.append(features)
            all_labels.append(label_val)
            all_meta.append((gold_norm, item_idx))

    return np.array(all_features), np.array(all_labels), all_meta

print("Extracting dev features...")
dev_features, dev_labels, dev_meta = extract_features_for_items(dev_sample, "dev")
print(f"  Dev: {len(dev_features)} candidates, {dev_labels.sum()} positives")

print("Extracting test features...")
test_features, test_labels, test_meta = extract_features_for_items(test_sample, "test")
print(f"  Test: {len(test_features)} candidates, {test_labels.sum()} positives")

# ── Train Logistic Regression on Dev Set ──
print("\nTraining Learned RAG Reranker (Logistic Regression on dev set)...")
if not dev_labels.any():
    raise RuntimeError(
        "No positive gold candidates were present in the dev top-N candidate sets; "
        "cannot train the reranker."
    )
clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
clf.fit(dev_features, dev_labels)

feature_names = ["mlm_log_prob", "rag_ngram_len", "log_rag_hits", "rag_match_flag"]
print(f"  Learned Weights: {dict(zip(feature_names, clf.coef_[0].round(4)))}")
print(f"  Intercept: {clf.intercept_[0]:.4f}")

# ── Evaluate: Baseline (MLM only) vs Learned Reranker ──
def evaluate_reranker(features, labels, meta, use_learned=False):
    """Evaluate Top-K accuracy using either baseline MLM scores or learned reranker."""
    if use_learned:
        scores = clf.predict_proba(features)[:, 1]
    else:
        scores = features[:, 0]  # Just MLM log-prob

    # Group by item_idx
    items_dict = {}
    for i, (gold_norm, item_idx) in enumerate(meta):
        items_dict.setdefault(item_idx, []).append((scores[i], labels[i], gold_norm))

    top1, top5, top10, top20, total = 0, 0, 0, 0, 0
    for item_idx, candidates in items_dict.items():
        # Sort by score descending
        ranked = sorted(candidates, key=lambda x: -x[0])
        gold_rank = next((i for i, (_, lbl, _) in enumerate(ranked) if lbl == 1), 999)

        top1 += (gold_rank == 0)
        top5 += (gold_rank < 5)
        top10 += (gold_rank < 10)
        top20 += (gold_rank < 20)
        total += 1

    return {
        "top1": top1/total*100 if total else 0,
        "top5": top5/total*100 if total else 0,
        "top10": top10/total*100 if total else 0,
        "top20": top20/total*100 if total else 0,
        "total": total
    }

res_baseline = evaluate_reranker(test_features, test_labels, test_meta, use_learned=False)
res_learned = evaluate_reranker(test_features, test_labels, test_meta, use_learned=True)

print("\n==================================================")
print("=== LEARNED RAG RERANKER BENCHMARK (Track A) ===")
print("==================================================")
print(f"Total Test Items: {res_baseline['total']}")
print()
print(f"Decoding Strategy                      Top-1    Top-5   Top-10   Top-20")
print(f"1. Baseline (MLM score only)          {res_baseline['top1']:.1f}%   {res_baseline['top5']:.1f}%   {res_baseline['top10']:.1f}%   {res_baseline['top20']:.1f}%")
print(f"2. Learned RAG Reranker               {res_learned['top1']:.1f}%   {res_learned['top5']:.1f}%   {res_learned['top10']:.1f}%   {res_learned['top20']:.1f}%")
diff10 = res_learned['top10'] - res_baseline['top10']
print(f"→ Net Learned RAG Reranker Top-10 Gain: {diff10:+.1f} percentage points")
print("==================================================")

# Save learned weights for future use
import json
weights_path = ROOT / "analysis" / "reports" / "learned_rag_weights.json"
weights_data = {
    "feature_names": feature_names,
    "coefficients": clf.coef_[0].tolist(),
    "intercept": clf.intercept_[0],
    "dev_size": len(dev_sample),
    "test_size": len(test_sample),
    "dev_split": dev_split_label,
    "test_split": test_split_label,
    "candidate_topn": TOPN,
    "baseline_top10": res_baseline["top10"],
    "learned_top10": res_learned["top10"],
    "gain_top10": diff10,
}
weights_path.write_text(json.dumps(weights_data, indent=2))
print(f"\nSaved learned weights → {weights_path}")
