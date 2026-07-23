"""Track C: Combined Benchmark — Expanded Corpus Model + Learned RAG vs Base.

Side-by-side comparison of:
1. Base Span-Refined Model (ft_msbert_span_refined)
2. Expanded Corpus Model (ft_msbert_span_refined_expanded)
3. Clitic-Augmented Model (ft_msbert_span_refined_cliticaug)

For every checkpoint, evaluates both raw MLM ranking and the Track A learned
RAG ranking on the same held-out sample. The reranker weights must have been
fit on the disjoint dev-scroll partition by tf_learned_rag_reranker.py.
"""
from collections import Counter
import json
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging as tlog

tlog.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import morph_dss
from utils.dss_split import load_partition
from utils.eval_split import resolve_scroll_filter
from utils.paths import repo_path

MODELS = [
    ("Base Span-Refined", repo_path("ft_msbert_span_refined")),
    ("Clitic-Augmented", repo_path("ft_msbert_span_refined_cliticaug")),
    ("Expanded Corpus (3.8x)", repo_path("ft_msbert_span_refined_expanded")),
]

WINDOW = 40
TOPN = 50
HEB = set(chr(c) for c in range(0x05D0, 0x05EB))
FINAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
DIVINE = {"יי", "ייי", "ה'", "יהו", "יהוה", "אדני"}
dev = "mps" if torch.backends.mps.is_available() else "cpu"

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

# Build retrieval features from training data only.
ngram_db = {}
train_rows = load_partition("train")
train_words = [
    [word for word in row["text"].strip().split() if heb(word)]
    for row in train_rows
]
morph_dss.lemmas(sorted({word for words in train_words for word in words}))
for clean in train_words:
    for n in range(3, 7):
        for i in range(len(clean) - n + 1):
            gram = tuple(norm(word) for word in clean[i:i + n])
            ngram_db.setdefault(gram[:-1], Counter())[gram[-1]] += 1


def rag_features(left_context, candidate_norm):
    best_ngram_len = 0
    best_hit_count = 0
    if len(left_context) >= 2:
        left_normed = [norm(word) for word in left_context[-6:]]
        for prefix_len in range(min(len(left_normed), 5), 1, -1):
            counts = ngram_db.get(tuple(left_normed[-prefix_len:]))
            hit_count = counts.get(candidate_norm, 0) if counts else 0
            if hit_count:
                best_ngram_len = max(best_ngram_len, prefix_len + 1)
                best_hit_count = max(best_hit_count, hit_count)
    return np.array(
        [
            best_ngram_len,
            np.log1p(best_hit_count),
            1.0 if best_hit_count else 0.0,
        ],
        dtype=float,
    )


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

allowed_scrolls, _ = resolve_scroll_filter("heldout")

scrolls = {}
for w in F.otype.s("word"):
    if F.biblical.v(w):
        continue
    sc = L.u(w, "scroll")
    if not sc:
        continue
    scroll_name = F.scroll.v(sc[0])
    if allowed_scrolls is not None and scroll_name not in allowed_scrolls:
        continue
    scrolls.setdefault(sc[0], []).append(winfo(w))

single_word_test_items = []
for ws in scrolls.values():
    for i in range(len(ws)):
        g, is_rec, is_pr = ws[i]
        if is_pr and heb(g):
            lo, hi = max(0, i - WINDOW), min(len(ws), i + WINDOW + 1)
            ctx, gap_pos = [], None
            for k in range(lo, hi):
                word_str, _, pr = ws[k]
                if k == i:
                    gap_pos = len(ctx)
                    ctx.append(word_str)
                elif heb(word_str):
                    ctx.append(word_str)
            if gap_pos is not None:
                single_word_test_items.append((ctx, gap_pos, g))

np.random.default_rng(42).shuffle(single_word_test_items)
sampled_items = single_word_test_items[:300]

def summarize_ranks(ranks):
    total = len(ranks)
    return {
        "top1": sum(rank == 0 for rank in ranks) / total * 100 if total else 0,
        "top5": sum(rank < 5 for rank in ranks) / total * 100 if total else 0,
        "top10": sum(rank < 10 for rank in ranks) / total * 100 if total else 0,
        "top20": sum(rank < 20 for rank in ranks) / total * 100 if total else 0,
        "total": total,
    }


def eval_model_dir(mdir, name, coefficients, intercept):
    print(f"Evaluating: {name} ({mdir})...")
    tok = AutoTokenizer.from_pretrained(str(mdir), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(mdir)).to(dev).eval()
    mask_id = tok.mask_token_id

    records = []
    inference_batch = 16
    for start in range(0, len(sampled_items), inference_batch):
        batch_items = sampled_items[start:start + inference_batch]
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
                input_ids=ids.to(dev),
                attention_mask=enc["attention_mask"].to(dev),
            ).logits.cpu()
        for row_idx, (ctx, gap_pos, gold) in enumerate(batch_items):
            positions = target_positions_by_row[row_idx]
            if not positions:
                continue
            lp = torch.log_softmax(logits[row_idx, positions[0]], -1)
            top = torch.topk(lp, TOPN)
            records.append(
                (
                    ctx[:gap_pos],
                    gold,
                    [tok.decode([idx]).strip() for idx in top.indices.tolist()],
                    top.values.tolist(),
                )
            )

    morph_dss.lemmas(
        word
        for left_context, gold, predictions, _ in records
        for word in [*left_context, gold, *predictions]
        if word
    )

    baseline_ranks = []
    reranked_ranks = []
    for left_context, gold, predictions, mlm_scores in records:
        gold_norm = norm(gold)
        baseline_ranks.append(
            next(
                (i for i, prediction in enumerate(predictions) if norm(prediction) == gold_norm),
                999,
            )
        )

        learned_candidates = []
        for prediction, mlm_score in zip(predictions, mlm_scores):
            retrieval = rag_features(left_context, norm(prediction))
            features = np.concatenate(([mlm_score], retrieval))
            learned_score = float(features @ coefficients + intercept)
            learned_candidates.append((learned_score, prediction))
        learned_candidates.sort(key=lambda item: -item[0])
        reranked_ranks.append(
            next(
                (
                    i
                    for i, (_, prediction) in enumerate(learned_candidates)
                    if norm(prediction) == gold_norm
                ),
                999,
            )
        )

    del model
    if dev == "mps":
        torch.mps.empty_cache()
    return {
        "mlm": summarize_ranks(baseline_ranks),
        "learned_rag": summarize_ranks(reranked_ranks),
    }


def print_row(label, result, baseline_top10):
    delta = result["top10"] - baseline_top10
    print(
        f"{label:42s} {result['top1']:7.1f}%  {result['top5']:7.1f}%  "
        f"{result['top10']:7.1f}%  {result['top20']:7.1f}%  {delta:+7.1f}"
    )

if __name__ == "__main__":
    weights_path = ROOT / "analysis" / "reports" / "learned_rag_weights.json"
    if not weights_path.is_file():
        raise RuntimeError(
            f"Learned reranker weights not found at {weights_path}; "
            "run eval/tf_learned_rag_reranker.py first."
        )
    weights = json.loads(weights_path.read_text())
    if weights.get("dev_split") != "dev-scrolls" or weights.get("test_split") != "heldout-scrolls":
        raise RuntimeError(
            "Reranker weights do not record the required disjoint dev/heldout protocol; "
            "rerun eval/tf_learned_rag_reranker.py."
        )
    coefficients = np.asarray(weights["coefficients"], dtype=float)
    intercept = float(weights["intercept"])

    results = {}
    for name, mdir in MODELS:
        if not mdir.is_dir():
            print(f"SKIP: {name} — checkpoint not found at {mdir}")
            continue
        results[name] = eval_model_dir(mdir, name, coefficients, intercept)

    print("\n" + "=" * 88)
    print("=== ULTIMATE COMBINED MODEL COMPARISON BENCHMARK (Track C) ===")
    print("=" * 88)
    print(
        f"{'Model / ranking':42s} {'Top-1':>8s} {'Top-5':>8s} "
        f"{'Top-10':>8s} {'Top-20':>8s} {'Δ T10':>8s}"
    )
    print("-" * 88)
    base_top10 = results["Base Span-Refined"]["mlm"]["top10"]
    for name, model_results in results.items():
        print_row(f"{name} — MLM", model_results["mlm"], base_top10)
        print_row(
            f"{name} — learned RAG",
            model_results["learned_rag"],
            base_top10,
        )
    print("=" * 88)
    print(f"Held-out items: {len(sampled_items)}; deltas are percentage points vs base MLM.")

    report_path = ROOT / "analysis" / "reports" / "ultimate_combined_benchmark.json"
    report_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "evaluation_split": "heldout-scrolls",
                    "reranker_fit_split": weights["dev_split"],
                    "sample_seed": 42,
                    "sample_size": len(sampled_items),
                    "candidate_topn": TOPN,
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved benchmark report → {report_path}")
