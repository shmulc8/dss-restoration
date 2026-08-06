"""Run one model against one masking protocol, and score it at the word level.

Deliberately separate from eval_utils.py: that module holds the masking /
prediction / evaluation *classes* and is a verbatim copy of the class cells in
experiments/baseline.ipynb (see CLAUDE.md). This module is the driver around
them, and is free to evolve on its own.

Two-stage by design:

    run_eval()        expensive, needs a GPU and a model -> predictions.jsonl
    score_run_dir()   cheap, CPU only, no model          -> word_scores.csv,
                                                            metrics.json,
                                                            summary.xlsx

predictions.jsonl is the durable artifact and carries everything the scorer
needs, so inventing a new metric means re-scoring existing runs in seconds
rather than re-running inference. manifest.json records SCORER_VERSION so a
stale derivation is detectable.

Uploaded to Colab alongside eval_utils.py, tokenizer_compat.py and the dataset
xlsx.
"""

import hashlib
import json
import platform
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from eval.masking import (
    Evaluator,
    MultiSpanPredictionPolicy,
    PercentageContentMaskingPolicy,
)
from utils.tokenizer_compat import emits_whitespace_tokens, load_tokenizer
from typing import List, Tuple
from scipy.stats import norm


def normalize_hebrew_lemma(text: str) -> str:
    """Normalize Hebrew orthographic and plene/defective spelling variants for Morpho-Lemmatic scoring."""
    if not text:
        return ""
    # Standardize final letters to medial form for lemmatic comparison
    final_map = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
    res = "".join(final_map.get(ch, ch) for ch in text)
    # Remove plene alef/vav/yod insertions common in Qumran orthography (e.g. לוא -> לא, כיא -> כי)
    res = res.replace("לוא", "לא").replace("כיא", "כי").replace("זואת", "זאת")
    return res


def mcnemar_test(hits_a: List[int], hits_b: List[int]) -> Tuple[float, float]:
    """Compute McNemar paired statistical test z-statistic and two-sided p-value.
    
    Args:
        hits_a: List of 1/0 hit indicators for Model A.
        hits_b: List of 1/0 hit indicators for Model B.
    """
    b = sum(1 for a, b in zip(hits_a, hits_b) if a == 1 and b == 0)
    c = sum(1 for a, b in zip(hits_a, hits_b) if a == 0 and b == 1)

    if b + c == 0:
        return 0.0, 1.0

    z = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = 2.0 * (1.0 - norm.cdf(float(np.sqrt(z))))
    return z, float(p_value)


# Bump when the scoring layer changes meaning. manifest.json records it, so
# build_report.py can tell a run scored under an older scorer from a current one.
SCORER_VERSION = 1

HIT_KS = (1, 3, 5, 10)

# Cluster bootstrap settings. Fixed, not exposed as run_eval kwargs: they are a
# property of how we report, not of the experiment, and holding them constant is
# what makes metrics.json reproducible from predictions.jsonl.
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_ALPHA = 0.05

DEFAULT_RESULTS_REPO = "almo2988/dss-eval-results"


# --------------------------------------------------------------------------
# 1a. Identity
# --------------------------------------------------------------------------


def sentence_uid(row):
    """Stable id for a *location* in the corpus, not for a piece of text.

    Deliberately independent of the sentence's words: with INCLUDE_PPP toggled,
    the same (scroll, fragment, line range) carries different text between
    dataset variants, and we still want to recognise it as the same item.
    text_sha() is what records whether the text also matches.
    """
    key = f"{row['scroll']}|{row['fragment']}|{row['line_start']}|{row['line_end']}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def text_sha(sentence):
    """Stable id for the text itself."""
    return hashlib.sha1(str(sentence).strip().encode()).hexdigest()[:12]


def dataset_fingerprint(path):
    """Stable id for an exact dataset file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _short_json_sha(payload, length):
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:length]


def protocol_id(fingerprint, split, policy_class, mask_ratio, span_concentration, seed):
    """The *task*. Two runs are comparable iff their protocol_ids are equal."""
    return _short_json_sha(
        {
            "dataset_fingerprint": fingerprint,
            "split": split,
            "policy_class": policy_class,
            "mask_ratio": mask_ratio,
            "span_concentration": span_concentration,
            "seed": seed,
        },
        8,
    )


def decode_id(predictor_class, top_k, beam_width, beam_depth):
    """The *search budget*.

    Kept out of protocol_id on purpose: beam width is a property of the system
    under test, not of the data. But a model given beam_width=10 against one
    given 3 is not a fair fight, so the viewer flags a mismatch inside one
    leaderboard.
    """
    return _short_json_sha(
        {
            "predictor_class": predictor_class,
            "top_k": top_k,
            "beam_width": beam_width,
            "beam_depth": beam_depth,
        },
        8,
    )


def make_run_id(model_id, protocol, decode, created=None):
    stamp = (created or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_id.replace('/', '--')}__{protocol}__{decode}__{stamp}"


# --------------------------------------------------------------------------
# 1c. Normalisation and character-level string metrics
# --------------------------------------------------------------------------


def normalize_he(s, collapse_finals=False):
    """Strip everything that is not an unpointed Hebrew letter.

    The corpus is nominally unpointed but does carry stray combining marks
    (U+05BD, U+05C5, U+059C, U+05C4 all occur in the sentence text), which would
    otherwise count as character errors.

    collapse_finals is off by default: folding final forms into their medial
    counterparts is a linguistic claim about what counts as the same word, not a
    cleanup step.
    """
    s = unicodedata.normalize("NFC", s)
    s = "".join(
        c for c in s if unicodedata.category(c) != "Mn"
    )  # strip nikkud/cantillation
    s = "".join(c for c in s if "א" <= c <= "ת")  # letters only
    if collapse_finals:
        s = s.translate(str.maketrans("ךםןףץ", "כמנפצ"))
    return s


def char_levenshtein(a, b):
    """Edit distance over *characters of the surface string*.

    Not Evaluator._levenshtein_similarity, which runs over token ids and
    therefore measures subwords for MsBERT and characters for TavBERT -- the
    exact cross-family incomparability this layer exists to fix.
    """
    n, m = len(a), len(b)

    if n == 0:
        return m
    if m == 0:
        return n

    previous = list(range(m + 1))

    for i in range(1, n + 1):
        current = [i] + [0] * m

        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )

        previous = current

    return previous[m]


def char_sim(a, b):
    """1 - normalised character edit distance. Two empty strings are identical."""
    longest = max(len(a), len(b))

    if longest == 0:
        return 1.0

    return 1.0 - char_levenshtein(a, b) / longest


# --------------------------------------------------------------------------
# 1c. Word-level scoring
# --------------------------------------------------------------------------

_METRIC_COLUMNS = (
    ["char_sim_top1", "char_sim_top1_norm", "char_sim_oracle", "char_sim_oracle_norm"]
    + [f"hit_at_{k}" for k in HIT_KS]
    + ["gold_rank", "mrr"]
    + [f"hit_at_{k}_norm" for k in HIT_KS]
    + ["gold_rank_norm", "mrr_norm"]
)

WORD_SCORE_COLUMNS = (
    [
        "sentence_uid",
        "text_sha",
        "word_index",
        "span_index",
        "span_word_count",
        "span_length",
        "alignment",
        "gold_word",
        "pred_top1",
        "n_candidates",
    ]
    + _METRIC_COLUMNS
    + ["token_accuracy", "levenshtein_similarity", "candidates_json"]
)

# Metrics that get an overall mean + bootstrap CI. gold_rank is excluded: it is
# null whenever the gold is not in the candidate list, so its mean would be a
# mean over the subset that happened to succeed. mrr is the well-defined version.
MEAN_METRICS = [c for c in _METRIC_COLUMNS if not c.startswith("gold_rank")] + [
    "token_accuracy",
    "levenshtein_similarity",
]


def split_candidate_words(tokens):
    """Split a WordPiece candidate's tokens into whole words.

    A token starts a new word iff it does not carry the "##" continuation
    prefix. Character-level candidates never reach here: their spans are always
    exactly one word (a space is a real token, so it breaks the span), and those
    are read straight off the candidate's decoded text.
    """
    words = []

    for token in tokens:
        if token.startswith("##"):
            if words:
                words[-1] += token[2:]
            else:
                words.append(token[2:])
        else:
            words.append(token)

    return words


def _per_word_candidates(span, n_words):
    """Candidate strings per gold-word slot, best rank first, deduped.

    Returns (alignment, [[(text, score), ...] per slot]).

    Alignment is decided by the **top-1 candidate**, because top-1 is what the
    headline metrics read. A lower-ranked candidate that splits into the wrong
    number of words is simply skipped, which is why n_candidates can come out
    below top_k and is recorded per row.
    """
    candidates = span["candidates"]

    if n_words == 1:
        # The whole candidate phrase is this word's prediction.
        slots = [[(c["text"], c["score"]) for c in candidates]]
        alignment = "exact"
    else:
        splits = [(split_candidate_words(c["tokens"]), c["score"]) for c in candidates]

        if not splits or len(splits[0][0]) != n_words:
            return "unaligned", None

        slots = [[] for _ in range(n_words)]
        for words, score in splits:
            if len(words) != n_words:
                continue  # cannot place it; skip this candidate
            for j in range(n_words):
                slots[j].append((words[j], score))
        alignment = "split"

    deduped = []
    for slot in slots:
        seen = {}
        for text, score in slot:  # rank order, so first win = best rank
            if text not in seen:
                seen[text] = score
        deduped.append(list(seen.items()))

    return alignment, deduped


def _word_metrics(gold, candidates, suffix=""):
    """char_sim / hit@k / gold_rank / mrr / oracle for one gold word."""
    texts = [text for text, _ in candidates]

    top1 = texts[0] if texts else ""

    rank = None
    for i, text in enumerate(texts):
        if text == gold:
            rank = i + 1
            break

    out = {
        f"char_sim_top1{suffix}": char_sim(top1, gold),
        f"char_sim_oracle{suffix}": max(
            (char_sim(t, gold) for t in texts), default=0.0
        ),
        f"gold_rank{suffix}": rank,
        f"mrr{suffix}": 1.0 / rank if rank else 0.0,
    }

    for k in HIT_KS:
        out[f"hit_at_{k}{suffix}"] = int(gold in texts[:k])

    return out


def score_span(span, evaluator=None):
    """One span dict from predictions.jsonl -> its word_scores rows.

    Returns (rows, n_gold_words). For an unaligned span the single row returned
    is the *span* row: it records that the span existed and what was predicted,
    with a null word_index and null metrics, so it can never be mistaken for a
    scored word.
    """
    evaluator = evaluator or Evaluator()

    word_indices = span["word_indices"]
    gold_words = span["gold_words"]
    n_words = len(word_indices)

    if n_words == 0:
        # Degenerate: more prediction spans than gold token slots (truncation).
        return [], 0

    candidates = span["candidates"]

    # Legacy, token-id-based, within-family only. Same definition as
    # Evaluator.evaluate: best partial score over the top-k candidates. Copied
    # onto every word row of the span, since it has no word-level meaning.
    gold_token_ids = span.get("gold_token_ids") or []
    span_length = span.get("span_length") or max(len(gold_token_ids), 1)

    n_tokens_correct = max(
        (
            evaluator._positional_matches(c["token_ids"], gold_token_ids)
            for c in candidates
        ),
        default=0,
    )
    legacy = {
        "token_accuracy": n_tokens_correct / span_length if span_length else None,
        "levenshtein_similarity": max(
            (
                evaluator._levenshtein_similarity(c["token_ids"], gold_token_ids)
                for c in candidates
            ),
            default=0.0,
        ),
    }

    alignment, slots = _per_word_candidates(span, n_words)

    base = {
        "sentence_uid": span["sentence_uid"],
        "text_sha": span["text_sha"],
        "span_index": span["span_index"],
        "span_word_count": n_words,
        "span_length": span_length,
        "alignment": alignment,
    }

    if alignment == "unaligned":
        row = dict(base)
        row.update(
            {
                "word_index": None,
                "gold_word": span.get("gold_text", " ".join(gold_words)),
                "pred_top1": candidates[0]["text"] if candidates else "",
                "n_candidates": None,
                "candidates_json": json.dumps(
                    [[c["text"], c["score"]] for c in candidates], ensure_ascii=False
                ),
            }
        )
        row.update({key: None for key in _METRIC_COLUMNS})
        row.update(legacy)
        return [row], n_words

    rows = []

    for j, (word_index, gold) in enumerate(zip(word_indices, gold_words)):
        slot = slots[j]

        row = dict(base)
        row.update(
            {
                "word_index": word_index,
                "gold_word": gold,
                "pred_top1": slot[0][0] if slot else "",
                "n_candidates": len(slot),
                "candidates_json": json.dumps(slot, ensure_ascii=False),
            }
        )
        row.update(_word_metrics(gold, slot))
        row.update(
            _word_metrics(
                normalize_he(gold),
                [(normalize_he(t), s) for t, s in slot],
                suffix="_norm",
            )
        )
        row.update(legacy)

        rows.append(row)

    return rows, n_words


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def mcnemar_test(hits_a: List[int], hits_b: List[int]) -> Tuple[float, float]:
    """Compute McNemar paired statistical test z-statistic and two-sided p-value.
    
    Args:
        hits_a: List of 1/0 hit indicators for Model A.
        hits_b: List of 1/0 hit indicators for Model B.
    """
    import numpy as np
    from scipy.stats import norm
    b = sum(1 for a, b in zip(hits_a, hits_b) if a == 1 and b == 0)
    c = sum(1 for a, b in zip(hits_a, hits_b) if a == 0 and b == 1)

    if b + c == 0:
        return 0.0, 1.0

    z = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = 2.0 * (1.0 - norm.cdf(np.sqrt(z)))
    return z, p_value


def _cluster_bootstrap(df, metrics, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap CIs, resampling *sentences*.

    Words inside a sentence share a context and a masking draw, so they are
    correlated; resampling words instead of sentences would give intervals that
    are far too tight. One shared index matrix across all metrics, which is both
    correct and cheap.
    """
    import numpy as np
    out = {}

    if df.empty:
        return {
            m: {"mean": None, "ci_low": None, "ci_high": None, "n": 0} for m in metrics
        }

    groups = df.groupby("sentence_uid", sort=True)
    order = list(groups.groups)
    n_clusters = len(order)

    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_clusters, size=(b, n_clusters))

    for metric in metrics:
        values = pd.to_numeric(df[metric], errors="coerce")
        per_sentence = values.groupby(df["sentence_uid"], sort=True)

        sums = per_sentence.sum().reindex(order).to_numpy(dtype=float)
        counts = per_sentence.count().reindex(order).to_numpy(dtype=float)

        total = counts.sum()

        if total == 0:
            out[metric] = {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
            continue

        resampled = sums[draw].sum(axis=1) / np.maximum(counts[draw].sum(axis=1), 1e-12)

        low, high = np.percentile(
            resampled, [100 * BOOTSTRAP_ALPHA / 2, 100 * (1 - BOOTSTRAP_ALPHA / 2)]
        )

        out[metric] = {
            "mean": float(sums.sum() / total),
            "ci_low": float(low),
            "ci_high": float(high),
            "n": int(total),
        }

    return out


def _slice_means(df, by, metrics):
    """Plain means per slice value. No CIs: slices are for shape, not for claims."""
    out = {}

    for value, group in df.groupby(by, sort=True):
        entry = {"n": int(len(group))}
        for metric in metrics:
            series = pd.to_numeric(group[metric], errors="coerce")
            entry[metric] = float(series.mean()) if series.notna().any() else None
        out[str(value)] = entry

    return out


def _histogram(series):
    counts = series.value_counts().sort_index()
    return {
        "mean": float(series.mean()) if len(series) else None,
        "histogram": {str(k): int(v) for k, v in counts.items()},
    }


def compute_metrics(word_df, sentences, spans):
    """metrics.json payload: overall means with CIs, slices, and the §1.3 diagnostics."""
    scored = word_df[word_df["alignment"] != "unaligned"].copy()

    spans_per_sentence = pd.Series([s["n_spans"] for s in sentences], dtype="float64")
    words_per_span = pd.Series([len(s["word_indices"]) for s in spans], dtype="float64")

    gold_len = scored["gold_word"].astype(str).str.len()

    unaligned = word_df[word_df["alignment"] == "unaligned"]

    payload = {
        "scorer_version": SCORER_VERSION,
        "n_sentences": len(sentences),
        "n_spans": len(spans),
        "n_words_scored": int(len(scored)),
        "n_words_in_unaligned_spans": int(unaligned["span_word_count"].sum()),
        "bootstrap": {
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "method": "percentile",
            "cluster": "sentence",
            "alpha": BOOTSTRAP_ALPHA,
        },
        "overall": _cluster_bootstrap(scored, MEAN_METRICS),
        "slices": {},
        "distributions": {
            "spans_per_sentence": _histogram(spans_per_sentence),
            "words_per_span": _histogram(words_per_span),
        },
    }

    if not scored.empty:
        payload["slices"] = {
            "span_word_count": _slice_means(scored, "span_word_count", MEAN_METRICS),
            "span_length": _slice_means(scored, "span_length", MEAN_METRICS),
            "gold_word_len": _slice_means(
                scored.assign(gold_word_len=gold_len), "gold_word_len", MEAN_METRICS
            ),
        }

        if "scroll" in scored.columns:
            payload["slices"]["scroll"] = _slice_means(scored, "scroll", MEAN_METRICS)

    return payload


# --------------------------------------------------------------------------
# predictions.jsonl -> derived files
# --------------------------------------------------------------------------


def read_predictions(run_dir):
    """(sentence records, span records) from a run's predictions.jsonl."""
    sentences, spans = [], []

    with open(Path(run_dir) / "predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record") == "sentence":
                sentences.append(record)
            else:
                spans.append(record)

    return sentences, spans


def score_records(sentences, spans):
    """(word_scores DataFrame, counts dict) — the whole derived layer, CPU only."""
    evaluator = Evaluator()

    rows = []
    for span in spans:
        span_rows, _ = score_span(span, evaluator)
        rows.extend(span_rows)

    word_df = pd.DataFrame(rows, columns=WORD_SCORE_COLUMNS)

    scroll_by_uid = {s["sentence_uid"]: s.get("scroll") for s in sentences}
    if not word_df.empty:
        word_df["scroll"] = word_df["sentence_uid"].map(scroll_by_uid)

    masked_words = sum(len(s["masked_word_indices"]) for s in sentences)
    aligned = (
        int((word_df["alignment"] != "unaligned").sum()) if not word_df.empty else 0
    )

    counts = {
        "sentences": len(sentences),
        "spans": len(spans),
        "masked_words": masked_words,
        "aligned_words": aligned,
        # Everything a masked word can fail to become a scored row: an
        # unsplittable candidate, or a mask lost to sequence truncation.
        "unaligned_words": masked_words - aligned,
        "unaligned_spans": int((word_df["alignment"] == "unaligned").sum())
        if not word_df.empty
        else 0,
    }

    return word_df, counts


def score_run_dir(run_dir):
    """Rebuild word_scores.csv / metrics.json / summary.xlsx from predictions.jsonl.

    The only entry point that writes the derived files, so run_eval and
    build_report.py --rescore cannot drift apart.
    """
    run_dir = Path(run_dir)

    sentences, spans = read_predictions(run_dir)
    word_df, counts = score_records(sentences, spans)

    word_df.to_csv(run_dir / "word_scores.csv", index=False)

    metrics = compute_metrics(word_df, sentences, spans)
    metrics["counts"] = counts
    _write_json(run_dir / "metrics.json", metrics)

    _write_summary_xlsx(run_dir / "summary.xlsx", word_df, metrics)

    return word_df, metrics, counts


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_summary_xlsx(path, word_df, metrics):
    try:
        if "overall" not in metrics or not metrics["overall"]:
            return
        overall = pd.DataFrame(
            [{"metric": name, **values} for name, values in metrics["overall"].items() if values]
        )

        slices = []
        for name, table in metrics.get("slices", {}).items():
            for value, entry in table.items():
                slices.append({"slice": name, "value": value, **entry})

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            overall.to_excel(writer, sheet_name="overall", index=False)
            pd.DataFrame(slices).to_excel(writer, sheet_name="slices", index=False)
            word_df.drop(columns=["candidates_json"], errors="ignore").to_excel(
                writer, sheet_name="words", index=False
            )
    except Exception as e:
        pass



# --------------------------------------------------------------------------
# 1b. The runner
# --------------------------------------------------------------------------


def resolve_device(device=None):
    """cuda -> mps -> cpu, the same priority the notebooks use."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_revision(model_id, hf_token=None):
    """Pin a hub model to an exact commit; a local path has no revision.

    Without this, "dicta-il/MsBERT" means whatever main happened to be on the
    day of the run, and two runs months apart are silently different models.
    """
    if Path(model_id).exists():
        return {"source": "local", "revision": None}

    try:
        from huggingface_hub import HfApi

        return {
            "source": "hub",
            "revision": HfApi(token=hf_token).model_info(model_id).sha,
        }
    except Exception as exc:  # offline, private, or renamed
        print(
            f"  ! could not resolve revision for {model_id}: {type(exc).__name__}: {exc}"
        )
        return {"source": "hub", "revision": None}


def run_eval(
    model_id: str,
    dataset_path: str,
    split: str = "val",
    *,
    mask_ratio: float = 0.3,
    span_concentration: float = 0.5,
    seed: int = 42,
    top_k: int = 10,
    beam_width: int = 10,
    beam_depth: int = 6,
    limit: int | None = None,  # smoke runs: first N sentences
    out_root: str = "results/runs",
    push_to_hub: bool = False,
    results_repo: str = DEFAULT_RESULTS_REPO,
    hf_token: str | None = None,
    device: str | None = None,  # None → cuda → mps → cpu
) -> str:  # returns run_dir
    from transformers import AutoModelForMaskedLM
    import transformers

    started = time.time()
    created = datetime.now(timezone.utc)

    device = resolve_device(device)
    print(f"device: {device}")

    tokenizer = load_tokenizer(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.eval()

    family = "char" if emits_whitespace_tokens(tokenizer) else "wordpiece"
    origin = _resolve_revision(model_id, hf_token)
    print(f"model: {model_id} ({family}, revision {origin['revision']})")

    # ---- data ----
    fingerprint = dataset_fingerprint(dataset_path)

    df = pd.read_excel(dataset_path).drop(columns=["Unnamed: 0"], errors="ignore")
    df = df[df["split"] == split].reset_index(drop=True)

    if limit is not None:
        df = df.head(limit)

    if df.empty:
        raise ValueError(f"no rows with split == {split!r} in {dataset_path}")

    # ---- identity ----
    protocol = protocol_id(
        fingerprint,
        split,
        "PercentageContentMaskingPolicy",
        mask_ratio,
        span_concentration,
        seed,
    )
    decode = decode_id("MultiSpanPredictionPolicy", top_k, beam_width, beam_depth)
    run_id = make_run_id(model_id, protocol, decode, created)

    run_dir = Path(out_root) / run_id
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        raise RuntimeError(
            f"{metrics_path} already exists and is non-empty -- refusing to clobber "
            f"run '{run_id}'. Delete that run dir, or let a new timestamp pick a new id."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id: {run_id}")

    # ---- inference ----
    policy = PercentageContentMaskingPolicy(
        tokenizer,
        mask_ratio=mask_ratio,
        span_concentration=span_concentration,
        seed=seed,
    )
    predictor = MultiSpanPredictionPolicy(
        model,
        tokenizer,
        top_k=top_k,
        beam_width=beam_width,
        beam_depth=beam_depth,
        device=device,
    )
    evaluator = Evaluator()

    try:
        from tqdm.auto import tqdm
    except ImportError:

        def tqdm(it, **kwargs):
            return it

    sentence_records, span_records = [], []

    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as out:
        for row in tqdm(df.to_dict("records"), desc=run_id[:40]):
            sentence = str(row["sentence"])

            uid = sentence_uid(row)
            sha = text_sha(sentence)

            # Masking is a pure function of (seed, location, text): identical text
            # at the same location is masked identically in every run, forever.
            example = policy.generate(sentence, uid=f"{uid}|{sha}")

            if example is None:
                continue

            predictions = predictor.predict(example)
            gold_spans = evaluator._gold_spans(example, predictions)

            sentence_record = {
                "record": "sentence",
                "sentence_uid": uid,
                "text_sha": sha,
                "scroll": row["scroll"],
                "fragment": str(row["fragment"]),
                "line_start": int(row["line_start"]),
                "line_end": int(row["line_end"]),
                "sentence": sentence,
                "masked_sentence": example.masked_sentence,
                "masked_word_indices": list(example.masked_word_indices),
                "gold_words": list(example.gold_words),
                "n_spans": len(predictions),
            }
            sentence_records.append(sentence_record)
            out.write(json.dumps(sentence_record, ensure_ascii=False) + "\n")

            for gold, span in zip(gold_spans, predictions):
                span_record = {
                    "record": "span",
                    "sentence_uid": uid,
                    "text_sha": sha,
                    "span_index": span["span_index"],
                    "span_length": span["span_length"],
                    "mode": span["mode"],
                    "word_indices": gold["word_indices"],
                    "gold_words": gold["gold_words"],
                    "gold_text": gold["gold_text"],
                    "gold_token_ids": gold["gold_token_ids"],
                    "candidates": span["topk_phrases"],
                }
                span_records.append(span_record)
                out.write(json.dumps(span_record, ensure_ascii=False) + "\n")

    # ---- derived files ----
    word_df, metrics, counts = score_run_dir(run_dir)

    manifest = {
        "run_id": run_id,
        "protocol_id": protocol,
        "decode_id": decode,
        "created_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "id": model_id,
            "revision": origin["revision"],
            "source": origin["source"],
            "tokenizer_family": family,
            "vocab_size": int(getattr(model.config, "vocab_size", 0)) or None,
            "n_params": int(sum(p.numel() for p in model.parameters())),
        },
        "dataset": {
            "path": Path(dataset_path).name,
            "fingerprint": fingerprint,
            "split": split,
            "n_sentences": int(len(df)),
            "limit": limit,
        },
        "masking": {
            "policy": "PercentageContentMaskingPolicy",
            "mask_ratio": mask_ratio,
            "span_concentration": span_concentration,
            "seed": seed,
            "per_sentence_seed": True,
        },
        "decoding": {
            "policy": "MultiSpanPredictionPolicy",
            "top_k": top_k,
            "beam_width": beam_width,
            "beam_depth": beam_depth,
        },
        "env": {
            "device": device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "scorer_version": SCORER_VERSION,
        "counts": counts,
        "duration_sec": round(time.time() - started, 1),
    }
    _write_json(run_dir / "manifest.json", manifest)

    overall = metrics["overall"]
    print(f"\n{run_id}")
    print(
        f"  words scored: {counts['aligned_words']} / {counts['masked_words']}"
        f"  ({counts['unaligned_words']} unaligned)"
    )
    for name in ["char_sim_top1", "hit_at_1", "hit_at_10", "mrr"]:
        entry = overall.get(name) or {}
        if entry.get("mean") is not None:
            print(
                f"  {name:<16} {entry['mean']:.4f} "
                f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"
            )

    if push_to_hub:
        _push_run(run_dir, run_id, results_repo, hf_token)

    return str(run_dir)


def _push_run(run_dir, run_id, results_repo, hf_token):
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)

    api.create_repo(results_repo, repo_type="dataset", private=True, exist_ok=True)

    # create_repo(private=True, exist_ok=True) does NOT flip an already-public repo
    # to private, and these files carry corpus text. Verify rather than assume --
    # same guard as tuning/finetune_msbert.ipynb uses for the weights.
    if not api.repo_info(results_repo, repo_type="dataset").private:
        raise RuntimeError(
            f"dataset repo {results_repo} exists and is PUBLIC. Refusing to push. "
            "Flip it to private in the repo settings, then re-run."
        )

    api.upload_folder(
        repo_id=results_repo,
        folder_path=str(run_dir),
        path_in_repo=f"runs/{run_id}",
        repo_type="dataset",
        commit_message=f"eval run {run_id}",
        ignore_patterns=["**/.ipynb_checkpoints/*", "**/__pycache__/*"],
    )

    print(
        f"\npushed -> https://huggingface.co/datasets/{results_repo}/tree/main/runs/{run_id}"
    )
