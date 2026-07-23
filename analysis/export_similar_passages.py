"""Export similar fit-corpus passages for each evaluated case.

The retrieval index is intentionally restricted to the `fit` partition. For the
held-out benchmark this avoids retrieving text from the held-out scrolls.
"""
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.clitic_join import join_likely_clitics
from utils.composition_lookup import composition_group_for_scroll
from utils.dss_split import load_partition

CASES_CSV = Path(os.environ.get(
    "CASES_CSV",
    ROOT / "analysis" / "reports" / "full_single_word_cases_refined_hebrew_only.csv",
))
OUT_JSON = Path(os.environ.get(
    "OUT_JSON",
    ROOT / "analysis" / "reports" / "similar_passages_fit_for_full_single_word_cases.json",
))
TOP_K = int(os.environ.get("TOP_K", "5"))
RETRIEVAL_PARTITION = os.environ.get("RETRIEVAL_PARTITION", "fit")
HEB = re.compile(r"[\u05d0-\u05ea]+")


def tokens(text: str) -> list[str]:
    return [token for token in HEB.findall(text.replace("⬚⬚⬚", " ")) if len(token) > 1]


def display_text(text: str) -> str:
    joined, _ = join_likely_clitics(text)
    return joined


def load_docs():
    rows = load_partition(RETRIEVAL_PARTITION)
    docs = []
    df = Counter()
    for idx, row in enumerate(rows):
        text = row["text"].strip()
        tok = tokens(text)
        tf = Counter(tok)
        for term in tf:
            df[term] += 1
        docs.append({
            "doc_id": f"{RETRIEVAL_PARTITION}-{idx + 1}",
            "book": row.get("book", ""),
            "sentence_path": row.get("sentence_path", ""),
            "composition": row.get("composition", ""),
            "composition_group": composition_group_for_scroll(row.get("book", "")),
            "genre": row.get("genre", ""),
            "section": row.get("section", ""),
            "text": text,
            "display_text": display_text(text),
            "tf": tf,
            "length": max(1, len(tok)),
        })
    n_docs = max(1, len(docs))
    idf = {term: math.log((n_docs + 1) / (count + 0.5)) + 1.0 for term, count in df.items()}
    for doc in docs:
        norm = 0.0
        weights = {}
        for term, count in doc["tf"].items():
            weight = (1.0 + math.log(count)) * idf.get(term, 0.0)
            weights[term] = weight
            norm += weight * weight
        doc["weights"] = weights
        doc["norm"] = math.sqrt(norm) or 1.0
    return docs, idf


def query_weights(text: str, idf: dict[str, float]):
    tf = Counter(tokens(text))
    weights = {}
    norm = 0.0
    for term, count in tf.items():
        weight = (1.0 + math.log(count)) * idf.get(term, 0.0)
        if weight <= 0:
            continue
        weights[term] = weight
        norm += weight * weight
    return weights, math.sqrt(norm) or 1.0


def score_docs(query: dict[str, float], query_norm: float, docs: list[dict]):
    scored = []
    q_terms = set(query)
    for doc in docs:
        overlap = q_terms & doc["weights"].keys()
        if not overlap:
            continue
        dot = sum(query[term] * doc["weights"][term] for term in overlap)
        score = dot / (query_norm * doc["norm"])
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:TOP_K]


def contains_word(text: str, word: str) -> bool:
    if not word:
        return False
    return word in tokens(text)


def candidate_hits(text: str, row: dict):
    hits = []
    for i in range(1, 6):
        candidate = row.get(f"model_top{i}", "")
        if contains_word(text, candidate):
            hits.append(candidate)
    return hits


def export():
    docs, idf = load_docs()
    with CASES_CSV.open() as fh:
        cases = list(csv.DictReader(fh))

    payload = {}
    for idx, row in enumerate(cases, start=1):
        query_text = row["context_for_reading"]
        case_group = composition_group_for_scroll(row.get("scroll", ""))
        q_weights, q_norm = query_weights(query_text, idf)
        passages = []
        for rank, (score, doc) in enumerate(score_docs(q_weights, q_norm, docs), start=1):
            text = doc["text"]
            passages.append({
                "rank": rank,
                "score": round(score, 4),
                "book": doc["book"],
                "sentence_path": doc["sentence_path"],
                "composition": doc["composition"],
                "same_composition": doc["composition_group"] == case_group,
                "genre": doc["genre"],
                "section": doc["section"],
                "source_scope": RETRIEVAL_PARTITION,
                "text": doc["display_text"],
                "gold_present": contains_word(text, row["target_word"]),
                "top1_present": contains_word(text, row["model_top1"]),
                "candidate_hits": candidate_hits(text, row),
            })
        payload[str(row["row_id"])] = passages
        if idx % 500 == 0:
            print(f"retrieved {idx}/{len(cases)}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"cases={len(cases)} docs={len(docs)} top_k={TOP_K} partition={RETRIEVAL_PARTITION}")


if __name__ == "__main__":
    export()
