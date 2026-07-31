"""Benchmark semantic embeddings against the transparent TF-IDF baseline.

The benchmark uses the same reconstruction-free DSS queries, external corpus,
book-diversified ranking, and manuscript-level Pesher positive control as
``cross_corpus_connections.py``.  It reports three independently specified
retrievers:

* TF-IDF: 80% word percentile + 20% character percentile;
* embedding: MiqraBERT cosine-similarity percentile;
* fixed hybrid: 50% TF-IDF percentile + 50% embedding percentile.

No weights are fitted on the positive control.  By default the Hugging Face
model must already exist in the local cache and remote custom code is disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.cross_corpus_connections import (
    DEFAULT_TARGETS,
    DEFAULT_TF_ROOT,
    QUMRAN_EXTRA_BOOKS,
    ROOT,
    SOURCE_WEIGHTS,
    Passage,
    _tfidf_similarities,
    evaluate_pesher_source_control,
    load_preserved_dss_queries,
    load_tf_passages,
    row_percentiles,
    sha256_files,
    source_receipt,
)

DEFAULT_MODEL = "davidmsmiley/MiqraBERT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF, MiqraBERT, and fixed hybrid DSS retrieval."
    )
    parser.add_argument("--dss-csv", type=Path, default=ROOT / "dss_chunks.csv")
    parser.add_argument(
        "--dss-preserved-jsonl",
        type=Path,
        default=ROOT / "data" / "derived" / "preserved_nonbib_chunks.jsonl",
    )
    parser.add_argument(
        "--bhsa-path",
        type=Path,
        default=DEFAULT_TF_ROOT / "bhsa" / "tf" / "2021",
    )
    parser.add_argument(
        "--extrabiblical-path",
        type=Path,
        default=DEFAULT_TF_ROOT / "extrabiblical" / "tf" / "0.2",
    )
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--chunk-words", type=int, default=100)
    parser.add_argument("--overlap-words", type=int, default=15)
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face network downloads. Off by default.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "analysis" / "cache" / "cross_corpus_embeddings",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT
        / "analysis"
        / "reports"
        / "cross_corpus_embedding_benchmark.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT
        / "analysis"
        / "reports"
        / "CROSS_CORPUS_EMBEDDING_BENCHMARK.md",
    )
    return parser.parse_args()


def choose_device(requested: str, torch_module) -> str:
    if requested != "auto":
        if requested == "mps" and not torch_module.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def passage_fingerprint(passages: Sequence[Passage]) -> str:
    digest = hashlib.sha256()
    for passage in passages:
        digest.update(passage.passage_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(passage.text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _cache_path(
    cache_dir: Path,
    *,
    model: str,
    max_length: int,
    passages: Sequence[Passage],
) -> Path:
    identity = hashlib.sha256(
        f"{model}\0{max_length}\0{passage_fingerprint(passages)}".encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{identity}.npy"


def encode_passages(
    passages: Sequence[Passage],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    allow_download: bool,
    cache_dir: Path,
) -> tuple[np.ndarray, dict]:
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive")

    cache_path = _cache_path(
        cache_dir,
        model=model_name,
        max_length=max_length,
        passages=passages,
    )
    if cache_path.is_file():
        matrix = np.load(cache_path, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(passages):
            raise ValueError(f"invalid embedding cache shape in {cache_path}")
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_name,
            local_files_only=not allow_download,
            trust_remote_code=False,
        )
        return matrix, {
            "model": model_name,
            "model_commit": getattr(config, "_commit_hash", None),
            "cache": str(cache_path),
            "cache_hit": True,
            "device": "cached",
            "max_length": max_length,
            "embedding_dimensions": int(matrix.shape[1]),
        }

    import torch
    from transformers import AutoModel, AutoTokenizer

    selected_device = choose_device(device, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=not allow_download,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        model_name,
        local_files_only=not allow_download,
        trust_remote_code=False,
    )
    model.eval()
    model.to(selected_device)

    batches: list[np.ndarray] = []
    texts = [passage.text for passage in passages]
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(selected_device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        batches.append(pooled.detach().cpu().numpy().astype(np.float32))

    matrix = np.concatenate(batches, axis=0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix, allow_pickle=False)
    commit = getattr(model.config, "_commit_hash", None)
    model.to("cpu")
    del model
    if selected_device == "mps":
        torch.mps.empty_cache()
    elif selected_device == "cuda":
        torch.cuda.empty_cache()
    return matrix, {
        "model": model_name,
        "model_commit": commit,
        "cache": str(cache_path),
        "cache_hit": False,
        "device": selected_device,
        "max_length": max_length,
        "embedding_dimensions": int(matrix.shape[1]),
    }


def diversified_details(
    queries: Sequence[Passage],
    external: Sequence[Passage],
    scores: np.ndarray,
    *,
    top_k: int,
) -> list[dict]:
    if scores.shape != (len(queries), len(external)):
        raise ValueError("score matrix does not match query/external passage counts")
    detailed = []
    for query_index, query in enumerate(queries):
        best_by_book: dict[tuple[str, str], int] = {}
        for external_index, candidate in enumerate(external):
            key = (candidate.corpus, candidate.book)
            previous = best_by_book.get(key)
            if previous is None or scores[query_index, external_index] > scores[
                query_index, previous
            ]:
                best_by_book[key] = external_index
        ranked = sorted(
            best_by_book.values(),
            key=lambda index: scores[query_index, index],
            reverse=True,
        )[:top_k]
        detailed.append(
            {
                "query": asdict(query),
                "matches": [
                    {
                        "rank": rank,
                        "corpus": external[index].corpus,
                        "book": external[index].book,
                        "reference": external[index].reference,
                        "passage_id": external[index].passage_id,
                        "score": round(float(scores[query_index, index]), 6),
                        "text": external[index].text,
                    }
                    for rank, index in enumerate(ranked, start=1)
                ],
            }
        )
    return detailed


def method_result(
    name: str,
    queries: Sequence[Passage],
    external: Sequence[Passage],
    scores: np.ndarray,
    *,
    top_k: int,
) -> dict:
    details = diversified_details(queries, external, scores, top_k=top_k)
    return {
        "method": name,
        "positive_control": evaluate_pesher_source_control(details),
        "passages": details,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Cross-corpus embedding retrieval benchmark",
        "",
        "Status: **exploratory calibration**. The embedding model is used as a",
        "retriever, not as proof of borrowing, authorship, or source direction.",
        "",
        "## Fair comparison",
        "",
        "- All methods use the same reconstruction-free DSS passages.",
        "- All methods search the same external passage windows.",
        "- Rankings keep at most one result per external book.",
        "- No retrieval weights were fitted on the positive control.",
        "- The control is macro-averaged and permuted at DSS manuscript level.",
        "",
        "## Known-source Pesher control",
        "",
        "| Retrieval method | Manuscripts | Passages | Macro Top-1 | Macro Top-3 | Top-1 p | Top-3 p |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["methods"].items():
        control = result["positive_control"]
        lines.append(
            f"| {name} | {control['manuscripts']} | {control['passages']} | "
            f"{100 * control['macro_top1_recovery']:.1f}% | "
            f"{100 * control['macro_top3_recovery']:.1f}% | "
            f"{control['top1_permutation_p']:.4g} | "
            f"{control['top3_permutation_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "TF-IDF remains the primary interpretable screen unless embeddings or the",
            "pre-specified 50/50 hybrid improve known-source recovery. Even if they do,",
            "the channels remain visible separately during passage adjudication.",
            "",
            "MiqraBERT was trained for Biblical Hebrew parallel retrieval. Its result",
            "is a useful semantic sensitivity test, but its Bible-domain training can",
            "favor biblical sources and does not validate rabbinic or epigraphic links.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    targets = None
    if args.targets.strip().lower() != "all":
        targets = {
            value.strip() for value in args.targets.split(",") if value.strip()
        }

    queries = load_preserved_dss_queries(
        args.dss_preserved_jsonl,
        metadata_csv=args.dss_csv,
        targets=targets,
        min_words=args.min_words,
    )
    external = load_tf_passages(
        args.bhsa_path,
        corpus="bhsa",
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_words=args.min_words,
    )
    external.extend(
        load_tf_passages(
            args.extrabiblical_path,
            corpus="extrabiblical",
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            min_words=args.min_words,
            excluded_books=QUMRAN_EXTRA_BOOKS,
        )
    )

    word_scores, char_scores = _tfidf_similarities(queries, external)
    tfidf_scores = (
        SOURCE_WEIGHTS["lexical"] * row_percentiles(word_scores)
        + SOURCE_WEIGHTS["orthographic"] * row_percentiles(char_scores)
    )
    query_embeddings, query_model = encode_passages(
        queries,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        allow_download=args.allow_download,
        cache_dir=args.cache_dir,
    )
    external_embeddings, external_model = encode_passages(
        external,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        allow_download=args.allow_download,
        cache_dir=args.cache_dir,
    )
    embedding_scores = query_embeddings @ external_embeddings.T
    hybrid_scores = (
        0.5 * row_percentiles(tfidf_scores)
        + 0.5 * row_percentiles(embedding_scores)
    )

    report = {
        "status": "exploratory_retrieval_calibration",
        "counts": {
            "query_passages": len(queries),
            "external_passages": len(external),
        },
        "methods": {
            "tfidf_80_word_20_char": method_result(
                "tfidf_80_word_20_char",
                queries,
                external,
                tfidf_scores,
                top_k=args.top_k,
            ),
            "miqrabert_embedding": method_result(
                "miqrabert_embedding",
                queries,
                external,
                embedding_scores,
                top_k=args.top_k,
            ),
            "fixed_50_tfidf_50_embedding": method_result(
                "fixed_50_tfidf_50_embedding",
                queries,
                external,
                hybrid_scores,
                top_k=args.top_k,
            ),
        },
        "protocol": {
            "targets": sorted(targets) if targets is not None else "all",
            "chunk_words": args.chunk_words,
            "overlap_words": args.overlap_words,
            "min_words": args.min_words,
            "top_k": args.top_k,
            "tfidf_weights": SOURCE_WEIGHTS,
            "hybrid_weights": {"tfidf": 0.5, "embedding": 0.5},
            "query_model": query_model,
            "external_model": external_model,
            "dss_query_source": str(args.dss_preserved_jsonl),
            "dss_query_source_sha256": sha256_files(
                args.dss_preserved_jsonl.parent,
                [args.dss_preserved_jsonl.name],
            ),
            "bhsa": source_receipt(args.bhsa_path),
            "extrabiblical": source_receipt(args.extrabiblical_path),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_markdown}")
    for name, result in report["methods"].items():
        control = result["positive_control"]
        print(
            f"{name}: Top-1={control['macro_top1_recovery']:.4f}, "
            f"Top-3={control['macro_top3_recovery']:.4f}"
        )


if __name__ == "__main__":
    main()
