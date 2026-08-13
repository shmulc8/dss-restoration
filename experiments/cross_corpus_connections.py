"""Find interpretable cross-corpus connections for preserved DSS passages.

This is an exploratory source-connection screen, not an authorship classifier.
It deliberately keeps three signals separate:

* lexical similarity: shared words and short phrases;
* orthographic similarity: character n-grams;
* style affinity: function words and surface distributions that do not require
  a modern Hebrew tagger.

The default external shelf contains the ETCBC BHSA Hebrew Bible and the
non-Qumran books in ETCBC/extrabiblical.  Source corpora are read from a local
Text-Fabric checkout; this script never downloads data implicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tf.fabric import Fabric

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HEBREW = re.compile(r"[\u05d0-\u05ea]+")
DEFAULT_TF_ROOT = Path.home() / "text-fabric-data" / "github" / "ETCBC"
DEFAULT_TARGETS = (
    "4QMMT",
    "Book_of_Jubilees",
    "Hodayot",
    "Instruction",
    "Pesharim",
    "Temple Scroll",
    "rewritten pentateuch",
)
SOURCE_WEIGHTS = {"lexical": 0.80, "orthographic": 0.20, "style": 0.00}
QUMRAN_EXTRA_BOOKS = {"B_1QHa", "B_1QM", "B_1QS"}
EPIGRAPHIC_BOOKS = {
    "Ajrud",
    "Arad",
    "Balaam",
    "Ketef_Hinnom",
    "Lachish",
    "Mesa",
    "Mesad_Hashavyahu",
    "Siloam",
}
RABBINIC_BOOKS = {"Pirqe", "Shirata"}
PESHER_SOURCE_BOOKS = {
    "1QpHab": "Habakuk",
    "1Q14": "Micha",
    "1Q15": "Zephania",
    "1Q16": "Psalmi",
    "4Q161": "Jesaia",
    "4Q162": "Jesaia",
    "4Q163": "Jesaia",
    "4Q164": "Jesaia",
    "4Q165": "Jesaia",
    "4Q166": "Hosea",
    "4Q167": "Hosea",
    "4Q168": "Micha",
    "4Q169": "Nahum",
    "4Q171": "Psalmi",
}
FUNCTION_WORDS = {
    "או",
    "אז",
    "אך",
    "אל",
    "אלה",
    "אם",
    "אשר",
    "את",
    "גם",
    "הוא",
    "היא",
    "הם",
    "המה",
    "הנה",
    "זה",
    "זאת",
    "כי",
    "כל",
    "כן",
    "לא",
    "לו",
    "מה",
    "מי",
    "מן",
    "נא",
    "עד",
    "עוד",
    "על",
    "עם",
    "פה",
    "רק",
    "שם",
}


@dataclass(frozen=True)
class Passage:
    passage_id: str
    corpus: str
    book: str
    reference: str
    text: str
    tokens: tuple[str, ...]
    composition: str = ""
    genre: str = ""
    section: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen preserved DSS passages against external Hebrew corpora."
    )
    parser.add_argument("--dss-csv", type=Path, default=ROOT / "dss_chunks.csv")
    parser.add_argument(
        "--dss-preserved-jsonl",
        type=Path,
        default=ROOT / "curation" / "derived" / "preserved_nonbib_chunks.jsonl",
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
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated composition labels. Use 'all' for all non-biblical rows.",
    )
    parser.add_argument("--chunk-words", type=int, default=100)
    parser.add_argument("--overlap-words", type=int, default=15)
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "comparison" / "reports" / "cross_corpus_connections.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "comparison" / "reports" / "CROSS_CORPUS_CONNECTIONS.md",
    )
    return parser.parse_args()


def hebrew_tokens(text: str) -> list[str]:
    return HEBREW.findall(text)


def normalized_text(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def fixed_windows(
    tokens: Sequence[str],
    refs: Sequence[str],
    *,
    chunk_words: int,
    overlap_words: int,
    min_words: int,
) -> Iterable[tuple[int, int, tuple[str, ...], str]]:
    if len(tokens) != len(refs):
        raise ValueError("tokens and refs must have identical lengths")
    if chunk_words < 1 or min_words < 1:
        raise ValueError("chunk_words and min_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    stride = chunk_words - overlap_words
    for start in range(0, len(tokens), stride):
        end = min(start + chunk_words, len(tokens))
        if end - start < min_words:
            break
        start_ref = refs[start]
        end_ref = refs[end - 1]
        reference = start_ref if start_ref == end_ref else f"{start_ref}–{end_ref}"
        yield start, end, tuple(tokens[start:end]), reference
        if end == len(tokens):
            break


def _tf_reference(api, word: int, book_name: str) -> str:
    verses = api.L.u(word, "verse")
    if not verses:
        return book_name
    verse = verses[0]
    chapter = api.F.chapter.v(verse)
    number = api.F.verse.v(verse)
    if chapter is None or number is None:
        return book_name
    return f"{book_name} {chapter}:{number}"


def load_tf_passages(
    path: Path,
    *,
    corpus: str,
    chunk_words: int,
    overlap_words: int,
    min_words: int,
    excluded_books: set[str] | None = None,
) -> list[Passage]:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Text-Fabric source not found: {path}. "
            "Install the corpus explicitly before running this screen."
        )
    tf = Fabric(locations=str(path), silent="deep")
    api = tf.load("book chapter verse g_cons_utf8", silent="deep")
    if api is None:
        raise RuntimeError(f"could not load Text-Fabric source: {path}")

    excluded_books = excluded_books or set()
    passages: list[Passage] = []
    for book_node in api.F.otype.s("book"):
        book = str(api.F.book.v(book_node))
        if book in excluded_books:
            continue
        tokens: list[str] = []
        refs: list[str] = []
        for word in api.L.d(book_node, "word"):
            pieces = hebrew_tokens(str(api.F.g_cons_utf8.v(word) or ""))
            reference = _tf_reference(api, word, book)
            tokens.extend(pieces)
            refs.extend([reference] * len(pieces))
        for start, end, window, reference in fixed_windows(
            tokens,
            refs,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            min_words=min_words,
        ):
            passages.append(
                Passage(
                    passage_id=f"{corpus}:{book}:{start}-{end}",
                    corpus=corpus,
                    book=book,
                    reference=reference,
                    text=normalized_text(window),
                    tokens=window,
                )
            )
    return passages


def load_dss_queries(
    path: Path,
    *,
    targets: set[str] | None,
    min_words: int,
) -> list[Passage]:
    if not path.is_file():
        raise FileNotFoundError(f"DSS chunk file not found: {path}")
    passages: list[Passage] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            if row.get("bib") != "nonbib":
                continue
            composition = (row.get("composition") or "").strip()
            if targets is not None and composition not in targets:
                continue
            tokens = tuple(hebrew_tokens(row.get("text") or ""))
            if len(tokens) < min_words:
                continue
            book = (row.get("book") or "").strip()
            reference = (row.get("sentence_path") or book).strip()
            passages.append(
                Passage(
                    passage_id=f"dss:{row_index}:{reference}",
                    corpus="dss",
                    book=book,
                    reference=reference,
                    text=normalized_text(tokens),
                    tokens=tokens,
                    composition=composition,
                    genre=(row.get("genre") or "").strip(),
                    section=(row.get("section") or "").strip(),
                )
            )
    return passages


def load_preserved_dss_queries(
    path: Path,
    *,
    metadata_csv: Path,
    targets: set[str] | None,
    min_words: int,
) -> list[Passage]:
    """Load reconstruction-free queries and attach non-textual labels by scroll."""
    metadata: dict[str, dict[str, str]] = {}
    candidates: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {
            "composition": Counter(),
            "genre": Counter(),
            "section": Counter(),
        }
    )
    with metadata_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            book = (row.get("book") or "").strip()
            if not book:
                continue
            for field in ("composition", "genre", "section"):
                value = (row.get(field) or "").strip()
                if value:
                    candidates[book][field][value] += 1
    for book, fields in candidates.items():
        metadata[book] = {
            field: counts.most_common(1)[0][0] if counts else ""
            for field, counts in fields.items()
        }

    passages: list[Passage] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            book = str(row.get("scroll") or "").strip()
            labels = metadata.get(book, {})
            composition = labels.get("composition", "")
            if targets is not None and composition not in targets:
                continue
            # HEBREW excludes the literal <GAP> marker by construction.
            tokens = tuple(hebrew_tokens(str(row.get("text") or "")))
            if len(tokens) < min_words:
                continue
            chunk_index = int(row.get("chunk_index", 0))
            passages.append(
                Passage(
                    passage_id=f"dss-preserved:{book}:{chunk_index}",
                    corpus="dss",
                    book=book,
                    reference=f"{book} preserved chunk {chunk_index}",
                    text=normalized_text(tokens),
                    tokens=tokens,
                    composition=composition,
                    genre=labels.get("genre", ""),
                    section=labels.get("section", ""),
                )
            )
    return passages


def style_features(tokens: Sequence[str]) -> dict[str, float]:
    if not tokens:
        return {}
    n = len(tokens)
    counts = Counter(tokens)
    result: dict[str, float] = {
        "type_token_ratio": len(counts) / n,
        "hapax_ratio": sum(value == 1 for value in counts.values()) / n,
        "mean_word_length": sum(map(len, tokens)) / n / 10.0,
    }
    for word in FUNCTION_WORDS:
        result[f"fw:{word}"] = counts[word] / n
    for character in "אבגדהוזחטיכלמנסעפצקרשת":
        result[f"prefix:{character}"] = (
            sum(token.startswith(character) for token in tokens) / n
        )
        result[f"suffix:{character}"] = (
            sum(token.endswith(character) for token in tokens) / n
        )
    for length in range(1, 11):
        result[f"length:{length}"] = (
            sum(min(len(token), 10) == length for token in tokens) / n
        )
    return result


def longest_shared_ngram(
    left: Sequence[str], right: Sequence[str], cap: int = 12
) -> int:
    if not left or not right:
        return 0
    right_positions: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(right):
        right_positions[token].append(index)
    best = 0
    for left_index, token in enumerate(left):
        for right_index in right_positions.get(token, []):
            length = 0
            while (
                length < cap
                and left_index + length < len(left)
                and right_index + length < len(right)
                and left[left_index + length] == right[right_index + length]
            ):
                length += 1
            best = max(best, length)
            if best == cap:
                return best
    return best


def percentile(score: float, population: np.ndarray) -> float:
    return float(np.mean(population <= score))


def row_percentiles(matrix: np.ndarray) -> np.ndarray:
    """Convert each query's scores to [0, 1] ranks on its external shelf."""
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("matrix must contain at least two candidates per query")
    order = np.argsort(matrix, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_numbers = np.arange(matrix.shape[0])[:, None]
    ranks[row_numbers, order] = np.arange(matrix.shape[1], dtype=float)
    return ranks / (matrix.shape[1] - 1)


def relation_type(
    *,
    lexical_percentile: float,
    style_percentile: float,
    shared_ngram_words: int,
) -> str:
    if (
        lexical_percentile >= 0.99
        and style_percentile >= 0.95
        and shared_ngram_words >= 3
    ):
        return "lexical_and_style_connection"
    if lexical_percentile >= 0.995 and shared_ngram_words >= 3:
        return "lexical_parallel_candidate"
    if style_percentile >= 0.995 and lexical_percentile < 0.90:
        return "style_affinity_without_lexical_overlap"
    return "nearest_neighbor_only"


def source_kind(corpus: str, book: str) -> str:
    if corpus == "bhsa":
        return "hebrew_bible"
    if book in EPIGRAPHIC_BOOKS:
        return "epigraphic_comparator"
    if book == "Pirqe":
        return "rabbinic_comparator_pirqe_avot"
    if book == "Shirata":
        return "rabbinic_comparator_mekhilta_shirata"
    return "external_comparator"


def interpretation_bucket(*, corpus: str, book: str, relation: str) -> str:
    kind = source_kind(corpus, book)
    if relation in {
        "lexical_parallel_candidate",
        "lexical_and_style_connection",
    }:
        if kind == "hebrew_bible":
            return "scriptural_parallel_candidate"
        if kind.startswith("rabbinic_comparator"):
            return "later_reception_or_shared_scriptural_source"
        if kind == "epigraphic_comparator":
            return "epigraphic_lexical_affinity_candidate"
    if relation == "style_affinity_without_lexical_overlap":
        return "surface_style_affinity_only"
    return "nearest_comparator_only"


def _tfidf_similarities(
    queries: Sequence[Passage], external: Sequence[Passage]
) -> tuple[np.ndarray, np.ndarray]:
    external_texts = [item.text for item in external]
    query_texts = [item.text for item in queries]

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
    )
    external_word_matrix = word_vectorizer.fit_transform(external_texts)
    query_word_matrix = word_vectorizer.transform(query_texts)
    word_scores = cosine_similarity(query_word_matrix, external_word_matrix)

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.995,
        sublinear_tf=True,
        norm="l2",
    )
    external_char_matrix = char_vectorizer.fit_transform(external_texts)
    query_char_matrix = char_vectorizer.transform(query_texts)
    char_scores = cosine_similarity(query_char_matrix, external_char_matrix)
    return word_scores, char_scores


def _style_similarities(
    queries: Sequence[Passage], external: Sequence[Passage]
) -> np.ndarray:
    vectorizer = DictVectorizer(sparse=True)
    external_matrix = normalize(
        vectorizer.fit_transform([style_features(item.tokens) for item in external]),
        norm="l2",
    )
    query_matrix = normalize(
        vectorizer.transform([style_features(item.tokens) for item in queries]),
        norm="l2",
    )
    return cosine_similarity(query_matrix, external_matrix)


def analyze_connections(
    queries: Sequence[Passage],
    external: Sequence[Passage],
    *,
    top_k: int,
) -> dict:
    if not queries:
        raise ValueError("no DSS query passages matched the requested targets")
    if not external:
        raise ValueError("no external passages were loaded")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    word_scores, char_scores = _tfidf_similarities(queries, external)
    style_scores = _style_similarities(queries, external)
    word_percentiles = row_percentiles(word_scores)
    char_percentiles = row_percentiles(char_scores)
    style_percentiles = row_percentiles(style_scores)
    combined_scores = (
        SOURCE_WEIGHTS["lexical"] * word_percentiles
        + SOURCE_WEIGHTS["orthographic"] * char_percentiles
        + SOURCE_WEIGHTS["style"] * style_percentiles
    )

    detailed = []
    composition_books: dict[str, Counter] = defaultdict(Counter)
    composition_lexical_sums: dict[str, Counter] = defaultdict(Counter)
    composition_strong: dict[str, Counter] = defaultdict(Counter)
    relation_counts: Counter = Counter()
    for query_index, query in enumerate(queries):
        best_by_book: dict[tuple[str, str], int] = {}
        for external_index, candidate in enumerate(external):
            key = (candidate.corpus, candidate.book)
            previous = best_by_book.get(key)
            if (
                previous is None
                or combined_scores[query_index, external_index]
                > combined_scores[query_index, previous]
            ):
                best_by_book[key] = external_index
        ranked = sorted(
            best_by_book.values(),
            key=lambda index: combined_scores[query_index, index],
            reverse=True,
        )[:top_k]

        matches = []
        for rank, external_index in enumerate(ranked, start=1):
            candidate = external[external_index]
            shared = longest_shared_ngram(query.tokens, candidate.tokens)
            lexical_pct = float(word_percentiles[query_index, external_index])
            style_pct = float(style_percentiles[query_index, external_index])
            relation = relation_type(
                lexical_percentile=lexical_pct,
                style_percentile=style_pct,
                shared_ngram_words=shared,
            )
            interpretation = interpretation_bucket(
                corpus=candidate.corpus,
                book=candidate.book,
                relation=relation,
            )
            relation_counts[relation] += 1
            if rank <= 3:
                source_book = f"{candidate.corpus}:{candidate.book}"
                composition_books[query.composition][source_book] += 1
                composition_lexical_sums[query.composition][source_book] += float(
                    word_scores[query_index, external_index]
                )
                if interpretation in {
                    "scriptural_parallel_candidate",
                    "later_reception_or_shared_scriptural_source",
                    "epigraphic_lexical_affinity_candidate",
                }:
                    composition_strong[query.composition][source_book] += 1
            matches.append(
                {
                    "rank": rank,
                    "passage_id": candidate.passage_id,
                    "corpus": candidate.corpus,
                    "book": candidate.book,
                    "source_kind": source_kind(candidate.corpus, candidate.book),
                    "reference": candidate.reference,
                    "combined_score": round(
                        float(combined_scores[query_index, external_index]), 4
                    ),
                    "lexical_score": round(
                        float(word_scores[query_index, external_index]), 4
                    ),
                    "orthographic_score": round(
                        float(char_scores[query_index, external_index]), 4
                    ),
                    "orthographic_percentile": round(
                        float(char_percentiles[query_index, external_index]), 4
                    ),
                    "style_score": round(
                        float(style_scores[query_index, external_index]), 4
                    ),
                    "lexical_percentile": round(lexical_pct, 4),
                    "style_percentile": round(style_pct, 4),
                    "shared_ngram_words": shared,
                    "relation_type": relation,
                    "interpretation_bucket": interpretation,
                    "text": candidate.text,
                }
            )
        detailed.append({"query": asdict(query), "matches": matches})

    summaries = {}
    query_counts = Counter(query.composition for query in queries)
    for composition, counts in composition_books.items():
        rows = []
        for source_book, support in counts.most_common():
            rows.append(
                {
                    "source_book": source_book,
                    "top3_support": support,
                    "query_passages": query_counts[composition],
                    "support_fraction": round(support / query_counts[composition], 4),
                    "strong_parallel_candidates": composition_strong[composition][
                        source_book
                    ],
                    "mean_lexical_score_when_supported": round(
                        composition_lexical_sums[composition][source_book] / support,
                        4,
                    ),
                }
            )
        summaries[composition] = rows[:10]

    report = {
        "status": "exploratory_candidate_screen",
        "warning": (
            "Connections are ranked hypotheses, not evidence of authorship, direct "
            "borrowing, or a lost source. Genre, date, dialect, formulaic language, "
            "and transmission can produce the same signal."
        ),
        "weights": SOURCE_WEIGHTS,
        "counts": {
            "query_passages": len(queries),
            "external_passages": len(external),
            "query_compositions": len(query_counts),
            "external_books": len({(item.corpus, item.book) for item in external}),
        },
        "relation_counts": dict(relation_counts),
        "composition_source_summary": summaries,
        "passages": detailed,
    }
    report["positive_controls"] = evaluate_pesher_source_control(detailed)
    return report


def evaluate_pesher_source_control(passages: Sequence[dict]) -> dict:
    """Test whether known named Pesher source books are recovered.

    The permutation test clusters at the DSS manuscript level, so overlapping
    passages from one manuscript do not masquerade as independent evidence.
    """
    by_manuscript: dict[str, list[dict]] = defaultdict(list)
    for passage in passages:
        query = passage["query"]
        if query["composition"] != "Pesharim":
            continue
        if query["book"] not in PESHER_SOURCE_BOOKS:
            continue
        by_manuscript[query["book"]].append(passage)
    if not by_manuscript:
        return {"status": "not_run", "reason": "no mapped Pesher passages"}

    manuscripts = sorted(by_manuscript)
    expected = {book: PESHER_SOURCE_BOOKS[book] for book in manuscripts}

    def manuscript_score(book: str, expected_book: str, cutoff: int) -> float:
        rows = by_manuscript[book]
        hits = []
        for row in rows:
            ranked_books = [
                match["book"]
                for match in row["matches"][:cutoff]
                if match["corpus"] == "bhsa"
            ]
            hits.append(expected_book in ranked_books)
        return float(np.mean(hits))

    per_manuscript = {}
    for book in manuscripts:
        per_manuscript[book] = {
            "expected_bhsa_book": expected[book],
            "passages": len(by_manuscript[book]),
            "top1_recovery": round(manuscript_score(book, expected[book], 1), 4),
            "top3_recovery": round(manuscript_score(book, expected[book], 3), 4),
        }

    observed_top1 = float(
        np.mean([manuscript_score(book, expected[book], 1) for book in manuscripts])
    )
    observed_top3 = float(
        np.mean([manuscript_score(book, expected[book], 3) for book in manuscripts])
    )
    rng = np.random.default_rng(0)
    expected_values = np.array([expected[book] for book in manuscripts], dtype=object)
    permutations = 10_000
    null_top1 = np.empty(permutations)
    null_top3 = np.empty(permutations)
    for iteration in range(permutations):
        shuffled = rng.permutation(expected_values)
        null_top1[iteration] = np.mean(
            [
                manuscript_score(book, str(source_book), 1)
                for book, source_book in zip(manuscripts, shuffled)
            ]
        )
        null_top3[iteration] = np.mean(
            [
                manuscript_score(book, str(source_book), 3)
                for book, source_book in zip(manuscripts, shuffled)
            ]
        )

    return {
        "status": "known_source_positive_control",
        "unit": "DSS manuscript",
        "manuscripts": len(manuscripts),
        "passages": sum(len(rows) for rows in by_manuscript.values()),
        "macro_top1_recovery": round(observed_top1, 4),
        "macro_top3_recovery": round(observed_top3, 4),
        "permutation_iterations": permutations,
        "top1_permutation_p": round(
            float((1 + np.sum(null_top1 >= observed_top1)) / (permutations + 1)), 6
        ),
        "top3_permutation_p": round(
            float((1 + np.sum(null_top3 >= observed_top3)) / (permutations + 1)), 6
        ),
        "mapping_scope": (
            "Standard Pesher source-book mappings only; ambiguous or unmapped "
            "Pesharim are excluded."
        ),
        "per_manuscript": per_manuscript,
    }


def sha256_files(path: Path, names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        source = path / name
        if not source.is_file():
            continue
        digest.update(name.encode("utf-8"))
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_receipt(path: Path) -> dict:
    feature_names = (
        "otype.tf",
        "oslots.tf",
        "book.tf",
        "chapter.tf",
        "verse.tf",
        "g_cons_utf8.tf",
    )
    repository = path
    while repository != repository.parent and not (repository / ".git").exists():
        repository = repository.parent
    return {
        "path": str(path),
        "git_revision": git_revision(repository),
        "selected_features_sha256": sha256_files(path, feature_names),
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Cross-corpus DSS connection screen",
        "",
        "Status: **exploratory candidate screen**. This report does not establish",
        "authorship, direct borrowing, or the existence of a lost source.",
        "",
        "## Scope",
        "",
        f"- DSS query passages: {report['counts']['query_passages']}",
        f"- Query compositions: {report['counts']['query_compositions']}",
        f"- External passages: {report['counts']['external_passages']}",
        f"- External books/sources: {report['counts']['external_books']}",
        "- Signals kept separate: lexical, character/orthographic, and surface style.",
        "- Top matches are diversified to at most one passage per external book.",
        "",
        "## External shelf",
        "",
        "| Corpus | Material | Interpretive role |",
        "| :--- | :--- | :--- |",
        "| BHSA | Full consonantal Hebrew Bible | Scriptural source/parallel search |",
        "| ETCBC/extrabiblical | Early inscriptions | Diachronic language controls |",
        "| ETCBC/extrabiblical | Pirqe Avot | Later rabbinic comparator |",
        "| ETCBC/extrabiblical | Mekhilta Shirata | Later reception comparator; contains biblical material |",
        "",
        "## Known-source positive control",
        "",
    ]
    control = report["positive_controls"]
    if control.get("status") == "known_source_positive_control":
        lines.extend(
            [
                "Before interpreting unknown connections, the same unsupervised ranking",
                "was tested on Pesharim whose scriptural source book is already known.",
                "",
                f"- Manuscripts: {control['manuscripts']}",
                f"- Passages: {control['passages']}",
                f"- Macro Top-1 source-book recovery: "
                f"{100 * control['macro_top1_recovery']:.1f}%",
                f"- Macro Top-3 source-book recovery: "
                f"{100 * control['macro_top3_recovery']:.1f}%",
                f"- Manuscript-level permutation p-values: "
                f"Top-1={control['top1_permutation_p']:.4g}, "
                f"Top-3={control['top3_permutation_p']:.4g}",
                "",
            ]
        )
    else:
        lines.extend([f"- Not run: {control.get('reason', 'unknown reason')}", ""])
    lines.extend(
        [
            "## Composition-level candidates",
            "",
        ]
    )
    for composition, rows in report["composition_source_summary"].items():
        lines.extend(
            [
                f"### {composition}",
                "",
                "| External source | Top-3 support | Strong parallels | Query passages | Support | Mean lexical |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['source_book']} | {row['top3_support']} | "
                f"{row['strong_parallel_candidates']} | {row['query_passages']} | "
                f"{100 * row['support_fraction']:.1f}% | "
                f"{row['mean_lexical_score_when_supported']:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            report["warning"],
            "",
            "The next adjudication step should inspect exact passages, remove known",
            "quotations and near-duplicates, repeat the analysis within matched genres,",
            "and test whether each connection is stable across feature families and",
            "window sizes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    targets = None
    if args.targets.strip().lower() != "all":
        targets = {value.strip() for value in args.targets.split(",") if value.strip()}
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
    report = analyze_connections(queries, external, top_k=args.top_k)
    report["protocol"] = {
        "targets": sorted(targets) if targets is not None else "all",
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "min_words": args.min_words,
        "top_k": args.top_k,
        "dss_query_source": str(args.dss_preserved_jsonl),
        "dss_query_source_sha256": sha256_files(
            args.dss_preserved_jsonl.parent, [args.dss_preserved_jsonl.name]
        ),
        "dss_label_metadata": str(args.dss_csv),
        "dss_label_metadata_sha256": sha256_files(
            args.dss_csv.parent, [args.dss_csv.name]
        ),
        "bhsa": source_receipt(args.bhsa_path),
        "extrabiblical": source_receipt(args.extrabiblical_path),
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
    print(json.dumps(report["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
