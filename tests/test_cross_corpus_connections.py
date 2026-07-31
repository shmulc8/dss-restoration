from analysis.cross_corpus_connections import (
    Passage,
    fixed_windows,
    hebrew_tokens,
    load_preserved_dss_queries,
    longest_shared_ngram,
    relation_type,
    row_percentiles,
    style_features,
)
from analysis.cross_corpus_embedding_benchmark import (
    diversified_details,
    passage_fingerprint,
)
import numpy as np


def test_hebrew_tokens_remove_editorial_and_latin_material():
    assert hebrew_tokens("abc [בראשית]־ברא 123") == ["בראשית", "ברא"]


def test_fixed_windows_keep_references_and_stop_before_short_tail():
    tokens = [f"מילה{i}" for i in range(12)]
    refs = [f"R{i // 4}" for i in range(12)]
    windows = list(
        fixed_windows(
            tokens,
            refs,
            chunk_words=6,
            overlap_words=2,
            min_words=4,
        )
    )
    assert [(start, end) for start, end, _, _ in windows] == [
        (0, 6),
        (4, 10),
        (8, 12),
    ]
    assert windows[0][3] == "R0–R1"


def test_style_features_are_length_normalized():
    short = style_features(["כי", "אמר"])
    doubled = style_features(["כי", "אמר", "כי", "אמר"])
    assert short["fw:כי"] == doubled["fw:כי"] == 0.5
    assert short["prefix:א"] == doubled["prefix:א"] == 0.5


def test_longest_shared_ngram_is_contiguous():
    left = ["א", "ב", "ג", "ד"]
    right = ["ז", "ב", "ג", "ח", "ד"]
    assert longest_shared_ngram(left, right) == 2


def test_row_percentiles_are_computed_within_each_query():
    result = row_percentiles(np.array([[3.0, 1.0, 2.0], [5.0, 7.0, 6.0]]))
    assert result.tolist() == [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]


def test_relation_labels_do_not_promote_similarity_without_controls():
    assert (
        relation_type(
            lexical_percentile=0.999,
            style_percentile=0.97,
            shared_ngram_words=4,
        )
        == "lexical_and_style_connection"
    )
    assert (
        relation_type(
            lexical_percentile=0.4,
            style_percentile=0.999,
            shared_ngram_words=1,
        )
        == "style_affinity_without_lexical_overlap"
    )
    assert (
        relation_type(
            lexical_percentile=0.999,
            style_percentile=0.2,
            shared_ngram_words=1,
        )
        == "nearest_neighbor_only"
    )


def test_preserved_queries_drop_gap_markers_and_use_metadata(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "book,composition,genre,section\n"
        "4QX,Instruction,Wisdom,sectarian_texts\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "preserved.jsonl"
    corpus.write_text(
        '{"scroll":"4QX","chunk_index":2,'
        '"text":"כי <GAP> אמר חכמה <GAP> דרך"}\n',
        encoding="utf-8",
    )
    rows = load_preserved_dss_queries(
        corpus,
        metadata_csv=metadata,
        targets={"Instruction"},
        min_words=4,
    )
    assert len(rows) == 1
    assert rows[0].tokens == ("כי", "אמר", "חכמה", "דרך")
    assert rows[0].reference == "4QX preserved chunk 2"


def test_embedding_ranking_keeps_only_best_passage_per_book():
    query = Passage("q", "dss", "4QX", "q", "א ב", ("א", "ב"))
    external = [
        Passage("a1", "bhsa", "Genesis", "1", "א", ("א",)),
        Passage("a2", "bhsa", "Genesis", "2", "ב", ("ב",)),
        Passage("b1", "bhsa", "Exodus", "1", "ג", ("ג",)),
    ]
    details = diversified_details(
        [query],
        external,
        np.array([[0.1, 0.9, 0.8]]),
        top_k=3,
    )
    assert [row["passage_id"] for row in details[0]["matches"]] == ["a2", "b1"]


def test_passage_fingerprint_changes_with_text():
    first = Passage("q", "dss", "4QX", "q", "א", ("א",))
    changed = Passage("q", "dss", "4QX", "q", "ב", ("ב",))
    assert passage_fingerprint([first]) != passage_fingerprint([changed])
