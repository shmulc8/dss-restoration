from analysis.cross_corpus_connections import Passage
from analysis.cross_corpus_quote_ablation import (
    QUOTE_GAP,
    benjamini_hochberg,
    mask_external_ngrams,
    ngram_inventory,
    surviving_inventory_matches,
)


def passage(text: str) -> Passage:
    tokens = tuple(text.split())
    return Passage("p", "test", "book", "ref", text, tokens)


def test_masking_removes_all_words_in_overlapping_external_trigrams() -> None:
    query = passage("א ב ג ד ה ו")
    inventory = ngram_inventory([passage("ב ג ד ה")], 3)
    residual, audit = mask_external_ngrams(query, inventory, n=3)
    assert residual.tokens == ("א", "ו")
    assert residual.text == f"א {QUOTE_GAP} ו"
    assert audit["matched_windows"] == 2
    assert audit["masked_words"] == 4


def test_masking_keeps_nonmatching_passage_unchanged() -> None:
    query = passage("א ב ג ד")
    residual, audit = mask_external_ngrams(
        query, {("ה", "ו", "ז")}, n=3
    )
    assert residual.tokens == query.tokens
    assert residual.text == query.text
    assert audit["masked_fraction"] == 0


def test_match_audit_does_not_bridge_quote_gap() -> None:
    inventory = {("א", "ב", "ג")}
    assert surviving_inventory_matches("א ב ג", inventory, n=3) == 1
    assert surviving_inventory_matches(
        f"א ב {QUOTE_GAP} ג", inventory, n=3
    ) == 0


def test_benjamini_hochberg_is_monotone_in_p_value_order() -> None:
    rows = [{"p": 0.001}, {"p": 0.02}, {"p": 0.04}]
    benjamini_hochberg(rows, "p")
    adjusted = [row["bh_adjusted_p"] for row in rows]
    assert adjusted == sorted(adjusted)
    assert adjusted[0] <= 0.01
