from eval.tf_embible_dss_benchmark import (
    boundary_f1,
    deduplicate,
    ensemble_candidates,
    embible_overlap_candidates,
    levenshtein,
    rank_with_penalty,
    summarize,
    valid_char_prefix,
    Item,
    sample_sha256,
)


def test_levenshtein() -> None:
    assert levenshtein("שלום", "שלום") == 0
    assert levenshtein("שלום", "שלם") == 1
    assert levenshtein("", "אב") == 2


def test_boundary_f1() -> None:
    assert boundary_f1("אב גד", "אב גד") == 1.0
    assert boundary_f1("אבגד", "אב גד") == 0.0
    assert boundary_f1("אב גד הו", "אב גד הו") == 1.0


def test_character_prefix_constraints() -> None:
    assert not valid_char_prefix((" ",), 2)
    assert not valid_char_prefix(("א", " ", " "), 1)
    assert not valid_char_prefix(("א", " "), 0)
    assert valid_char_prefix(("א", " "), 2)


def test_length_penalty_changes_ranking() -> None:
    rows = [("אב", -1.0, 1), ("אב גד", -1.2, 2)]
    assert rank_with_penalty(rows, 0.0)[0][0] == "אב"
    assert rank_with_penalty(rows, 1.0)[0][0] == "אב גד"


def test_deduplicate_and_ensemble() -> None:
    rows = deduplicate([("אב", -2.0, 1), ("אב", -1.0, 1), ("גד", -1.5, 1)])
    assert rows[0] == ("אב", -1.0, 1)
    fused = ensemble_candidates(
        [("אב", 1.0), ("גד", 0.0)],
        [("גד", 1.0), ("אב", 0.0)],
        0.5,
    )
    assert {row[0] for row in fused} == {"אב", "גד"}


def test_embible_overlap_uses_intersection_then_character_fallback() -> None:
    overlap = embible_overlap_candidates(
        [("אב", 2.0), ("גד", 1.0)],
        [("גד", 3.0), ("הו", 2.0)],
    )
    assert [row[0] for row in overlap] == ["גד"]
    fallback = embible_overlap_candidates(
        [("אב", 2.0)],
        [("גד", 3.0), ("הו", 2.0)],
    )
    assert [row[0] for row in fallback] == ["גד", "הו"]


def test_summary_includes_word_hits() -> None:
    item = Item(
        item_id="x",
        scroll="s",
        left=("א",),
        gold=("אב", "גד"),
        right=("ה",),
    )
    result = summarize(
        [
            (
                item,
                [
                    ("אב הו", 1.0),
                    ("אב גד", 0.5),
                ],
            )
        ]
    )
    assert result["word_hit1"] == 50.0
    assert result["word_hit5"] == 100.0


def test_sample_hash_is_order_independent_and_gold_sensitive() -> None:
    first = Item("a", "s", ("א",), ("אב",), ("ג",))
    second = Item("b", "s", ("א",), ("גד",), ("ה",))
    assert sample_sha256([first, second]) == sample_sha256([second, first])
    changed = Item("b", "s", ("א",), ("דה",), ("ה",))
    assert sample_sha256([first, second]) != sample_sha256([first, changed])
