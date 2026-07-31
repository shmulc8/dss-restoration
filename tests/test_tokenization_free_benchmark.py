from eval.tf_embible_dss_benchmark import Item
from eval.tf_tokenization_free_benchmark import normalize_candidate, summarize


def test_normalize_candidate_removes_sentinel_and_normalizes_space() -> None:
    assert normalize_candidate("<extra_id_0>  אמת   ו צדק") == "אמת ו צדק"


def test_summary_uses_exact_complete_span_and_unknown_length_ranking() -> None:
    items = [
        Item("one", "1QX", ("שמאל",), ("אמת",), ("ימין",)),
        Item("two", "1QX", ("שמאל",), ("אמת", "וצדק"), ("ימין",)),
    ]
    result = summarize(
        items,
        [
            ["אמת", "אמת וצדק"],
            ["אמת", "אמת וצדק"],
        ],
    )
    assert result["exact_top1"] == 50.0
    assert result["exact_top5"] == 100.0
    assert result["exact_top10"] == 100.0
    assert result["by_word_count"]["1"]["exact_top1"] == 100.0
    assert result["by_word_count"]["2"]["exact_top1"] == 0.0
