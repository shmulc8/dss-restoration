import numpy as np

from training.finetune_tavbert_preserved_nonbib import choose_word_span


def test_choose_word_span_is_contiguous_and_bounded() -> None:
    text = "אב גד הו זח טי"
    rng = np.random.default_rng(7)
    for _ in range(20):
        start, end, word_count = choose_word_span(text, rng)
        target = text[start:end]
        assert 1 <= word_count <= 3
        assert len(target.split()) == word_count
        assert target == target.strip()
