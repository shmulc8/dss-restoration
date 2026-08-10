import numpy as np

from curation.preserved_corpus import GAP_TOKEN
from tuning.finetune_preserved_mlm import choose_preserved_words


def test_mask_selection_never_uses_real_gaps() -> None:
    words = ["אב", GAP_TOKEN, "גד", "הו", GAP_TOKEN, "זח"]
    for seed in range(20):
        chosen = choose_preserved_words(words, np.random.default_rng(seed))
        assert chosen
        assert all(words[index] != GAP_TOKEN for index in chosen)
