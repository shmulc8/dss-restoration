from experiments.cross_corpus_connections import Passage
from experiments.cross_corpus_retrieval_injection import (
    extract_proposals,
    merge_candidates,
    token_ids_for_phrase,
)


class FakeTokenizer:
    def __call__(self, value, *, add_special_tokens):
        assert add_special_tokens is False
        ids = {"א": [1], "ב": [2], "מפוצל": [3, 4]}
        return {"input_ids": ids[value]}

    def decode(self, ids):
        values = {1: "א", 2: "ב", 3: "מ", 4: "פ"}
        return "".join(values[item] for item in ids)


def test_extract_proposals_caps_each_candidate_length_equally():
    documents = [
        Passage(
            "d",
            "test",
            "book",
            "1",
            "א ב ג ד",
            ("א", "ב", "ג", "ד"),
        )
    ]
    proposals = extract_proposals(
        [(0, 1.0)],
        documents,
        max_per_length=2,
    )
    counts = {
        length: sum(row["word_count"] == length for row in proposals)
        for length in (1, 2, 3)
    }
    assert counts == {1: 2, 2: 2, 3: 2}


def test_extract_proposals_answer_removal_excludes_complete_gold_document():
    documents = [
        Passage("a", "test", "A", "1", "א ב ג", ("א", "ב", "ג")),
        Passage("b", "test", "B", "1", "ד ה ו", ("ד", "ה", "ו")),
    ]
    proposals = extract_proposals(
        [(0, 0.9), (1, 0.8)],
        documents,
        max_per_length=10,
        removed_phrase=("ב",),
    )
    assert "ב" not in {row["text"] for row in proposals}
    assert "ה" in {row["text"] for row in proposals}


def test_token_ids_reject_phrase_with_split_word():
    tokenizer = FakeTokenizer()
    assert token_ids_for_phrase("א ב", tokenizer) == (1, 2)
    assert token_ids_for_phrase("מפוצל", tokenizer) is None


def test_merge_candidates_respects_per_length_proposal_limit():
    baseline = [("א", -1.0, 1)]
    proposals = [
        {
            "text": "ב",
            "word_count": 1,
            "retrieval_support": 1.0,
            "proposal_rank": 1,
        },
        {
            "text": "ג",
            "word_count": 1,
            "retrieval_support": 0.5,
            "proposal_rank": 2,
        },
    ]
    scores = {"ב": (-0.5, 1), "ג": (-0.25, 1)}
    merged = merge_candidates(
        baseline,
        proposals,
        scores,
        limit_per_length=1,
    )
    assert {candidate for candidate, _, _ in merged} == {"א", "ב"}
