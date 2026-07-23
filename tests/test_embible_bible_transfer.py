import pytest

from eval.embible_bible_transfer_benchmark import (
    normalize_bible_text,
    restore_verse,
    sample_items,
)


def test_restore_verse_replaces_known_characters() -> None:
    row = {
        "verse": "אב? ד?",
        "missing_dictionary": {"2": "ג", "5": "ה"},
    }
    assert restore_verse(row) == "אבג דה"


def test_restore_verse_rejects_inconsistent_source() -> None:
    row = {"verse": "אבג", "missing_dictionary": {"1": "ד"}}
    with pytest.raises(ValueError, match="expected"):
        restore_verse(row)


def test_normalize_bible_text_removes_marks_and_maqaf() -> None:
    assert normalize_bible_text("בְּרֵאשִׁית־בָּרָא׃") == "בראשית ברא"


def test_bible_sampling_uses_unique_verses() -> None:
    rows = [
        {
            "name": "Book",
            "verse_idx": index,
            "resolved_verse": " ".join(["אב"] * 20),
            "missing_dictionary": {},
        }
        for index in range(6)
    ]
    selected, _ = sample_items(
        rows,
        "test",
        per_length=2,
        max_words=2,
        max_chars=18,
        context_words=2,
        seed=1,
    )
    assert len(selected) == 4
    assert len({item.scroll for item in selected}) == 4
