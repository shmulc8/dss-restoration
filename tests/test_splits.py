"""Unit tests for unified frozen scroll splits module."""

from utils.splits import (
    load_frozen_scroll_splits,
    get_scroll_sets,
    validate_split_disjointness,
    SPLITS_JSON,
)


def test_frozen_splits_json_exists():
    assert SPLITS_JSON.exists(), "Frozen split JSON file must exist"


def test_load_frozen_scroll_splits():
    assignment = load_frozen_scroll_splits()
    assert isinstance(assignment, dict)
    assert len(assignment) > 0
    assert "1QS" in assignment or "CD" in assignment


def test_split_disjointness():
    assert validate_split_disjointness() is True
    sets = get_scroll_sets()
    assert len(sets["train"]) > 0
    assert len(sets["val"]) > 0
    assert len(sets["test"]) > 0

    # Ensure no overlap
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
