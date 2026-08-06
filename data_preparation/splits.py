"""Unified frozen scroll split loader and validator.

Implements Decision Point #3 from UNIFICATION_DECISION_POINTS.md:
- Reads the canonical `data_preparation/dss_scroll_splits_v1.json` file.
- Provides strict manuscript-disjoint partition lookups.
- Validates that no scroll straddles multiple splits.
"""

import json
from pathlib import Path
from typing import Dict, Set

SPLITS_JSON = Path(__file__).resolve().parent / "dss_scroll_splits_v1.json"


def load_frozen_scroll_splits() -> Dict[str, str]:
    """Load the canonical scroll -> split assignment dictionary."""
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(f"Frozen split file missing: {SPLITS_JSON}")
    with open(SPLITS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["scroll_assignment"]


def get_scroll_sets() -> Dict[str, Set[str]]:
    """Return scroll name sets for each partition ('train', 'val', 'test')."""
    assignment = load_frozen_scroll_splits()
    sets: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    for scroll, split in assignment.items():
        if split in sets:
            sets[split].add(scroll)
    return sets


def validate_split_disjointness() -> bool:
    """Assert that train, val, and test scroll sets are strictly disjoint."""
    sets = get_scroll_sets()
    train_val_intersect = sets["train"] & sets["val"]
    train_test_intersect = sets["train"] & sets["test"]
    val_test_intersect = sets["val"] & sets["test"]

    if train_val_intersect or train_test_intersect or val_test_intersect:
        raise ValueError(
            f"Split overlap detected! "
            f"train/val: {train_val_intersect}, "
            f"train/test: {train_test_intersect}, "
            f"val/test: {val_test_intersect}"
        )
    return True
