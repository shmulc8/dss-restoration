"""Shared evaluation split helpers for DSS restoration experiments."""

from functools import lru_cache

from utils.dss_split import load_split


@lru_cache(maxsize=1)
def scroll_sets():
    train, test, _ = load_split()
    return {
        "train": {row["book"] for row in train},
        "heldout": {row["book"] for row in test},
    }


def resolve_scroll_filter(mode: str):
    """Return (scroll_names_or_none, human_label) for the requested eval split."""
    normalized = mode.strip().lower()
    if normalized == "all":
        return None, "all-nonbib"
    if normalized in {"heldout", "test"}:
        return scroll_sets()["heldout"], "heldout-scrolls"
    if normalized == "train":
        return scroll_sets()["train"], "train-scrolls"
    raise ValueError(f"Unknown EVAL_SCROLL_SPLIT={mode!r}; expected all|heldout|train")
