"""Shared evaluation split helpers for DSS restoration experiments."""

from functools import lru_cache

from utils.dss_split import protocol_splits


@lru_cache(maxsize=1)
def scroll_sets():
    splits = protocol_splits()
    return {
        "train": splits["train_books"],
        "dev": splits["dev_books"],
        "fit": splits["fit_books"],
        "heldout": splits["heldout_books"],
    }


def resolve_scroll_filter(mode: str):
    """Return (scroll_names_or_none, human_label) for the requested eval split."""
    normalized = mode.strip().lower()
    if normalized == "all":
        return None, "all-nonbib"
    if normalized == "fit":
        return scroll_sets()["fit"], "fit-scrolls"
    if normalized == "dev":
        return scroll_sets()["dev"], "dev-scrolls"
    if normalized in {"heldout", "test"}:
        return scroll_sets()["heldout"], "heldout-scrolls"
    if normalized == "train":
        return scroll_sets()["train"], "train-scrolls"
    raise ValueError(f"Unknown EVAL_SCROLL_SPLIT={mode!r}; expected all|fit|train|dev|heldout")
