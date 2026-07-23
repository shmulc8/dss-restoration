"""Helpers for composition-aware DSS evaluation."""

import csv
from functools import lru_cache

from utils.paths import repo_path

CSV = repo_path("dss_chunks.csv")


def _group_id(book: str, composition: str) -> str:
    composition = (composition or "").strip()
    if composition:
        return composition
    return f"BOOK::{book}"


@lru_cache(maxsize=1)
def scroll_to_composition_group() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with CSV.open() as fh:
        for row in csv.DictReader(fh):
            book = row["book"]
            mapping.setdefault(book, _group_id(book, row.get("composition", "")))
    return mapping


def composition_group_for_scroll(scroll: str) -> str:
    return scroll_to_composition_group().get(scroll, f"BOOK::{scroll}")
