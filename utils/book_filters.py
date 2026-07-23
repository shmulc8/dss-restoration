"""Book-level corpus filters used for controlled DSS benchmarks."""

CONSERVATIVE_ARAMAIC_BOOKS = {
    "1Q20",
    "11Q10",
    "4Q196",
    "4Q197",
    "4Q198",
    "4Q199",
    "4Q200",
    "4Q201",
    "4Q202",
    "4Q203",
    "4Q204",
    "4Q205",
    "4Q206",
    "4Q207",
    "4Q208",
    "4Q209",
    "4Q210",
    "4Q211",
    "4Q212",
    "4Q213",
    "4Q214",
    "4Q243",
    "4Q244",
    "4Q245",
    "4Q246",
    "4Q550",
    "4Q551",
    "4Q552",
    "4Q553",
    "4Q554",
    "4Q555",
    "4Q556",
}


def resolve_book_exclusions(mode: str):
    normalized = mode.strip().lower()
    if normalized in {"", "all", "none", "off"}:
        return set(), "all-books"
    if normalized in {"no-aram", "exclude_aramaic", "exclude-aramaic", "hebrew-only"}:
        return set(CONSERVATIVE_ARAMAIC_BOOKS), "excluding-conservative-aramaic-books"
    raise ValueError(
        f"Unknown BOOK_FILTER_MODE={mode!r}; expected all|none|no-aram|hebrew-only"
    )
