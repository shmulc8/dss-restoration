"""Shared, deterministic DSS splits used by training and evaluation.

The primary protocol is book-level:
1. `heldout` scrolls are never seen during model development.
2. The remaining `fit` scrolls are split into `train` and `dev`.

This prevents overlapping chunks from leaking across experimental boundaries
while still leaving room for real model selection before touching the final
held-out benchmark.
"""
import csv
from functools import lru_cache
import numpy as np
from utils.paths import repo_path

CSV = repo_path("dss_chunks.csv")

SECT = {"sectarian_texts": "sect", "non_sectarian_texts": "non"}


def _load_rows():
    rows = [r for r in csv.DictReader(CSV.open())
            if r["text"].strip() and len(r["text"].split()) >= 20]
    return rows


@lru_cache(maxsize=8)
def protocol_splits(final_frac_books=0.30, dev_frac_within_fit=0.20, seed=0, n_bib_contrast=60):
    rows = _load_rows()
    nonbib = [r for r in rows if r["bib"] == "nonbib"]
    bib = [r for r in rows if r["bib"] == "bib"]

    books = sorted(set(r["book"] for r in nonbib))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(books))
    n_final_books = max(1, int(len(books) * final_frac_books))
    final_books = {books[i] for i in perm[:n_final_books]}

    fit_books = [books[i] for i in perm[n_final_books:]]
    n_dev_books = max(1, int(len(fit_books) * dev_frac_within_fit))
    dev_books = set(fit_books[:n_dev_books])
    train_books = set(fit_books[n_dev_books:])

    train = [r for r in nonbib if r["book"] in train_books]
    dev = [r for r in nonbib if r["book"] in dev_books]
    heldout = [r for r in nonbib if r["book"] in final_books]
    fit = train + dev

    bsel = rng.choice(len(bib), size=min(n_bib_contrast, len(bib)), replace=False)
    bib_contrast = [bib[i] for i in bsel]

    return {
        "train": train,
        "dev": dev,
        "fit": fit,
        "heldout": heldout,
        "bib_contrast": bib_contrast,
        "train_books": train_books,
        "dev_books": dev_books,
        "fit_books": set(fit_books),
        "heldout_books": final_books,
    }


def load_partition(name, final_frac_books=0.30, dev_frac_within_fit=0.20, seed=0):
    splits = protocol_splits(
        final_frac_books=final_frac_books,
        dev_frac_within_fit=dev_frac_within_fit,
        seed=seed,
    )
    normalized = name.strip().lower()
    aliases = {
        "test": "heldout",
        "final": "heldout",
        "train+dev": "fit",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"train", "dev", "fit", "heldout"}:
        raise ValueError(f"Unknown partition {name!r}; expected train|dev|fit|heldout")
    return splits[normalized]


def load_split(test_frac_books=0.30, seed=0, n_bib_contrast=60):
    """Backward-compatible 2-way split: fit vs heldout."""
    splits = protocol_splits(
        final_frac_books=test_frac_books,
        dev_frac_within_fit=0.20,
        seed=seed,
        n_bib_contrast=n_bib_contrast,
    )
    return splits["fit"], splits["heldout"], splits["bib_contrast"]


if __name__ == "__main__":
    splits = protocol_splits()
    tr, dv, te, bc = splits["train"], splits["dev"], splits["heldout"], splits["bib_contrast"]
    def sect_n(rs):
        return sum(1 for r in rs if SECT.get(r["section"]) == "sect")
    print(f"train nonbib chunks: {len(tr)} ({sect_n(tr)} sectarian)")
    print(f"dev   nonbib chunks: {len(dv)} ({sect_n(dv)} sectarian)")
    print(f"test  nonbib chunks: {len(te)} ({sect_n(te)} sectarian)")
    print(f"bib contrast chunks: {len(bc)}")
    print(f"train words ~ {sum(len(r['text'].split()) for r in tr):,}")
