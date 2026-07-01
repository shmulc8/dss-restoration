"""Shared, deterministic DSS split used by both the evaluator and the finetuner.

Book-level split (whole scrolls to train OR test) so the 15-word chunk overlap
cannot leak test text into training. Train = non-biblical only (specialize to
sectarian Hebrew, don't memorize scripture). A small biblical sample is returned
separately purely as a canon-recall contrast for eval.
"""
import csv
import numpy as np
from utils.paths import repo_path

CSV = repo_path("dss_chunks.csv")

SECT = {"sectarian_texts": "sect", "non_sectarian_texts": "non"}


def load_split(test_frac_books=0.30, seed=0, n_bib_contrast=60):
    rows = [r for r in csv.DictReader(CSV.open())
            if r["text"].strip() and len(r["text"].split()) >= 20]
    nonbib = [r for r in rows if r["bib"] == "nonbib"]
    bib = [r for r in rows if r["bib"] == "bib"]

    books = sorted(set(r["book"] for r in nonbib))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(books))
    n_test_books = max(1, int(len(books) * test_frac_books))
    test_books = {books[i] for i in perm[:n_test_books]}

    train = [r for r in nonbib if r["book"] not in test_books]
    test = [r for r in nonbib if r["book"] in test_books]

    bsel = rng.choice(len(bib), size=min(n_bib_contrast, len(bib)), replace=False)
    bib_contrast = [bib[i] for i in bsel]

    return train, test, bib_contrast


if __name__ == "__main__":
    tr, te, bc = load_split()
    def sect_n(rs):
        return sum(1 for r in rs if SECT.get(r["section"]) == "sect")
    print(f"train nonbib chunks: {len(tr)} ({sect_n(tr)} sectarian)")
    print(f"test  nonbib chunks: {len(te)} ({sect_n(te)} sectarian)")
    print(f"bib contrast chunks: {len(bc)}")
    print(f"train words ~ {sum(len(r['text'].split()) for r in tr):,}")
