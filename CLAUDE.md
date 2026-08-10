# CLAUDE.md

## Project scope

This repository evaluates language-model assistance for non-biblical Dead Sea
Scroll lacunae using the ETCBC/dss Text-Fabric corpus. Training is
reconstruction-free: editor-supplied letters are removed and damage is emitted
as anonymous `<GAP>` markers.

Two tracks must remain separate:

1. **Synthetic exact recovery:** hide preserved one-to-three-word spans and
   score the full string under unknown character length, word count, and word
   boundaries.
2. **Natural-lacuna literature agreement:** rank any attributed Qumran Digital
   proposal at one manuscript target. This is not historical ground truth.

There is no human-participant study and no image-derived ink analysis.

## Authoritative evidence

- [`docs/RESULTS.md`](docs/RESULTS.md) is the human-readable register.
- `experiments/results/paper/paper_results_snapshot.json` is the aggregate
  machine-readable snapshot.
- `docs/paper_results_manifest.json` drives generated LaTeX tables.
- `experiments/paper_protocol_v1.json` is the current protocol despite its
  compatibility filename; its schema is `dss-paper-protocol-v2`.

Do not revive the superseded 74-target / 63.5% QD result or describe editor-
derived length as physical measurement. The current QD denominator is 93
targets. Visible traces, editor length, and context are separate conditions;
all references remain in every denominator.

## Frozen data and split

The checkpoint-associated authority is
`curation/derived/preserved_nonbib_manifest.json`: 736 identifiers, 1,647
chunks, train/dev/heldout partitions, and zero scroll overlap. Active paper
experiments load the derived corpus through `curation.preserved_corpus`.
`data_preparation/dss_scroll_splits_v1.json` belongs to an older unification
experiment and must not be mixed into checkpoint claims.

The split is manuscript-disjoint but not composition-disjoint. The paper
reports a composition-unseen QD subset and a near-duplicate audit; it does not
claim composition-grouped training.

## Current paper evidence

- QD context-only MLM: 15.1% Top-10.
- QD visible-trace MLM: 50.5% Top-10; frequency under the same traces: 36.6%.
- Balanced synthetic spans: 300 targets across 79 held-out scrolls; word-span
  exact Top-10 7.7%, with 22.0% / 1.0% / 0.0% by one/two/three words.
- Matched ByT5 seeds 41--43: 1.3%, 1.3%, and 1.0% exact Top-10.

The positive claim is evidence-conditioned candidate narrowing. The negative
claim is that unknown-length multiword restoration remains unsolved.

## Validation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python curation/validate_preserved_nonbib_corpus.py --derived-only
.venv/bin/python experiments/validate_leakage.py
.venv/bin/python experiments/validate_paper_protocol.py
.venv/bin/python experiments/audit_split_similarity.py
.venv/bin/python experiments/run_paper_benchmark.py
.venv/bin/python experiments/build_paper_data_profile.py
.venv/bin/python docs/generate_paper_tables.py
```

`models/` and `scratch/` contain local evidence and must not be deleted. The
source corpus is CC BY-NC 4.0 and requires attribution to Martin G. Abegg Jr.,
James E. Bowley, and Edward M. Cook.
