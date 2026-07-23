# Preserved non-biblical DSS corpus

This directory is generated from `ETCBC/dss` Text-Fabric 2.0 using:

```bash
.venv/bin/python data/build_preserved_nonbib_corpus.py
.venv/bin/python data/validate_preserved_nonbib_corpus.py
```

## Data rules

- Only words with `biblical == 0` are considered.
- Signs with `rec == 1` are modern editorial reconstructions and are never
  emitted as training text.
- Modern removals and explicit `#` unknown material are also redacted.
- Each affected source word becomes `<GAP>` in the training sequence.
- Lacuna records retain source-word-count and missing-character-count
  estimates, visible preserved-letter patterns, and adjacent preserved
  context. They do not retain reconstructed letters.
- Scroll-level train, development, and held-out sets are disjoint.

`rec == 1` identifies a modern reconstruction but does not identify the
individual editor or edition responsible for that reading. These files
therefore do not themselves support per-researcher or inter-editor
comparisons. The separately sourced Qumran Digital files below do.

## Files

- `preserved_nonbib_chunks.jsonl`: model-training sequences.
- `nonbib_lacunae.jsonl`: reconstruction-free lacuna metadata.
- `preserved_nonbib_manifest.json`: rules, counts, splits, and checksums.
- `qd_researcher_variants.jsonl`: one-time 2026-05-21 Qumran Digital snapshot
  of explicit, bibliographically attributed readings at selected disputed
  held-out words. It is evaluation-only and is never used for training.
- `qd_researcher_variants_manifest.json`: source, scope, warning, counts, and
  bibliography for the attributed snapshot.

`eval/score_qd_researcher_benchmark.py` interprets the cached QD editorial
notation offline. Its primary metric is one target (not one publication row),
keeps only single-word lacuna restorations, preserves visible-letter and
approximate-length constraints, and reports other editorial variants as
exclusions instead of concatenating them into model candidates.

The source transcription is based on Martin Abegg's data files and is
attributed in Text-Fabric to Martin G. Abegg Jr., James E. Bowley, and Edward
M. Cook. The derived data follows the source corpus's CC BY-NC 4.0 license; it
is not covered by the repository's software license.

The Qumran Digital variant snapshot is a separate source under CC BY-SA 4.0.
Its importer reuses the stored files without network access unless
`--refresh` is explicitly supplied.
