# Qumran Digital attributed-reading benchmark

## Result

The preserved-only DSS model was compared with 1,559 explicit readings from
240 bibliographic sources at 267 selected disputed words in held-out,
non-biblical scrolls.

| Scope | N | Top-1 | Top-5 | Top-10 | Top-20 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All scorable attributed readings | 1,559 | 2.6% | 5.7% | 8.0% | 9.1% |
| Consonantally distinct from QD initial reading | 1,155 | 1.2% | 3.7% | 6.6% | 7.7% |

The largest publication-level samples are:

| Publication | N | Top-1 | Top-10 | Distinct N | Distinct Top-10 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| García Martínez / Tigchelaar, *Study Edition* | 118 | 0.8% | 10.2% | 109 | 11.0% |
| Qimron 2010 | 92 | 4.3% | 7.6% | 51 | 5.9% |
| Qimron 2013 | 84 | 1.2% | 7.1% | 70 | 5.7% |
| Parry / Tov, *PrCon I* | 61 | 3.3% | 9.8% | 45 | 6.7% |
| Qimron 2020 | 58 | 3.4% | 5.2% | 31 | 6.5% |
| Lohse 1971 | 55 | 3.6% | 9.1% | 48 | 6.2% |
| Habermann 1959 | 49 | 4.1% | 8.2% | 46 | 6.5% |
| Broshi 1992 | 39 | 2.6% | 10.3% | 34 | 11.8% |
| Rabin 1958 | 33 | 3.0% | 9.1% | 31 | 9.7% |
| Wacholder / Abegg 1995 | 29 | 3.4% | 10.3% | 21 | 9.5% |

Do not compare the overall 8.0% directly with the 36.3% preserved intact-word
benchmark as if they measured the same task. Qumran Digital selected these
targets because the readings are disputed or otherwise notable. They are
substantially harder, and a single target often has several incompatible
published readings.

## Protocol

- Source snapshot: Qumran Digital, 2026-05-21.
- Corpus: non-biblical DSS manuscripts only.
- Split: the pre-existing reconstruction-free held-out scroll split. None of
  these scrolls was used to fine-tune the preserved-only model.
- Selection: words for which Qumran Digital displays an alternative reading
  inline in its transcription.
- Attribution: only API records with a bibliography and an explicit reading;
  readings that Qumran Digital inferred because a checked edition recorded no
  difference were excluded.
- Context: the full target word was masked. Surrounding words containing
  square-bracket researcher reconstructions were replaced with `<GAP>`.
- Quality filter: at least ten visible context words; incomplete proposed
  readings containing `○`, `--`, or `.` were not scored.
- Match: exact Hebrew consonants after removing editorial punctuation and
  combining marks.
- Unit: one publication's reading at one target. Publication-level results
  with very small N should not be interpreted as rankings of researchers.

## Interpretation and limitations

This measures whether the model assigns a high contextual rank to readings
reported in scholarly literature. It does not establish which reading is
physically correct. Qumran Digital explicitly warns that much of its variant
collection is working data that has not yet been checked extensively.

The sample is not a complete census of Qumran Digital's variants. Its public
API exposes full variants one word at a time, while only a selected subset is
marked inline. To avoid a large crawl, the importer queries only those inline
targets and stores the result once.

SQE 0.33.0 was also audited. Although its schema supports multiple editions,
the public database snapshot has one non-archived edition per manuscript and
all sanitized character assignments belong to its system user. It therefore
cannot supply an independent per-editor comparison in its current public
form.

## Stored artifacts and reproduction

- `data/derived/qd_researcher_variants.jsonl`: 1,811 explicit attributed
  readings at 346 targets before scoring-quality filters.
- `data/derived/qd_researcher_variants_manifest.json`: snapshot metadata,
  scope, counts, and 253-source bibliography.
- `analysis/reports/qd_researcher_comparison.json`: aggregate metrics and
  scored records.

The following is offline when the stored snapshot exists:

```bash
.venv/bin/python eval/build_qd_researcher_benchmark.py
.venv/bin/python eval/score_qd_researcher_benchmark.py
```

The first command prints that it is reusing the stored snapshot. Network
collection occurs only when `--refresh` is explicitly supplied.
