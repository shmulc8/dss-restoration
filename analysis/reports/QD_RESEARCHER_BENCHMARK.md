# Qumran Digital constrained restoration benchmark

## Result

This experiment evaluates the reconstruction-free preserved-only model on
single-word lacunae from the stored Qumran Digital snapshot. Unlike the
superseded whole-word-mask experiment, it retains visibly preserved letters
and an approximate lacuna-derived word length
(±1 character).

| Unit | N | Top-1 | Top-5 | Top-10 | Top-20 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Target: constrained MLM | 74 | 40.5% | 62.2% | 63.5% | 67.6% |
| Target: constrained MLM + train-only RAG | 74 | 40.5% | 62.2% | 63.5% | 67.6% |
| Unique target-reading pair: constrained MLM | 99 | 30.3% | 59.6% | 60.6% | 64.6% |
| Unique target-reading pair: MLM + RAG | 99 | 30.3% | 59.6% | 60.6% | 64.6% |
| QD initial reading control | 74 | 20.3% | 41.9% | 43.2% | 44.6% |

Baseline target-level Top-10 95% cluster-bootstrap interval:
**51.4%–74.3%**. The RAG weight
(0.5) was selected on preserved
non-biblical dev scrolls only; held-out targets were not used for tuning.
Without manuscript constraints, the same target-level Top-10 is
9.5%. The difference measures the value of physical
evidence supplied to the decoder, not an improvement in the language model.

### Length-tolerance sensitivity

| Allowed difference | Eligible targets | Top-1 | Top-10 | Top-20 |
| :--- | ---: | ---: | ---: | ---: |
| ±0 | 59 | 52.5% | 69.5% | 71.2% |
| ±1 | 74 | 40.5% | 63.5% | 67.6% |
| ±2 | 75 | 40.0% | 62.7% | 66.7% |

The conclusion is stable across exact-length, ±1, and ±2 decoding. The number
of eligible targets changes because a published proposal outside a tolerance
is not treated as physically compatible at that setting.

## Largest publication samples

Each publication contributes at most one observation per target; duplicate
publication rows and duplicate readings do not receive extra weight.

| Publication | Targets | Top-1 | Top-10 |
| :--- | ---: | ---: | ---: |
| Study Edition | 24 | 20.8% | 62.5% |
| Qimron 2013 | 23 | 30.4% | 52.2% |
| PrCon I | 10 | 40.0% | 50.0% |
| Qimron 2020 | 9 | 44.4% | 66.7% |
| Wacholder/Abegg 1995 | 9 | 33.3% | 55.6% |
| DJD XXIX | 8 | 37.5% | 50.0% |
| Qimron 2014 | 8 | 50.0% | 75.0% |
| Brown-deVost 2019 | 6 | 16.7% | 50.0% |
| Qimron 2010 | 6 | 50.0% | 83.3% |
| Lohse 1971 | 5 | 20.0% | 80.0% |

## Scope and exclusions

- Cached source snapshot: Qumran Digital 2026-05-21;
  the scorer performs no network requests.
- Corpus: held-out non-biblical DSS scrolls only.
- Training: preserved letters only; square-bracket scholarly restorations are
  absent from fine-tuning data.
- Primary unit: one manuscript target. Success means any distinct,
  bibliographically attributed restoration compatible with the physical
  pattern is in Top-K.
- Input rows: 1811; eligible targets:
  74; unique compatible target-reading pairs:
  99.
- Multiword readings, scribal corrections, modern alternatives, incomplete
  readings, non-lacuna variants, and readings contradicting visible letters
  are reported as exclusions rather than concatenated into fake words.

This is still a literature-agreement benchmark, not physical ground truth.
QD selected these locations because they are disputed, and its variant
collection is working data. Publication-level samples are descriptive and
must not be treated as a ranking of researchers.

## Reproduction

Both commands below are offline when the stored snapshot exists:

```bash
.venv/bin/python eval/build_qd_researcher_benchmark.py
.venv/bin/python eval/score_qd_researcher_benchmark.py
```

Only an explicit `eval/build_qd_researcher_benchmark.py --refresh` contacts
Qumran Digital.
