# Qumran Digital constrained restoration benchmark

## Result

This experiment evaluates a reconstruction-free preserved-only model on
single-word lacunae from the stored Qumran Digital snapshot. It separates
context, transcription-visible Hebrew, and an editor-derived length proxy.
The latter is an oracle/editor-assisted condition, not an independent physical
measurement. Every condition retains the same 93 targets and all attributed
readings, including proposals that disagree with a supplied filter.

| Condition | N | Top-1 | Top-10 | 95% scroll CI | Empty | <10 cand. |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| MLM: context only | 93 | 4.3% | 15.1% | 7.4--23.2% | 0 | 0 |
| MLM: soft traces | 93 | 35.5% | 55.9% | 45.6--65.6% | 0 | 0 |
| MLM: visible traces | 93 | 33.3% | 50.5% | 40.3--60.0% | 13 | 37 |
| MLM: editor-derived length | 93 | 6.5% | 16.1% | 8.4--24.2% | 0 | 0 |
| MLM: traces + editor-derived length | 93 | 32.3% | 50.5% | 40.2--60.5% | 15 | 45 |
| Frequency: context only | 93 | 0.0% | 3.2% | 0.0--7.3% | 0 | 0 |
| Frequency: visible traces | 93 | 22.6% | 36.6% | 26.7--47.6% | 29 | 64 |
| Frequency: traces + editor-derived length | 93 | 18.3% | 33.3% | 23.9--43.9% | 30 | 68 |
| Retrieval: visible traces | 93 | 21.5% | 36.6% | 26.7--47.6% | 29 | 64 |

Visible-trace conditioning improves Top-10 over context-only by 35.5 points;
the paired scroll-cluster 95% interval is
24.5--46.5
points. The comparison is an input ablation on transcribed evidence, not a
claim about image-derived ink or historical truth. Candidate coverage is
reported because a restrictive filter can return fewer than ten hypotheses.

### Length-tolerance sensitivity

| Allowed difference | Eligible targets | Top-1 | Top-10 | Top-20 |
| :--- | ---: | ---: | ---: | ---: |
| ±0 | 93 | 33.3% | 44.1% | 45.2% |
| ±1 | 93 | 32.3% | 50.5% | 53.8% |
| ±2 | 93 | 32.3% | 50.5% | 53.8% |

The target denominator stays fixed across tolerances; incompatible proposals
remain references but are not retrievable under that particular filter.

## Largest publication samples

Each publication contributes at most one observation per target; duplicate
publication rows and duplicate readings do not receive extra weight.

| Publication | Targets | Top-1 | Top-10 |
| :--- | ---: | ---: | ---: |
| Study Edition | 39 | 12.8% | 38.5% |
| Qimron 2013 | 35 | 20.0% | 34.3% |
| PrCon I | 23 | 17.4% | 21.7% |
| Wacholder/Abegg 1995 | 16 | 18.8% | 31.2% |
| Qimron 2010 | 15 | 20.0% | 33.3% |
| Qimron 2020 | 12 | 33.3% | 50.0% |
| Qimron 2014 | 11 | 36.4% | 54.5% |
| DJD XXIX | 10 | 30.0% | 40.0% |
| Brown-deVost 2019 | 8 | 12.5% | 37.5% |
| Zimmermann 1998 | 7 | 14.3% | 42.9% |

## Scope and exclusions

- Cached source snapshot: Qumran Digital 2026-05-21;
  the scorer performs no network requests.
- Corpus: held-out non-biblical DSS scrolls only.
- Training: preserved letters only; square-bracket scholarly restorations are
  absent from fine-tuning data.
- Primary unit: one manuscript target. Success means any distinct,
  bibliographically attributed restoration is in Top-K.
- Input rows: 1811; eligible targets:
  93; unique compatible target-reading pairs:
  184.
- Multiword readings, scribal corrections, modern alternatives, incomplete
  readings, and non-lacuna variants are reported as exclusions rather than
  concatenated into artificial single tokens.

This is still a literature-agreement benchmark, not physical ground truth.
QD selected these locations because they are disputed, and its variant
collection is working data. Publication-level samples are descriptive and
must not be treated as a ranking of researchers.

## Reproduction

Both commands below are offline when the stored snapshot exists:

```bash
.venv/bin/python curation/build_qd_researcher_benchmark.py
.venv/bin/python experiments/run_qd_benchmark.py
```

Only an explicit `curation/build_qd_researcher_benchmark.py --refresh` contacts
Qumran Digital.
