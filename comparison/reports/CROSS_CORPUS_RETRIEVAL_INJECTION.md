# Retrieval-conditioned candidate generation

Status: **single-checkpoint exploratory ablation**, not a final paper result.
Retrieval proposes phrases, but the preserved-only word model scores them
before they enter the unknown-length candidate ranking.

## Exact complete-span recovery

| Retrieval shelf | Dev proposals/length | Top-1 | Top-10 | Candidate recall | Delta Top-10 | 95% cluster CI | Holm p | Answer-removed Top-10 |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| No injection | 0 | 4.3% | 15.0% | 20.7% | — | — | — | — |
| dss_train_only | 0 | 4.3% | 15.0% | 20.7% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| hebrew_bible | 0 | 4.3% | 15.0% | 20.7% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| early_hebrew_inscriptions | 0 | 4.3% | 15.0% | 20.7% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| later_rabbinic | 0 | 4.3% | 15.0% | 20.7% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| all_external | 0 | 4.3% | 15.0% | 20.7% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| dss_plus_external | 0 | 4.3% | 15.0% | 20.7% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |

## Maximum-injection diagnostic (not selected)

The table below forces 50 proposals per candidate length only to locate the bottleneck. It is not a development-selected system.

| Retrieval shelf | Candidate recall | Gold proposed | Top-10 | Answer-removed recall |
| :--- | ---: | ---: | ---: | ---: |
| dss_train_only | 26.7% | 49/300 | 15.0% | 20.7% |
| hebrew_bible | 22.0% | 24/300 | 15.0% | 20.7% |
| early_hebrew_inscriptions | 21.0% | 12/300 | 15.0% | 20.7% |
| later_rabbinic | 21.7% | 16/300 | 15.0% | 20.7% |
| all_external | 22.0% | 23/300 | 15.0% | 20.7% |
| dss_plus_external | 26.0% | 46/300 | 15.0% | 20.7% |

## Interpretation

Development-selected retrieval proposals did not improve held-out exact Top-10. Even the unselected maximum-injection diagnostic left Top-10 unchanged while raising candidate recall as high as 26.7%. Retrieval can supply missing answers, but this word model assigns them scores below the useful ranking boundary. The negative result is retained.

Every shelf proposes the same maximum number of one-, two-, and
three-word phrases, so the hidden word count is not supplied. Retrieval
queries contain only the visible context. The answer-removal stress test
excludes every retrieved document containing the complete held-out answer
before proposal extraction.

This run uses one trained checkpoint and the frozen 300-span pilot.
It does not satisfy the locked three-seed paper promotion gate.
