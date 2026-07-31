# Cross-corpus retrieval ablation for DSS restoration

Status: **single-checkpoint exploratory ablation**, not a final paper result.
All conditions rerank the same unknown-length candidate pool on the same
reconstruction-free held-out spans.

## Exact complete-span recovery

| Retrieval shelf | Dev alpha | Top-1 | Top-10 | Delta Top-10 | 95% cluster CI | Holm p | Answer-removed Top-10 |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| No retrieval | 0 | 4.3% | 15.0% | — | — | — | — |
| dss_train_only | 0.0 | 4.3% | 15.0% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| hebrew_bible | 0.0 | 4.3% | 15.0% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| early_hebrew_inscriptions | 0.0 | 4.3% | 15.0% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| later_rabbinic | 0.0 | 4.3% | 15.0% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| all_external | 0.0 | 4.3% | 15.0% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |
| dss_plus_external | 0.0 | 4.3% | 15.0% | +0.0 | [+0.0, +0.0] | 1.0000 | 15.0% |

## Candidate-generation ceiling

The frozen candidate pool contains the complete held-out answer for 20.7% of spans. By hidden word count the ceilings are 47.0% (one), 14.0% (two), and 1.0% (three). Retrieval can reorder candidates but cannot recover an answer absent from this pool.

## Retrieval diagnostics

| Retrieval shelf | Passages | Gold exists in shelf | Gold recall@20 | Supported gold candidate | Retrieval MRR | nDCG@20 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| dss_train_only | 1002 | 126/300 | 25.7% | 41/300 | 0.132 | 0.093 |
| hebrew_bible | 5557 | 96/300 | 17.3% | 37/300 | 0.093 | 0.055 |
| early_hebrew_inscriptions | 12 | 20/300 | 6.7% | 17/300 | 0.040 | 0.045 |
| later_rabbinic | 187 | 46/300 | 11.3% | 23/300 | 0.057 | 0.037 |
| all_external | 5756 | 96/300 | 17.3% | 37/300 | 0.094 | 0.054 |
| dss_plus_external | 6758 | 143/300 | 28.7% | 46/300 | 0.154 | 0.096 |

## Interpretation

No external shelf improved held-out exact Top-10 under the development-selected reranking rule. The negative result is retained.

Retrieval uses only the visible eight words on each side. DSS retrieval
indexes preserved training scrolls only. The answer-removal stress test
drops every retrieved document containing the complete held-out answer
before candidate support is calculated.

This run uses one trained checkpoint and one frozen 300-span pilot sample.
It does not satisfy the locked three-seed paper promotion gate.
