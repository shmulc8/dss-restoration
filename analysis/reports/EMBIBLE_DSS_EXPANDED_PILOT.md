# Expanded Embible-style DSS diagnostic

Status: model-selection pilot, 23 July 2026. This is not a frozen paper result.

The run uses 60 development spans and 300 held-out synthetic-damage spans,
balanced across one, two, and three hidden words. Targets are physically
preserved DSS text hidden artificially; they are not real manuscript lacunae.
The character arm is TavBERT fine-tuned only on preserved non-biblical DSS
segments.

Development sample SHA-256:
`c2b9a99a604702a20a9a64f0e5f5f01f70e997918a4547dd13e36b25094b4d7f`.
Held-out sample SHA-256:
`d3a8701df04ae725655c6a1d423cf84b2fb82a6d55bbf76ac78eda152be18ec5`.
Word checkpoint SHA-256:
`013ca67a3808034c70510da61d7e3a458196cd9ee8245a9d9072b91e4bd4bfc6`.
Character checkpoint SHA-256:
`018e908859e92251c4694b96e11cffacb1bd4d2e9371002aa92b3a96515eab74`.

| System | Exact Top-1 | Exact Top-5 | Exact Top-10 | Top-1 CER |
| :--- | ---: | ---: | ---: | ---: |
| Preserved-only word model | 4.3% | 12.7% | 15.0% | 0.824 |
| Preserved-only TavBERT | 2.7% | 4.7% | 6.0% | 0.862 |
| Embible paper-style overlap | 3.0% | 4.7% | 4.7% | 0.853 |
| Dev-fitted rank fusion | 5.7% | 12.0% | 15.0% | 0.811 |
| Oracle word-length filter | 12.7% | 19.7% | 20.7% | 0.756 |

Exact Top-10 by hidden span length:

| System | 1 word | 2 words | 3 words |
| :--- | ---: | ---: | ---: |
| Preserved-only word model | 40.0% | 5.0% | 0.0% |
| Preserved-only TavBERT | 17.0% | 1.0% | 0.0% |
| Embible paper-style overlap | 13.0% | 1.0% | 0.0% |
| Dev-fitted rank fusion | 41.0% | 4.0% | 0.0% |

Development selected word length penalty -1.0, character length penalty 0.0,
and rank-fusion word weight 0.6. Character oracle-length CharHit@1 is 20.6% and
CharHit@5 is 55.8% over 2,377 characters.

The one-token-per-word decoder can represent 554/600 target words (92.3%) and
259/300 complete spans (86.3%). Complete-span coverage is 95.0%, 87.0%, and
77.0% for one-, two-, and three-word targets. Primary scores include all 300
spans rather than conditioning on representability.

The rank fusion ties word-only Top-10 and improves Top-1 by four cases, but it
does not improve two- or three-word Top-10. Under the locked simplicity
tie-break, the word-only system remains the best implemented baseline.

This sample extends the same deterministic sampling stream as the earlier
30-span diagnostic, so the smaller sample is nested inside it. The run has no
clustered confidence interval and cannot be described as an independent
confirmation or final test.
