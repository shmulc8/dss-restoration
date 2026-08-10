# Domain Shift Insight: Non-Biblical DSS vs. Biblical Hebrew

This diagnostic applies the DSS-trained checkpoint and decoder to held-out
Biblical Hebrew. The word model reaches 80.0%, 42.5%, and 27.5% exact Top-10 on
one-, two-, and three-word spans, respectively. The contrast with the DSS pilot
is consistent with domain shift, but does not by itself prove which linguistic
property causes the difference.

The one-/two-/three-word synthetic damage used in the DSS benchmark is applied.
This is an apples-to-apples transfer diagnostic, not a reproduction of
Embible's random character/word masking or published metrics. It is not a P0
physical-lacuna evaluation and no SOTA claim is made.

| Hidden words | System | Exact Top-1 | Exact Top-5 | Exact Top-10 | Seq WordHit@1 | Seq WordHit@5 |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | uwc_word | 57.5% | 77.5% | 80.0% | 57.5% | 77.5% |
| 1 | char_unknown | 17.5% | 22.5% | 25.0% | 17.5% | 22.5% |
| 1 | embible_overlap_ensemble | 20.0% | 22.5% | 22.5% | 20.0% | 22.5% |
| 1 | rank_ensemble | 57.5% | 77.5% | 80.0% | 57.5% | 77.5% |
| 1 | cwc_word_oracle | 80.0% | 87.5% | 90.0% | 80.0% | 87.5% |
| 2 | uwc_word | 15.0% | 32.5% | 42.5% | 18.8% | 43.8% |
| 2 | char_unknown | 0.0% | 2.5% | 2.5% | 1.2% | 3.8% |
| 2 | embible_overlap_ensemble | 2.5% | 2.5% | 2.5% | 2.5% | 3.8% |
| 2 | rank_ensemble | 15.0% | 32.5% | 42.5% | 18.8% | 43.8% |
| 2 | cwc_word_oracle | 55.0% | 62.5% | 62.5% | 65.0% | 68.8% |
| 3 | uwc_word | 10.0% | 25.0% | 27.5% | 13.3% | 30.8% |
| 3 | char_unknown | 0.0% | 2.5% | 2.5% | 0.0% | 2.5% |
| 3 | embible_overlap_ensemble | 0.0% | 2.5% | 2.5% | 0.0% | 2.5% |
| 3 | rank_ensemble | 10.0% | 25.0% | 27.5% | 13.3% | 30.8% |
| 3 | cwc_word_oracle | 32.5% | 42.5% | 42.5% | 46.7% | 50.8% |

Development tuning used 60 Biblical spans and the
held-out evaluation used 120 spans, with at most
one target per verse in each split. No Biblical text was used for training.
