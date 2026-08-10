# Embible-style synthetic-damage DSS benchmark

## Held-out synthetic-lacuna results

| System | N spans | Exact Top-1 | Exact Top-5 | Exact Top-10 | Seq WordHit@1 | Seq WordHit@5 | Top-1 CER | Boundary F1 | Word-count MAE | Failure |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uwc_word | 300 | 4.3% | 12.7% | 15.0% | 2.2% | 7.7% | 0.824 | 0.338 | 0.990 | 0.0% |
| char_unknown | 300 | 2.7% | 4.7% | 6.0% | 1.3% | 2.5% | 0.862 | 0.333 | 1.000 | 0.0% |
| embible_overlap_ensemble | 300 | 3.0% | 4.7% | 4.7% | 1.5% | 2.5% | 0.853 | 0.330 | 1.000 | 0.0% |
| rank_ensemble | 300 | 5.7% | 12.0% | 15.0% | 2.8% | 7.5% | 0.811 | 0.332 | 0.980 | 0.0% |
| cwc_word_oracle | 300 | 12.7% | 19.7% | 20.7% | 14.0% | 18.3% | 0.756 | 0.570 | 1.143 | 45.0% |

Character oracle-length diagnostic: CharHit@1
20.6%, CharHit@5 55.8% over
2377 characters.

Word-tokenizer representability: 92.3%
of target words and 86.3%
of complete spans. All spans remain in the primary denominator.

## Contiguous damage severity

With eight context words on each side, hiding one, two, or three words removes
5.9%, 11.1%, or 15.8% of the displayed word sequence. These are close to
Embible's 5%, 10%, and 15% conditions, but the DSS targets remain contiguous.

| Approximate masked share | UWC Top-10 | Character Top-10 | Embible ensemble Top-10 | Rank ensemble Top-10 |
| :--- | ---: | ---: | ---: | ---: |
| 5.9% / 1 word(s) | 40.0% | 17.0% | 13.0% | 41.0% |
| 11.1% / 2 word(s) | 5.0% | 1.0% | 1.0% | 4.0% |
| 15.8% / 3 word(s) | 0.0% | 0.0% | 0.0% | 0.0% |

## Interpretation

`uwc_word`, `char_unknown`, `embible_overlap_ensemble`, and `rank_ensemble` do
not receive the gold span length, word count, or word boundaries.
`cwc_word_oracle` and `char_oracle_length` are ceiling diagnostics and must not
be compared as real-world systems.

`embible_overlap_ensemble` follows the rule described in the Embible paper:
intersect the Top-5 character sequences with the word candidates, average
normalized scores, and fall back to the character list when no overlap exists.
The candidate pool is smaller than the paper's Top-1,000 pool and is reported
as a scaled paper-protocol adaptation, not an exact code reproduction.
`rank_ensemble` is our separate dev-fitted baseline.

Targets are contiguous physically preserved words that we hide artificially in
reconstruction-free held-out DSS scrolls. They are **synthetic lacunae, not real
manuscript lacunae**. This is directly analogous to Embible's evaluation on
randomly masked Tanakh verses, which Embible itself lists as a limitation. The
character model is preserved-only DSS fine-tuned TavBERT.
This report is an implemented baseline matrix, not a final paper result.

`Seq WordHit@K` asks whether a gold word appears in its correct position within
one of the top K complete sequences. It is stricter than, and not numerically
identical to, Embible's independently calculated WordHit@K.
