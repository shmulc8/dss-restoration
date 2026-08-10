# Embible-style synthetic-damage DSS benchmark

## Held-out synthetic-lacuna results

| System | N spans | Exact Top-1 | Exact Top-5 | Exact Top-10 | Seq WordHit@1 | Seq WordHit@5 | Top-1 CER | Boundary F1 | Word-count MAE | Failure |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uwc_word | 300 | 4.0% | 6.7% | 7.7% | 2.0% | 4.0% | 0.893 | 0.333 | 0.993 | 0.0% |
| char_unknown | 300 | 1.0% | 2.3% | 2.7% | 0.5% | 1.3% | 0.898 | 0.333 | 1.000 | 0.0% |
| embible_overlap_ensemble | 300 | 2.3% | 2.3% | 2.3% | 1.2% | 1.3% | 0.884 | 0.333 | 1.000 | 0.0% |
| rank_ensemble | 300 | 4.3% | 7.0% | 7.7% | 2.2% | 4.2% | 0.868 | 0.333 | 0.980 | 0.0% |
| cwc_word_oracle | 300 | 6.7% | 10.7% | 12.0% | 6.7% | 8.8% | 0.860 | 0.477 | 1.353 | 54.3% |

Character oracle-length diagnostic: CharHit@1
18.1%, CharHit@5 51.4% over
2416 characters.

Word-tokenizer representability: 89.8%
of target words and 81.7%
of complete spans. All spans remain in the primary denominator.

## Contiguous damage severity

With 2 context words on each side, the masked share is calculated
over the displayed sequence. The DSS targets are contiguous and are not
numerically equated with Embible's masking conditions.

| Approximate masked share | UWC Top-10 | Character Top-10 | Embible ensemble Top-10 | Rank ensemble Top-10 |
| :--- | ---: | ---: | ---: | ---: |
| 20.0% / 1 word(s) | 22.0% | 8.0% | 7.0% | 22.0% |
| 33.3% / 2 word(s) | 1.0% | 0.0% | 0.0% | 1.0% |
| 42.9% / 3 word(s) | 0.0% | 0.0% | 0.0% | 0.0% |

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
