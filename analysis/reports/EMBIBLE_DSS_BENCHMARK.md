# Embible-style synthetic-damage DSS benchmark

## Held-out synthetic-lacuna results

| System | N spans | Exact Top-1 | Exact Top-5 | Exact Top-10 | Seq WordHit@1 | Seq WordHit@5 | Top-1 CER | Boundary F1 | Word-count MAE | Failure |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uwc_word | 30 | 3.3% | 16.7% | 16.7% | 1.7% | 10.0% | 0.890 | 0.300 | 1.033 | 0.0% |
| char_unknown | 30 | 6.7% | 6.7% | 6.7% | 3.3% | 3.3% | 0.837 | 0.333 | 1.000 | 0.0% |
| embible_overlap_ensemble | 30 | 6.7% | 6.7% | 6.7% | 3.3% | 3.3% | 0.802 | 0.333 | 1.000 | 0.0% |
| rank_ensemble | 30 | 3.3% | 6.7% | 10.0% | 1.7% | 3.3% | 0.856 | 0.267 | 1.000 | 0.0% |
| cwc_word_oracle | 30 | 20.0% | 33.3% | 33.3% | 15.0% | 21.7% | 0.738 | 0.567 | 1.100 | 46.7% |

Character oracle-length diagnostic: CharHit@1
15.3%, CharHit@5 48.3% over
236 characters.

## Contiguous damage severity

With eight context words on each side, hiding one, two, or three words removes
5.9%, 11.1%, or 15.8% of the displayed word sequence. These are close to
Embible's 5%, 10%, and 15% conditions, but the DSS targets remain contiguous.

| Approximate masked share | UWC Top-10 | Character Top-10 | Embible ensemble Top-10 | Rank ensemble Top-10 |
| :--- | ---: | ---: | ---: | ---: |
| 5.9% / 1 word(s) | 50.0% | 20.0% | 20.0% | 30.0% |
| 11.1% / 2 word(s) | 0.0% | 0.0% | 0.0% | 0.0% |
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
character model is the cached TavBERT base checkpoint;
it has not yet been fine-tuned on the preserved-only DSS corpus. This report is
therefore an implemented baseline matrix, not a final paper result.

`Seq WordHit@K` asks whether a gold word appears in its correct position within
one of the top K complete sequences. It is stricter than, and not numerically
identical to, Embible's independently calculated WordHit@K.
