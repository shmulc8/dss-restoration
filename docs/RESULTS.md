# Current evidence register

Status: 11 August 2026. The authoritative team-review snapshot is
[`paper_results_snapshot.json`](../experiments/results/paper/paper_results_snapshot.json).
Older result files are provenance records and do not override this register.

## Claim boundary

- Synthetic damage measures exact recovery of preserved text hidden for testing.
- Natural lacunae measure agreement with one or more attributed modern proposals.
- Literature agreement is not recovery of historical truth.
- Empty candidate sets, short lists, tokenizer failures, and unrepresentable
  targets remain in every primary denominator.
- No human-participant study is included.

## A. Reconstruction-free corpus

- 736 non-biblical scroll identifiers; 377 contribute at least one eligible
  modeling chunk.
- 1,647 chunks, 95,736 preserved word tokens, and 27,814 lacuna records.
- Editorial reconstruction text is removed from training labels; gaps become
  anonymous `<GAP>` markers.
- Median lacuna extent is four source-word positions; 32.7% contain six or more.
- Artifact: [`paper_data_profile.json`](../experiments/results/paper/paper_data_profile.json).

## B. Natural-lacuna evidence ablation

- Scope: 93 Qumran Digital single-word targets across 40 held-out scrolls, with
  184 distinct attributed readings.
- Three matched MLM seeds: 41, 42, and 43; identical two-epoch, batch-16,
  learning-rate $3\times10^{-5}$ training on 1,002 chunks.
- Context-only MLM: 5.0% mean Top-1 and 14.7% mean Top-10.
- Soft-trace MLM: 35.1% mean Top-1 and 56.6% mean Top-10; seed-and-scroll
  hierarchical 95% interval 46.5%--66.7%.
- Hard-trace MLM: 52.3% mean Top-10; seed SD 2.2 points versus 0.6 for soft traces.
- Paired soft-trace minus context delta: +41.9 points; hierarchical 95% interval
  +30.8--+53.0.
- Frequency with hard traces: 36.6% Top-10. The soft-MLM advantage is +20.1
  points; hierarchical 95% interval +9.2--+30.2.
- Candidate coverage: the visible-trace MLM produces no candidate for 13 targets
  and fewer than ten for 37; these remain misses.
- Editor-derived length alone reaches 16.1% Top-10. Adding it to traces leaves
  Top-10 unchanged at 50.5%; it is an oracle-assisted diagnostic, not physical
  ground truth.
- Targeted composition exclusion retains 785/1,002 training chunks and removes
  all 23 QD-associated non-empty labels. On 68 labeled targets, regular and
  excluded seed-42 checkpoints have identical Top-10 hit rates: 11.8% context,
  52.9% soft traces, and 50.0% hard traces.
- Context-only Top-10 is 6.5%, 11.8%, 14.0%, and 15.1% at 2, 5, 10, and full
  stored context; soft traces remain 55.9%--57.0%.
- Exact-context retrieval ties frequency at 36.6%. No QD reading forms an exact
  preserved-training trigram or five-gram, though 59 targets contain a reading
  seen somewhere in training vocabulary.
- Artifacts: [`qd_method_extensions_summary.json`](../experiments/results/paper/qd_method_extensions_summary.json),
  [`qd_composition_exclusion_summary.json`](../experiments/results/paper/qd_composition_exclusion_summary.json),
  and [`qd_memorization_audit.json`](../experiments/results/paper/qd_memorization_audit.json).

## C. Unknown-length synthetic spans

- Scope: 300 non-overlapping spans, 100 each of one, two, and three words,
  manuscript-balanced across 79 held-out scrolls.
- Character length, word count, and word boundaries are not supplied.
- Preserved-only word span: 4.0% Top-1 and 7.7% exact Top-10; manuscript-cluster
  95% interval 5.0%--10.8%.
- Preserved-only TavBERT: 2.7% exact Top-10.
- Embible-style overlap: 2.3% exact Top-10.
- Dev-fitted rank fusion: 7.7% exact Top-10; the simpler word model is retained
  under the locked tie rule.
- Word-model Top-10 by hidden words: 22.0%, 1.0%, and 0.0%.
- Complete-span tokenizer coverage: 81.7%; uncovered targets remain misses.
- Artifact: [`span_balanced_300_20260811.json`](../experiments/results/paper/span_balanced_300_20260811.json).

## D. Matched ByT5 replication

All checkpoints use the same corpus hash, three epochs, batch size 32, learning
rate $10^{-3}$, and balanced target hash; only training seed varies.

| Seed | Top-1 | Exact Top-10 |
| ---: | ---: | ---: |
| 41 | 0.0% | 1.3% |
| 42 | 0.7% | 1.3% |
| 43 | 0.0% | 1.0% |

Every checkpoint scores 0% Top-10 on both two- and three-word strata.

## E. Split audit

- Train/development and held-out scroll identifiers are disjoint.
- No exact normalized chunk duplicate crosses the boundary.
- Across 407 held-out chunks, maximum preserved-word five-gram Jaccard against
  any train/development chunk is 0.174; none reaches 0.5.
- Twenty-six of 88 non-empty composition labels cross splits; 15 held-out scrolls
  have labels unseen in train/development. The targeted exclusion sensitivity
  addresses QD overlap but is not a representative population split.
- Artifact: [`split_similarity_audit.json`](../experiments/results/paper/split_similarity_audit.json).

## Unsupported claims

The repository does not claim state-of-the-art restoration, historical truth,
successful unknown-length multiword restoration, image-derived ink analysis,
editor-specific accuracy, or demonstrated scholar productivity.
