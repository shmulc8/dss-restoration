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
- Context-only MLM: 4.3% Top-1 and 15.1% Top-10.
- Visible-trace MLM: 33.3% Top-1 and 50.5% Top-10; manuscript-cluster 95% interval
  40.3%--60.0%.
- Paired visible-trace minus context delta: +35.5 points; 95% interval
  +24.5--+46.5.
- Frequency baseline under the same traces: 36.6% Top-10. The paired MLM advantage
  is +14.0 points; 95% interval +3.7--+23.1.
- Candidate coverage: the visible-trace MLM produces no candidate for 13 targets
  and fewer than ten for 37; these remain misses.
- Editor-derived length alone reaches 16.1% Top-10. Adding it to traces leaves
  Top-10 unchanged at 50.5%; it is an oracle-assisted diagnostic, not physical
  ground truth.
- Composition-unseen subset: 24 targets from 11 scrolls; context-only 16.7%,
  visible-trace MLM 62.5%, and visible-trace frequency 33.3% Top-10.
- Artifact: [`qd_evidence_conditions_20260811.json`](../experiments/results/paper/qd_evidence_conditions_20260811.json).

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
  have labels unseen in train/development. A genuinely composition-grouped
  training experiment remains future work.
- Artifact: [`split_similarity_audit.json`](../experiments/results/paper/split_similarity_audit.json).

## Unsupported claims

The repository does not claim state-of-the-art restoration, historical truth,
successful unknown-length multiword restoration, image-derived ink analysis,
editor-specific accuracy, or demonstrated scholar productivity. The MLM
comparisons use one training seed; only the ByT5 negative result has matched
seed replication.
