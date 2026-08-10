# Locked methodology

This document describes the protocol implemented in the team-review paper.
Machine-readable details live in
[`paper_protocol_v1.json`](../experiments/paper_protocol_v1.json).

## 1. Corpus

- Source: ETCBC/dss Text-Fabric 2.0, non-biblical scrolls only.
- A source word may become a training label only when all emitted letters are
  preserved; editor-supplied or damage-marked material becomes `<GAP>`.
- Frozen checkpoint split: 1,002 train, 238 development, and 407 held-out
  chunks; scroll identifiers do not cross partitions.
- All 736 identifiers remain in the archival registry. Only 377 yield an
  eligible reconstruction-free modeling chunk.

## 2. Evaluation tracks

### Synthetic exact recovery

Hide 100 one-word, 100 two-word, and 100 three-word preserved spans. Sampling
cycles through manuscripts, rejects overlapping hidden intervals, and covers
79 held-out scrolls. The decoder receives two context words on each side but
not character length, word count, or boundaries. The primary metric is exact
complete-span Top-10; every failure remains in the denominator.

Decoder penalties and ensemble weights are selected on a separate 60-target
development sample. Report exact Top-1/5/10, character error rate, boundary F1,
word-count error, decoder failures, tokenizer coverage, and results by hidden
word count. Gold word length is an oracle diagnostic only.

### Natural-lacuna literature agreement

The fixed Qumran Digital snapshot yields 93 eligible single-word targets across
40 held-out scrolls and 184 attributed readings. One target is one primary
observation; success means any attributed proposal is in Top-K. A proposal is
never removed merely because it conflicts with a tested filter.

Evaluate these conditions on the identical targets and references:

1. context only;
2. transcription-visible Hebrew segments;
3. editor-derived length within one character (oracle-assisted);
4. visible segments plus editor-derived length (oracle-assisted).

Run both the preserved-only MLM and a training-corpus frequency baseline where
applicable. Report candidate-set coverage because a hard trace filter may
return fewer than ten candidates or none.

## 3. Statistics

- Manuscript location is the observation unit.
- Absolute Top-10 intervals resample scroll clusters.
- Condition comparisons use paired scroll-cluster bootstrap deltas.
- The synthetic benchmark additionally reports paired exact McNemar tests with
  Holm adjustment for the declared system family.
- The QD MLM uses matched seeds 41, 42, and 43 with identical data and
  hyperparameters. Hierarchical intervals resample both seeds and scrolls.
  ByT5 uses matched seeds 41, 42, and 43 as well.

## 4. Leakage and generalization

- Train/development and held-out scrolls are disjoint.
- Test targets never select penalties, ensemble weights, or checkpoints.
- Exact normalized duplicate chunks and preserved-word five-gram Jaccard are
  audited across the boundary.
- Composition labels cross the manuscript split. In addition to the unseen
  subset, train a seed-42 sensitivity checkpoint after excluding every
  training scroll whose label occurs among labeled QD targets. Report its
  reduced training volume; this is not a representative new split.

## 5. Claim boundary

Synthetic recovery is not natural-damage accuracy. QD literature agreement is
not historical truth. Visible letters are copied from a transcription, not
inferred from images. Editor-derived length is never called a physical
measurement. The current evidence supports candidate narrowing and exposes a
failure on unknown-length multiword restoration; it does not support a
state-of-the-art or end-to-end restoration claim.
