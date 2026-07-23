# Current evidence register

Status: 23 July 2026.

This file is the sole index of paper-facing numerical evidence. Results not
listed here are exploratory, superseded, or awaiting a frozen protocol.

## Evidence classes

| Class | Meaning |
| :--- | :--- |
| Synthetic-damage diagnostic | The reference is preserved text hidden artificially; the task is not a real lacuna |
| Literature-agreement pilot | The reference is one or more modern attributed proposals at a real lacuna |
| Editorial-label pilot | The reference is an anonymous Text-Fabric reconstruction used only for evaluation |
| Paper result | Frozen protocol, full leakage audit, uncertainty, and predeclared analysis complete |

No result currently qualifies as a final paper result.

## Retained results

### A. Preserved-language recovery

- Unit: one intact preserved word.
- Test scope: 300 sampled targets from held-out non-biblical scrolls.
- Context: editorial reconstructions are redacted as unlabelled masks.
- Preserved-only MsBERT: 13.7% Top-1, 30.7% Top-5, 36.3% Top-10, 43.7%
  Top-20.
- Status: grounded diagnostic.
- Limitation: this samples intact words and does not model unknown-length real
  damage.
- Artifact:
  [`analysis/reports/preserved_nonbib_intact_benchmark.json`](../analysis/reports/preserved_nonbib_intact_benchmark.json).

### B. Attributed Qumran Digital comparison

- Unit: one real manuscript target.
- Scope: 74 held-out non-biblical single-word lacunae; 99 distinct compatible
  target-reading pairs.
- Input evidence: visible letters and approximate word length with tolerance of
  one character.
- Target-level constrained MLM: 40.5% Top-1, 63.5% Top-10, 67.6% Top-20.
- Cluster-bootstrap 95% interval for Top-10: 51.4%–74.3%.
- The same targets without physical constraints: 9.5% Top-10.
- Status: literature-agreement pilot.
- Limitation: selected disputed sites, single-word only, and modern proposals
  rather than physical truth.
- Artifact:
  [`analysis/reports/QD_RESEARCHER_BENCHMARK.md`](../analysis/reports/QD_RESEARCHER_BENCHMARK.md).

### C. Train-only RAG ablation

- Retrieval source: preserved non-biblical training scrolls only.
- Weight: selected on development scrolls.
- Same 74 Qumran Digital targets: 63.5% to 63.5% Top-10.
- Held-out Text-Fabric single-word subset: 60.0% to 64.0% Top-10, 25 spans.
- Held-out Text-Fabric multiword subset: 41.4% to 41.8% slot Top-10, 440
  slots in 100 spans.
- Exact multiword sequences: 7.0% to 9.0% Top-10, 100 spans.
- Status: literature-agreement or editorial-label pilot, depending on row.
- Limitation: known word-slot count, small balanced sample, and no inferential
  claim for the observed deltas.
- Artifact:
  [`analysis/reports/PRESERVED_RAG_LACUNA_LENGTHS.md`](../analysis/reports/PRESERVED_RAG_LACUNA_LENGTHS.md).

### D. Embible-style synthetic-damage character/word baseline

- Unit: one contiguous span of physically preserved text, hidden artificially
  for testing. These are synthetic lacunae, not real manuscript lacunae.
- Scope: 15 development and 30 held-out spans, balanced across one, two, and
  three words.
- Primary information: unknown character length, word count, and word
  boundaries. Search caps are three words and 18 characters.
- Word-only exact Top-10: 16.7%.
- Base TavBERT character-only exact Top-10: 6.7%.
- Scaled Embible overlap ensemble exact Top-10: 6.7%.
- Separate dev-fitted rank ensemble exact Top-10: 10.0%.
- All four primary systems score 0% exact Top-10 on the two- and three-word strata;
  their aggregate hits come from the single-word stratum.
- Oracle diagnostics: word-boundary-filtered Top-10 is 33.3%, with 46.7%
  candidate failure; character oracle-length CharHit@5 is 48.3% over 236
  characters.
- Status: synthetic-damage diagnostic pilot.
- Limitation: 30 targets, no confidence interval, base TavBERT is not
  preserved-DSS fine-tuned, the Embible word-candidate pool is scaled down from
  1,000 to at most 60, and neither ensemble improves exact recovery over UWC.
  Like Embible's masked-Tanakh evaluation, this does not measure accuracy on
  naturally occurring lacunae. The named overlap condition follows the paper's
  stated rule; it is not an exact reproduction of the discrepant public backend
  implementation.
- Frozen held-out sample hash:
  `9d3e547ba461b7ec2743e6948cc4e8b9f4c72fd7652fd476a91bce94ea265132`.
- Artifacts:
  [`analysis/reports/EMBIBLE_DSS_BENCHMARK.md`](../analysis/reports/EMBIBLE_DSS_BENCHMARK.md)
  and
  [`analysis/reports/embible_dss_benchmark.json`](../analysis/reports/embible_dss_benchmark.json).

### E. Fixed-decoder Bible domain transfer

- Source: Embible's released Biblical validation and test JSONL at pinned
  backend commit `7c9e769274a273d0b357b066d932f1c6833ca5f8`.
- Canonical resolution: 526/535 validation and 527/536 test rows matched a
  unique unpointed verse; nine ambiguous rows in each split were excluded.
- Evaluation: 60 development and 120 held-out spans, balanced across one, two,
  and three hidden words, with at most one target per verse.
- Biblical text was used only for development calibration and held-out
  evaluation, never model training.
- UWC exact Top-10: 80.0% for one word, 42.5% for two words, and 27.5% for three
  words. The corresponding DSS diagnostic is 50.0%, 0.0%, and 0.0%.
- Overall balanced UWC exact Top-10: 50.0% on Bible versus 16.7% on DSS.
- Base TavBERT exact Top-10: 25.0%, 2.5%, and 2.5%. The paper-style overlap
  ensemble scores 22.5%, 2.5%, and 2.5%.
- Development selected rank-ensemble word weight 1.0, making it identical to
  UWC. The character arm therefore adds no value in this diagnostic.
- Oracle-boundary CWC exact Top-10: 90.0%, 62.5%, and 42.5%.
- Status: domain-transfer diagnostic pilot.
- Interpretation: the same word model and decoder recover multiword Biblical
  spans but not DSS spans. Domain generalization is therefore a major measured
  bottleneck, although the different sample sizes and small DSS strata prevent
  a final inferential claim.
- Artifact:
  [`analysis/reports/EMBIBLE_BIBLE_TRANSFER.md`](../analysis/reports/EMBIBLE_BIBLE_TRANSFER.md)
  and
  [`analysis/reports/embible_bible_transfer.json`](../analysis/reports/embible_bible_transfer.json).

## Claims that are not supported

The repository does not currently claim:

- a single end-to-end DSS restoration accuracy;
- a large RAG gain;
- state-of-the-art performance;
- successful unknown-length multiword restoration;
- recovery of the manuscript's original wording at real lacunae;
- demonstrated scholar productivity or accuracy gains;
- cross-period generalization;
- editor-specific accuracy rankings.

The earlier master reports and gold-length RAG pipeline were removed because
their units, splits, or information conditions did not support those claims.
They remain recoverable from Git history for audit but are not current results.
