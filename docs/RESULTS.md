# Current evidence register

Status: 23 July 2026.

This file is the sole index of paper-facing numerical evidence. Results not
listed here are exploratory, superseded, or awaiting a frozen protocol.

## Evidence classes

| Class | Meaning |
| :--- | :--- |
| Grounded diagnostic | The hidden reference is physically preserved text, but the task is not a real lacuna |
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
