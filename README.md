# Dead Sea Scrolls lacuna restoration

This repository studies whether language models and retrieved textual parallels
can help scholars assess missing text in the non-biblical Dead Sea Scrolls.
The project is deliberately reconstruction-free during fine-tuning: modern
editorial restorations are removed from training targets and from the retrieval
index.

- [Team-review PDF](docs/paper.pdf)
- [Locked paper methodology](docs/METHODOLOGY.md)
- [Final method and system decision](docs/archive/BEST_METHOD.md)
- [Current evidence register](docs/RESULTS.md)

## Research claim

The intended contribution is a leakage-controlled evaluation framework for DSS
restoration, not a claim that a model has recovered the original wording of a
damaged manuscript. The framework separates two questions:

1. Can the model recover preserved DSS language after we hide it synthetically
   for testing?
2. Does it rank attested modern scholarly proposals highly at real lacunae?
The current draft does not include or propose a human-subjects study.

## Current results

Synthetic exact recovery and natural-lacuna literature agreement answer
different questions and must not be collapsed into one accuracy headline.

| Track | Evaluation unit | Result | Interpretation |
| :--- | :--- | :--- | :--- |
| Natural lacuna, context only | 93 Qumran Digital targets / 40 scrolls | MLM 14.7% mean Top-10 | Three matched seeds; agreement with attributed readings, not truth |
| Natural lacuna, soft traces | Same 93 targets | MLM 56.6% mean Top-10 | No candidates discarded; seed-and-scroll 95% interval 46.5--66.7 |
| Natural lacuna, hard traces | Same 93 targets | MLM 52.3% mean; frequency 36.6% Top-10 | Hard filtering is less stable and can empty candidate sets |
| Composition exclusion | 68 labeled QD targets / 28 scrolls | Soft-trace 52.9% with and without exclusion | Seed-42 sensitivity; 785/1,002 training chunks retained |
| Unknown-length synthetic spans | 300 non-overlapping targets / 79 scrolls | word model 7.7% exact Top-10 | 22.0% / 1.0% / 0.0% for one/two/three words |
| ByT5 replication | Same 300 targets, matched seeds 41--43 | 1.3%, 1.3%, 1.0% exact Top-10 | Stable negative sequence-model result |

Soft-trace conditioning changes mean MLM Top-10 from 14.7% to 56.6% (paired
seed-and-scroll 95% interval for the 41.9-point delta: 30.8--53.0). The
editor-derived length proxy adds no Top-10 gain and is labeled oracle-assisted.

The exact provenance, limitations, and status of every retained number are in
[`docs/RESULTS.md`](docs/RESULTS.md). Superseded master reports and the earlier
RAG evaluation that leaked gold length have been removed from the current
repository. Git history preserves them for audit.

## Reproducibility release

The frozen paper snapshot, its source hashes, and the exact validation commands
are documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The CI workflow
runs the full test suite and evidence validators, checks that generated paper
artifacts are current, compiles `docs/paper.tex`, and publishes the compiled PDF
as a workflow artifact.

## Clean data and training path

The derived corpus is built from `ETCBC/dss` Text-Fabric 2.0:

1. keep only non-biblical scroll material;
2. replace reconstructed or unknown material with anonymous `<GAP>` slots;
3. permit only physically preserved words to become fine-tuning labels;
4. split by scroll before training, tuning, retrieval, or evaluation.

The checked-in manifest records 736 archival scroll identifiers, of which 377
contribute at least one reconstruction-free MLM chunk, together with 1,647
chunks and 27,814 lacuna records. Scrolls without eligible model text remain in
the registry for provenance and lacuna-shape analysis. Rebuild and validate the
corpus with:

```bash
.venv/bin/python curation/build_preserved_nonbib_corpus.py
.venv/bin/python curation/validate_preserved_nonbib_corpus.py
.venv/bin/python experiments/validate_leakage.py
```

Fine-tune the current preserved-only baseline with:

```bash
.venv/bin/python tuning/unified_trainer.py
```

This checkpoint is a baseline, not a claim of a final restoration model. The
evaluation uses development-only decoder selection, manuscript-cluster
uncertainty, a composition-unseen stress subset, realistic candidate failures,
and an unknown-length decoder. The baseline matrix also adopts Embible's Hebrew
character/word comparison: word-only, TavBERT-style character-only, and
character-word ensembles.
Like Embible, this comparison creates synthetic damage in intact text so the
answer is known. Real manuscript lacunae are a separate literature-agreement
track, not an automatic accuracy test.

The benchmark also shows that the word decoder can express only 81.7% of
complete targets because it assumes one tokenizer token per missing word.
Unrepresentable targets remain misses.

## Current evaluation entry points

```bash
# List the paper-facing evaluations without running them
.venv/bin/python experiments/run_all_experiments.py --list

# Run validation only
.venv/bin/python experiments/run_all_experiments.py --checks

# Validate the machine-readable promotion and evaluation contract directly
.venv/bin/python experiments/validate_paper_protocol.py

# Rerun the paper evaluations (requires local checkpoints)
.venv/bin/python experiments/run_all_experiments.py --pilots

# Run only the Embible-style character/word matrix
.venv/bin/python experiments/run_all_experiments.py --pilots --only embible

# Run one matched ByT5 seed
.venv/bin/python experiments/run_all_experiments.py --pilots --only byt5-42
```

The runner includes only the paper-facing pipeline. Older experimental scripts
are not registered paper results.

## Result terminology

- **Synthetic preserved recovery:** physically preserved transcription is
  hidden artificially, creating a known-answer benchmark rather than a real
  lacuna.
- **Literature agreement:** a prediction matches at least one
  attributed modern proposal.
- **Slot score:** an individual missing word is evaluated independently.
- **Exact-sequence score:** every word in the proposed span must match in order.
The primary synthetic metric is exact complete-span Top-10
under unknown length. Top-1/5/20, character error rate, reciprocal rank, slot
scores, calibration, and abstention are secondary diagnostics.

## Source and licensing boundary

Text-Fabric attributes its transcription to Martin G. Abegg Jr., James E.
Bowley, and Edward M. Cook, based on Martin Abegg's data files. Its `rec`
feature identifies modern reconstruction but does not attribute each reading to
an editor. Publication-level comparisons therefore use the separately cached,
attributed Qumran Digital snapshot. Those comparisons measure agreement with
the literature, not manuscript truth.

The data manifest and cached-source notes contain the applicable source and
license information. Do not redistribute external editions beyond their
licenses.
