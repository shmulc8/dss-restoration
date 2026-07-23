# Locked methodology for a paper-ready DSS restoration study

Status: protocol specification, 23 July 2026. Existing numbers are pilots. A
result becomes paper-facing only after the benchmark manifest, exclusions,
metrics, and analysis code are frozen.

## 1. Research questions

The paper must answer four questions separately.

**RQ1 — synthetic-damage recovery.** Can a model recover text that is
physically preserved but hidden artificially under a realistic model of
manuscript damage?

**RQ2 — literature agreement.** At real lacunae, does the model rank one or
more physically compatible, attributed scholarly restorations highly?

**RQ3 — retrieval contribution.** On exactly the same test targets, does
retrieval from eligible training material improve the base model, and when?

**RQ4 — scholar utility.** Do model candidates, retrieved parallels, or their
combination improve scholars' final decisions, time, or confidence?

RQ1 has known transcription answers because the damage is synthetic. RQ2 has
editorial reference readings at natural lacunae, but no observable manuscript
truth. Neither substitutes for RQ4.

## 2. Corpus and provenance

### 2.1 Training corpus

- Source: ETCBC DSS Text-Fabric 2.0.
- Scope: non-biblical scrolls only for the primary model.
- Modern reconstructions (`rec == 1`), explicit unknown material, and modern
  removals are never prediction labels.
- Missing material is represented by an anonymous `<GAP>` input. Its editorial
  text is not retained in the training example.
- Each generated file receives a schema version, build parameters, source
  version, row counts, and SHA-256 digest.

### 2.2 Split hierarchy

Freeze splits before training:

1. **Primary split: scroll-disjoint.** No manuscript scroll may occur in more
   than one of train, development, and test.
2. **Hard generalization split: composition-disjoint.** Witnesses of a held-out
   composition are absent from training and retrieval.
3. **Near-duplicate control.** Normalize orthography, construct centered
   n-grams and longer shingles, and remove or group cross-split duplicates.
4. **Version control.** Alternate transcriptions or editions of the same
   passage stay in one split.

Publish the frozen identifiers and hashes. Development data alone is used for
hyperparameters, decoding rules, thresholds, RAG weights, and early stopping.

## 3. Training

### 3.1 Required baselines

Train or evaluate:

1. pretrained MsBERT without DSS fine-tuning;
2. frequency and local n-gram baselines;
3. preserved-only masked-language-model fine-tuning;
4. preserved-only contiguous-span fine-tuning;
5. the same best model with retrieval variants;
6. an unknown-length sequence model or decoder.

A character-aware baseline should be included if it can use visible fragments
without converting them into gold length. Following Embible, the baseline
matrix should also test whether combining character-level and word-level
rankings helps with partial words and whitespace uncertainty.

### 3.2 Training objective

- Artificial targets come only from preserved words.
- Sample contiguous spans, not only independent tokens.
- Match the synthetic span-length distribution to the empirical distribution
  of real DSS damage, with separate reporting for 1, 2, 3, 4–5, and 6+ words.
- Simulate partial visible letters and damaged surrounding context from the
  observed corpus distribution.
- Keep natural examples and a macro-balanced training/evaluation view; do not
  optimize solely for a uniformly balanced artificial test.
- Log the exact base checkpoint, tokenizer, optimizer, learning-rate schedule,
  batch size, epochs, masking distribution, seed, hardware, package versions,
  corpus hash, and code commit.

### 3.3 Model selection

- Tune only on development scrolls.
- Use at least three training seeds for the final selected configuration.
- Select by the predeclared primary development metric, not by test Top-K.
- Report all seeds, mean and dispersion; do not report only the best run.
- Freeze the final checkpoint and decoder before scoring the test set.

### 3.4 Embible-derived character/word experiment

Embible contributes a particularly relevant Hebrew experimental design. It
compares word models with TavBERT character predictions and distinguishes
unknown from known whitespace and word-length information. The first controlled
DSS baseline matrix is implemented in `eval/tf_embible_dss_benchmark.py`, with
stronger splits and span metrics. Its current character arm uses base TavBERT;
it is not yet a final trained-system comparison.

Embible does **not** evaluate naturally occurring inscription lacunae. It
randomly masks 5%, 10%, or 15% of 1,071 Tanakh verses, and explicitly names
testing masked Biblical verses instead of actual inscriptions as its main
limitation. Our analogous automatic benchmark also hides preserved text
artificially. We improve domain relevance and leakage control by using
scroll-disjoint, non-biblical DSS text, but we do not turn synthetic damage into
evidence of accuracy at real lacunae.

Required systems:

1. **word-only unconstrained:** MsBERT ranks words without a gold character
   length;
2. **character-only:** TavBERT or an equivalent character model predicts
   letters and whitespace;
3. **word-only constrained:** the word model is filtered by a physically
   observable character budget and/or word boundaries;
4. **Embible overlap ensemble:** intersect the Top-5 character sequences with
   word-model candidates, average normalized model scores on overlaps, and
   fall back to character predictions when there is no overlap;
5. **rank-fusion ensemble:** retain our dev-fitted character/word rank fusion as
   a separate baseline rather than calling it Embible's ensemble;
6. **oracle boundary diagnostic:** supply gold boundaries only to measure the
   ceiling created by boundary information.

Required information regimes:

| Regime | Character budget | Word boundaries | Role |
| :--- | :---: | :---: | :--- |
| U0 | unknown | unknown | Primary real-world condition |
| U1 | unknown | predicted | Character-model segmentation ablation |
| P0 | approximate physical estimate | unknown | Realistic layout condition |
| P1 | approximate physical estimate | predicted | Full character-word ensemble |
| O1 | exact | exact | Oracle ceiling only |

Embible's unconstrained condition assumes a single missing word when whitespace
is unknown. We will not make that assumption for multiword DSS lacunae: U0 must
search over both content and the number of words, and decoding failure counts as
a miss. Consequently our U0 is a DSS-motivated extension of UWC, not a literal
replication of Embible's single-word assumption.

We audited the public [Embible repository](https://github.com/harelm4/Embible)
at root commit `63dc79f1e4240b01883f5fe03e4e3389b8f2bc0d` and backend master
`7c9e769`. The released code does not exactly instantiate the prose protocol:
the paper describes Top-5 character candidates, Top-1,000 word candidates, and
character fallback when their overlap is empty; backend `Char_model.py` retains
up to 100 character completions, `SameLengthAndCharsWordModel.py` starts from a
raw Top-10,000 word pool before filtering, and `ensemble_v2.py` falls back only
when the word list is empty. Our named Embible-overlap condition follows the
paper-described Top-5/intersection/fallback rule, records its smaller candidate
pool, and must therefore be called a **scaled paper-protocol adaptation**, not
an exact reproduction of either the reported experiment or the released code.

For comparability, the implemented matrix hides one, two, and three contiguous
words within eight context words on each side: approximately 5%, 10%, and 15%
of the displayed word sequence. A future direct replication may also use
Embible's two random-masking strategies, but it remains secondary. The primary
Track A benchmark samples contiguous damage rather than uniformly masking
Biblical-style text.

The implemented report includes CharHit@1/5, sequence-derived WordHit@1/5,
exact-span Top-K, CER, whitespace boundary F1, generated word-count error, and
decoder failure rate. Sequence-derived WordHit asks whether a word appears in
the correct position in one of the Top-K complete sequences, so it is stricter
than and not numerically identical to Embible's independent WordHit. Length
penalties and ensemble weight are selected on development scrolls by exact-span
performance, not perplexity. Gold length and boundaries are confined to
explicitly labelled oracle diagnostics.

### 3.5 Fixed-decoder Bible transfer diagnostic

To distinguish a weak completion method from a DSS domain-generalization
failure, run the identical DSS-trained models and unknown-length decoder on
Embible's released Biblical validation/test split. Restore each masked row by
matching it to the canonical unpointed verse at the pinned upstream commit,
exclude ambiguous matches, and then create the same contiguous 1/2/3-word
damage used in the DSS diagnostic. Tune penalties and ensemble weight only on
Biblical validation verses and report the held-out test verses separately.

This diagnostic changes the text domain while holding the task, decoder,
candidate limits, and metrics fixed. Biblical text remains evaluation-only.
It is not a reproduction of Embible's random masking and must not be compared
directly to Embible's published WordHit tables.

## 4. Evaluation tracks

### Track A — synthetic preserved-ink recovery

Construct targets from genuinely preserved text, then apply realistic synthetic
damage. This is the primary known-answer benchmark. Its targets are not real
lacunae, even though the source language is physically preserved DSS text.

Evaluate three information regimes:

1. **Unknown length (primary):** neither the character count nor word-slot count
   is supplied.
2. **Approximate physical constraint:** only information derivable from the
   manuscript, with a predeclared tolerance, is supplied.
3. **Known slot count (diagnostic):** the number of missing word slots is
   supplied for comparison with older masked-LM work.

Report natural-distribution micro averages and equal-bucket macro averages.
Also stratify by scroll, composition, genre, language, gap length, visible
letters, surrounding damage, named entities, and formulaicity.

### Track B — real scholarly lacunae

- Use attributed restorations from Qumran Digital or another licensed,
  versioned source.
- Treat the original missing wording as unobservable: automatic “accuracy” is
  unavailable at these locations.
- Preserve every compatible alternative rather than choosing a single gold
  reading.
- One manuscript location is one primary observation, even when a proposal is
  repeated by several publications.
- Score success when any compatible attributed reading is recovered.
- Use minimum character error rate across compatible alternatives.
- Report single-word and multiword targets separately.
- Name the outcome **literature agreement**, never ground-truth accuracy.
- Publish all exclusion reasons and counts.

### Track C — scholar-assistance study

Use a blinded, counterbalanced within-subject design with four conditions:

1. manuscript context only;
2. ranked model candidates;
3. retrieved parallels;
4. candidates plus parallels.

Use approximately 60–100 real targets sampled across difficulty and gap-length
strata. Record the final proposed reading, exact or minimum CER agreement, time,
confidence, candidate adoption, parallel relevance, and perceived usefulness.
Randomize condition order and candidate order. Measure inter-annotator
agreement and use paired tests.

## 5. Primary and secondary metrics

### Primary

**Exact complete-span Top-10 under unknown length** on Track A. A hit requires
the full sequence in the correct order. Failed decoding counts as a miss.

### Secondary

- exact-span Top-1, Top-5, and Top-20;
- top-1 character error rate;
- minimum CER over acceptable alternatives;
- mean reciprocal rank;
- slot-level Top-K, explicitly labelled diagnostic;
- decoder failure rate and generated-length error;
- calibration, selective accuracy, and abstention coverage;
- latency and memory use for practical deployment.

Never combine slots from different lacunae into a span count. Never include
single-word examples in a “multiword” aggregate.

## 6. RAG protocol

The retrieval corpus contains only preserved text from training scrolls.
Development scrolls may tune retrieval settings but may not be indexed for the
test evaluation.

Run a paired ablation on one frozen test set:

1. base pretrained model;
2. preserved-only fine-tuned model;
3. plus visible-letter constraints;
4. plus approximate physical constraints;
5. plus lexical/BM25 or n-gram retrieval;
6. plus dense retrieval;
7. plus reranking;
8. best predeclared combination.

For every retrieval system report:

- Recall@K of an eligible useful parallel;
- MRR and nDCG for expert relevance judgments;
- coverage and no-result rate;
- downstream accuracy conditional on useful retrieval;
- diversity and near-duplicate rate;
- examples where retrieval helps and hurts.

Add a no-answer-string stress test: remove retrieved passages containing the
reference completion and test whether structural parallels still help. Audit
every test query for answer strings, same-passage versions, and held-out
composition leakage.

## 7. Statistical analysis

- Treat the manuscript location as the independent evaluation unit.
- Use paired comparisons because all systems score the same frozen targets.
- Report absolute deltas with 95% confidence intervals.
- Bootstrap by scroll, not by word slot. For composition-disjoint analysis,
  cluster at composition level.
- Use paired permutation tests or clustered bootstrap intervals for system
  differences.
- Correct for multiple comparisons within each declared family of ablations.
- Publish sample sizes, exclusions, random seeds, and missing-output counts.

Small pilots may be shown descriptively but may not be described as
improvements without uncertainty and a paired test.

## 8. Contamination and leakage audit

Before any paper result:

- verify train/dev/test scroll and composition intersections are empty;
- verify retrieval contains train material only;
- search normalized test answers and centered n-grams in training and retrieval;
- group duplicate or parallel versions before splitting;
- verify no gold character count or slot count reaches an unknown-length
  decoder;
- verify all tuning occurred on development data;
- retain a target-level audit table showing input evidence, candidates,
  retrieved passages, ranks, and exclusion reason;
- record the repository commit and hashes for data, model, and result files.

## 9. Promotion gate

A number may enter the abstract, README headline, or presentation conclusion
only if all are true:

- the experiment is registered in `eval/run_all_experiments.py`;
- its test set is frozen and hashed;
- its unit and information regime are explicit;
- leakage checks pass;
- confidence intervals are present;
- multiword results use exact complete sequences;
- RAG uses a matched paired ablation;
- the claim states whether it is preserved recovery, literature agreement, or
  scholar utility.

## 10. Relationship to prior work

This protocol combines the strongest evaluation practices from:

- [Pythia (EMNLP 2019)](https://aclanthology.org/D19-1668/): character error
  rate, ranked hypotheses, held-out inscriptions, and expert comparison.
- [Blank Language Models (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.420/):
  variable-length completion rather than a fixed number of masks.
- [Babylonian restoration (PNAS 2020)](https://doi.org/10.1073/pnas.2003794117):
  ranking metrics and controlled human questions.
- [Akkadian MLM (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.384/):
  ranked evaluation plus blinded plausibility judgments by specialists.
- [Ithaca (Nature 2022)](https://www.nature.com/articles/s41586-022-04448-z):
  Top-K/CER evaluation and staged human–AI collaboration.
- [MAAT (ML4AL 2024)](https://aclanthology.org/2024.ml4al-1.7/): real lacunae,
  multiple acceptable readings, and known/approximate/unknown-length regimes.
- [Embible (Findings of EACL 2024)](https://aclanthology.org/2024.findings-eacl.56/):
  Hebrew character/word ensembles, separate CharHit@K and WordHit@K, and
  explicit known- versus unknown-whitespace conditions. It fine-tunes on 22,144
  Biblical verses, tunes on 535, and tests on 536 under 5%, 10%, and 15% random
  masking. This makes it a useful Hebrew model and robustness baseline rather
  than evidence for real DSS lacunae.
- [Aeneas (Nature 2025)](https://www.nature.com/articles/s41586-025-09292-5):
  unknown-length restoration, retrieved parallels, and paired expert studies.
- [ARI (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.2148/):
  retrieval ablations, deduplication, real-damage simulation, and blinded
  expert ranking on real damaged documents.

The DSS contribution is the joint application of these controls to
reconstruction-free Second Temple Hebrew and Aramaic, with physical evidence,
alternative scholarly readings, and train-only parallel retrieval kept
separate and auditable.
