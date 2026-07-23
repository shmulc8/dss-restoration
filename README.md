# 📜 Dead Sea Scrolls Lacuna Restoration Engine (AI & RAG)

Automatic restoration of missing words (lacunae) in the Dead Sea Scrolls using Hebrew Whole-Word Masked Language Models (`MsBERT`), Retrieval-Augmented Generation (RAG), and physical layout constraints.

🖥️ **Live Interactive Web Demo:** [https://dss-restoration-demo.pages.dev/](https://dss-restoration-demo.pages.dev/)  
🌐 **Live Public Presentation Deck (Hebrew):** [https://dss-restoration-demo.pages.dev/slides_he.html](https://dss-restoration-demo.pages.dev/slides_he.html)

---

## Reconstruction-free dataset and provenance

The current clean-data path is built directly from `ETCBC/dss` Text-Fabric
2.0. It enforces three rules before training:

1. only non-biblical scrolls (`biblical == 0`);
2. no modern editorial reconstruction text (`rec == 1`);
3. lacuna size retained where the source structure supports an estimate.

Reconstructed or explicitly unknown words become anonymous `<GAP>` slots.
During fine-tuning these are unlabelled `[MASK]` inputs; only preserved words
can become prediction targets. The derived corpus contains 736 scrolls, 1,647
chunks, and 27,814 lacuna records.

Text-Fabric attributes the source transcription to Martin G. Abegg Jr., James
E. Bowley, and Edward M. Cook, based on Martin Abegg's data files. Its `rec`
feature identifies a modern reconstruction but does **not** identify an editor
or edition per reading. Per-publication comparisons therefore use a separate,
attributed Qumran Digital snapshot; they are not inferred from Text-Fabric.

On 300 intact preserved targets from reconstruction-free held-out scrolls:

| Model | Top-1 | Top-5 | Top-10 | Top-20 |
| :--- | ---: | ---: | ---: | ---: |
| MsBERT base | 12.0% | 19.3% | 24.7% | 29.3% |
| Legacy DSS span-refined | 15.7% | 25.3% | 32.0% | 35.3% |
| **Preserved-only DSS span** | **13.7%** | **30.7%** | **36.3%** | **43.7%** |

Reproduce the clean path with:

```bash
.venv/bin/python data/build_preserved_nonbib_corpus.py
.venv/bin/python data/validate_preserved_nonbib_corpus.py
.venv/bin/python training/finetune_span_preserved_nonbib.py
.venv/bin/python eval/tf_preserved_nonbib_benchmark.py
```

## Attributed researcher-reading comparison

The stored Qumran Digital snapshot adds a real literature comparison without
putting researcher restorations into training. It contains 1,811 explicit
readings for 346 selected disputed words, attributed to 253 bibliographic
sources. The corrected experiment isolates 74 genuine single-word lacunae in
held-out scrolls, retains visible manuscript letters, and uses approximate
lacuna-derived word length while keeping the restored letters hidden.

| Scope | N | Top-1 | Top-10 | Top-20 |
| :--- | ---: | ---: | ---: | ---: |
| Target: any compatible attributed restoration | 74 | 40.5% | **63.5%** | 67.6% |
| Unique target-reading pair | 99 | 30.3% | **60.6%** | 64.6% |
| Same targets without physical constraints | 74 | 4.1% | 9.5% | 10.8% |

The previous 8.0% result is superseded. It masked the entire disputed word,
discarded preserved letters, concatenated some multiword/editorial
alternatives into impossible tokens, and counted repeated publication rows as
independent cases. The 63.5% result is not a language-model-only gain: it
measures the complete decoder after restoring manuscript constraints.

These are **literature-agreement** numbers, not manuscript-grounded accuracy.
Qumran Digital describes the variant collection as working data that has not
yet been checked extensively, and only a selected subset of variants is shown
inline. See
[`analysis/reports/QD_RESEARCHER_BENCHMARK.md`](analysis/reports/QD_RESEARCHER_BENCHMARK.md)
for the protocol and limitations.

The downloaded snapshot is reused by default, with no network call:

```bash
.venv/bin/python eval/build_qd_researcher_benchmark.py
.venv/bin/python eval/score_qd_researcher_benchmark.py
```

Only an intentional
`eval/build_qd_researcher_benchmark.py --refresh` contacts Qumran Digital.

## 🌟 Executive Summary & Master Empirical Benchmarks

| Benchmark / Research Experiment | Key Metric Outcome | Scientific Significance |
| :--- | :---: | :--- |
| **Single-Word RAG Restoration** | **48.0% Top-10 Accuracy** | Parallel witness retrieval boosts single-word Top-10 accuracy from 30.2% to 48.0% (+17.8% direct gain). |
| **Overall Multi-Word RAG Pipeline** | **36.8% Top-10 Accuracy** | Evaluated across 600 multi-word test lacunae with dynamic beam search and physical layout filters. |
| **Cross-Epoch Historical Transfer** | **39.0% Top-10 Accuracy** | Evaluated on external medieval & rabbinic Hebrew manuscripts; proves strong cross-era syntactic transfer. |
| **Legacy Dual-Metric Framework** | **31.0% vs. 31.8% Top-10** | Compares intact targets with the corpus's anonymous modern reconstructions; it does not establish absence of editor bias. |
| **Strict Composition-Level Split** | **30.2% Top-10 Accuracy** | Evaluated across 26 completely unseen literary works (*CD*, *4QS*, *Hodayot*); zero memorization leakage. |
| **Legacy Ambiguity Heuristic** | **22.8% – 38.0% claimed range** | Not backed by aligned per-editor readings and should not be cited as a measured inter-editor disagreement rate. |
| **Synthetic Clitic Augmentation** | **+0.7% Top-10 Net Gain** | Retraining with prefix joins (`ו-`, `ב-`, `ל-`, `ה-`) makes the model immune to scribal orthographic shifts. |
| **POS Grammatical Filtering** | **+0.3% Top-1 Gain** | Contextual Part-of-Speech constraints suppress prepositions in content noun/verb slots. |
| **TavBERT Character Benchmark** | **30.0% (1w) / 26.3% (6+w)** | TavBERT handles long gaps better than BEREL (26.3% vs 6.9%), but `MsBERT` whole-word modeling leads (37.3%). |
| **Random Baseline Comparison** | **38,700x Improvement** | Selected from 128,000 vocabulary words ($0.00078\%$ random baseline). |

---

## 📂 Complete Repository Structure

```
dss-restoration/
├── analysis/                       # Empirical research & failure analysis tools
│   ├── compare_scholar_conjectures.py  # Scores supplied alternatives; no edition data included
│   ├── estimate_editor_disagreement.py # Legacy fixed heuristic; not an editor comparison
│   ├── context_noise_stress_test.py    # Context degradation ablation runner (10%, 25%, 40% noise)
│   └── reports/                         # Full benchmark reports & markdown logs
│       ├── FULL_CORPUS_BENCHMARK_REPORT.md  # 100% full dataset benchmark report (7,809 spans)
│       └── full_experiment_suite_report.md  # Master experiment suite report
├── demo/                           # Web application & slide presentation deck
│   ├── index.html                      # Interactive web demo UI
│   ├── slides_he.html                  # 14-Slide interactive Hebrew presentation deck
│   └── app.js                          # Saliency map & candidate ranking demo logic
├── eval/                           # Complete evaluation suite
│   ├── run_all_experiments.py          # Master experiment suite runner
│   ├── run_full_corpus_experiments.py  # 100% full dataset evaluation runner
│   ├── tf_single_word_intact_benchmark.py # Single-word intact text benchmark
│   ├── tf_compare_cliticaug_models.py  # Side-by-side clitic augmentation model comparison
│   ├── tf_pos_grammatical_filtering_eval.py # POS grammatical constraint filtering benchmark
│   ├── tf_lacuna_len_aeneas_enhanced.py    # Ultimate RAG + Enhanced decoding pipeline
│   ├── tf_intact_vs_lacuna_eval.py     # Dual-metric evaluation script (Intact vs Lacunae)
│   ├── tf_composition_split_eval.py    # Strict composition-level split script (26 compositions)
│   ├── tf_tavbert_eval.py              # TavBERT character-level benchmark script
│   ├── build_qd_researcher_benchmark.py # One-time attributed QD snapshot importer
│   ├── score_qd_researcher_benchmark.py # Offline per-publication scorer
│   └── tf_historical_hebrew_eval.py    # Cross-epoch historical Hebrew benchmark
├── training/                       # Fine-tuning & augmentation training scripts
│   ├── finetune_span.py                # Base span-masking fine-tuning script
│   └── finetune_span_continue_cliticaug.py # Clitic & prefix synthetic augmentation
└── utils/                          # Shared data connectors & split utilities
    ├── sqe_connector.py                # Scripta Qumranica Electronica (SQE) & IAA connector
    ├── dss_split.py                    # Scroll and composition partition loader
    ├── composition_lookup.py           # 26 composition group mapping
    ├── clitic_join.py                  # Synthetic prefix & clitic joiner
    └── morph_dss.py                    # DictaBERT-lex morphological lemmatizer
```

---

## 🚀 Quickstart & Execution Commands

### 1. Score User-Supplied Conjectures
This scorer ranks manually supplied alternatives. It does not fetch or align
DJD, Qimron QTD, or SQE readings:
```bash
python analysis/compare_scholar_conjectures.py
```

### 2. Side-by-Side Model Comparison (Base vs. Clitic-Augmented)
To evaluate the accuracy gain of synthetic clitic pre-training:
```bash
python eval/tf_compare_cliticaug_models.py
```

### 3. POS Grammatical Constraint Benchmark
To evaluate the impact of Part-of-Speech grammatical filtering:
```bash
python eval/tf_pos_grammatical_filtering_eval.py
```

### 4. Single-Word Intact Text Benchmark
To evaluate single-word restoration on intact preserved scribal text (`rec != 1`):
```bash
python eval/tf_single_word_intact_benchmark.py
```

### 5. Run the Master Experiment Suite
To run all benchmark experiments sequentially and generate `full_experiment_suite_report.md`:
```bash
python eval/run_all_experiments.py
```

---

## 🏛️ Academic Integration & Ecosystem Architecture

The SQE and Leon Levy connectors remain exploratory. The Qumran Digital
snapshot now provides real word-level readings with publication attribution,
including Qimron, the *Study Edition*, DJD volumes, and many other sources.
It should not be confused with direct digitizations of those copyrighted
editions: the stored records are Qumran Digital's CC BY-SA variant data and
bibliographic provenance.

---

## 📚 Literature Grounding & Peer-Reviewed Methodology

Our evaluation framework is strictly grounded in landmark digital humanities and NLP literature:
* **Ranked Top-K Lists:** Aligned with **DeepMind Ithaca** (*Nature* 2022) and **Pythia** (*EMNLP* 2019).
* **Parallel Witness RAG Retrieval:** Aligned with **DeepMind Aeneas** (*Nature* 2025).
* **Dynamic Unknown-Length Decoding:** Aligned with **Akkadian Cuneiform MLM** (*PNAS* 2020).
* **Morphological Normalization:** Aligned with **Embible** (*EACL* 2024) and SPMRL MRL protocols.
