# 📜 Dead Sea Scrolls Lacuna Restoration Engine (AI & RAG)

Automatic restoration of missing words (lacunae) in the Dead Sea Scrolls using Hebrew Whole-Word Masked Language Models (`MsBERT`), Retrieval-Augmented Generation (RAG), and physical layout constraints.

🖥️ **Live Interactive Web Demo:** [https://dss-restoration-demo.pages.dev/](https://dss-restoration-demo.pages.dev/)  
🌐 **Live Public Presentation (Hebrew):** [https://dss-restoration-demo.pages.dev/slides_he.html](https://dss-restoration-demo.pages.dev/slides_he.html)

---

## 🌟 Executive Summary & Key Results

| Benchmark / Experiment | Key Metric Outcome | Significance |
| :--- | :---: | :--- |
| **Single-Word RAG Restoration** | **48.0% Top-10 Accuracy** | Parallel witness retrieval boosts single-word Top-10 accuracy from 30.2% to 48.0% (+17.8% gain). |
| **Overall Multi-Word RAG Pipeline** | **36.8% Top-10 Accuracy** | Evaluated across 600 multi-word test lacunae with dynamic beam search. |
| **Cross-Epoch Historical Transfer** | **39.0% Top-10 Accuracy** | Evaluated on external medieval & rabbinic Hebrew manuscripts; proves strong syntactic transfer. |
| **Dual-Metric Framework** | **31.0% vs. 31.8% Top-10** | Intact Ink Accuracy (`rec = 0`) matches Real Lacunae Editor Concordance (`rec = 1`), proving no editor bias. |
| **Strict Composition-Level Split** | **30.2% Top-10 Accuracy** | Evaluated across 26 completely unseen literary works (*CD*, *4QS*, *Hodayot*); zero memorization leakage. |
| **Inter-Editor Disagreement Rate** | **22.8% – 38.0% Ambiguity** | Analyzed across 25,155 lacuna words; proves Second Temple Hebrew routinely allows multiple valid synonyms. |
| **Random Baseline Comparison** | **38,700x Improvement** | Selected from 128,000 vocabulary words ($0.00078\%$ random baseline). |

---

## 📂 Repository Structure

```
dss-restoration/
├── analysis/                       # Empirical research & failure analysis tools
│   ├── compare_scholar_conjectures.py  # Automated Comparative Epigraphic Scorer
│   ├── estimate_editor_disagreement.py # Inter-editor disagreement & ambiguity estimator
│   ├── context_noise_stress_test.py    # Context degradation ablation runner
│   └── reports/                         # Full benchmark reports & JSON logs
│       ├── FULL_CORPUS_BENCHMARK_REPORT.md  # 100% full dataset benchmark report
│       └── full_experiment_suite_report.md  # Master experiment suite report
├── demo/                           # Web application & slide presentation deck
│   ├── index.html                      # Interactive web demo UI
│   ├── slides_he.html                  # 14-Slide interactive Hebrew presentation deck
│   └── app.js                          # Saliency map & candidate ranking demo logic
├── eval/                           # Benchmark evaluation suite
│   ├── run_all_experiments.py          # Master experiment suite runner
│   ├── run_full_corpus_experiments.py  # 100% full dataset evaluation runner
│   ├── tf_single_word_intact_benchmark.py # Single-word intact text benchmark
│   ├── tf_lacuna_len_aeneas_enhanced.py    # Ultimate RAG + Enhanced decoding pipeline
│   ├── tf_intact_vs_lacuna_eval.py     # Dual-metric evaluation script
│   ├── tf_composition_split_eval.py    # Strict composition-level split script
│   ├── tf_tavbert_eval.py              # TavBERT character-level benchmark script
│   └── tf_historical_hebrew_eval.py    # Cross-epoch historical Hebrew benchmark
├── training/                       # Fine-tuning & augmentation training scripts
│   ├── finetune_span.py                # Base span-masking fine-tuning script
│   └── finetune_span_continue_cliticaug.py # Clitic & prefix synthetic augmentation
└── utils/                          # Shared data connectors & split utilities
    ├── sqe_connector.py                # Scripta Qumranica Electronica (SQE) & IAA connector
    ├── dss_split.py                    # Scroll and composition partition loader
    ├── composition_lookup.py           # 26 composition group mapping
    └── morph_dss.py                    # DictaBERT-lex morphological lemmatizer
```

---

## 🚀 Quickstart & Execution Guide

### 1. Run the Master Experiment Suite
To run all 6 benchmark experiments in sequence:
```bash
python eval/run_all_experiments.py
```

### 2. Score Competing Scholar Conjectures
To test the Automated Comparative Epigraphic Scorer on competing reconstructions (e.g. DJD vs. alternative conjectures):
```bash
python analysis/compare_scholar_conjectures.py
```

### 3. Run Single-Word Restoration Benchmark
To evaluate single-word completion on intact preserved scribal text (`rec != 1`):
```bash
python eval/tf_single_word_intact_benchmark.py
```

### 4. Run 100% Full-Corpus Evaluation
To evaluate across all 7,809 lacuna spans in the entire Qumran non-biblical corpus:
```bash
python eval/run_full_corpus_experiments.py
```

---

## 🏛️ Academic Integration & External Corpora

* **Scripta Qumranica Electronica (SQE):** Interfaces with SQE TEI-XML digital scholarly editions (Prof. Eshbal Ratzon / Göttingen / IAA) to output candidate completions directly into digital scroll editors.
* **Leon Levy Digital Library (IAA):** Maps scroll plate numbers to high-resolution multispectral infrared (IR) imagery for physical line-width constraints.
* **Qumran Text Database (QTD):** Compares predictions against Elisha Qimron's 4-volume critical Hebrew edition (Ben-Gurion University Press).

---

## 📚 Literature Grounding & Methodology

Our evaluation framework is strictly grounded in landmark digital humanities and NLP literature:
* **Ranked Top-K Lists:** Aligned with **DeepMind Ithaca** (*Nature* 2022) and **Pythia** (*EMNLP* 2019).
* **Parallel Witness RAG Retrieval:** Aligned with **DeepMind Aeneas** (*Nature* 2025).
* **Dynamic Unknown-Length Decoding:** Aligned with **Akkadian Cuneiform MLM** (*PNAS* 2020).
* **Morphological Normalization:** Aligned with **Embible** (*EACL* 2024) and SPMRL MRL protocols.
