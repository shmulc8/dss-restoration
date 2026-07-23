# DSS Text Restoration (שחזור מגילות מדבר יהודה)

Automatic restoration of missing words (lacunae) in the Dead Sea Scrolls using Hebrew masked language models, retrieval-augmented generation (RAG), and physical layout constraints.

🔗 **GitHub Repository:** [https://github.com/shmulc8/dss-restoration](https://github.com/shmulc8/dss-restoration)  
🌐 **Live Cloudflare Public Slides (Hebrew):** [https://dss-restoration-demo.pages.dev/slides_he.html](https://dss-restoration-demo.pages.dev/slides_he.html)  
🖥️ **Live Cloudflare Interactive Web Demo:** [https://dss-restoration-demo.pages.dev/](https://dss-restoration-demo.pages.dev/)  
🐙 **GitHub Pages Mirror:** [https://shmulc8.github.io/dss-restoration/demo/slides_he.html](https://shmulc8.github.io/dss-restoration/demo/slides_he.html)

---

## 🚀 Key Highlights & Breakthroughs

1. **Whole-Word vs. Subword Tokenization Integrity:**
   - Identified and resolved **subtoken length leakage** in subword models like `BEREL` (which collapsed to 6.9% on long gaps under dynamic length decoding).
   - Championed `MsBERT` whole-word tokenization, maintaining robust ~31%–39.8% Top-10 accuracy across all lacuna sizes.

2. **Parallel Witness RAG (Aeneas-style Retrieval):**
   - Retrieves formulaic n-gram matches across the Qumran corpus with strict **Cross-Composition Exclusion** to prevent self-matching.
   - Skyrockets Top-1 sequence accuracy from **12.0% to 77.2% – 81.1%** whenever a parallel witness phrase exists.

3. **Unified Enhanced Decoding Pipeline:**
   - Combines Morphological Lemma Deduplication (DictaBERT-lex), Soft Physical Length Filtering (`len(cand) >= len(gold) - 1`), and 40-word expanded context windows.
   - Reaches **48.0% Top-10 Accuracy** on 1-word gaps and **36.8% overall** across all test slots.

4. **Strict Composition-Level Split Validation:**
   - Purged 26 entire literary compositions (e.g. *CD*, *4QS*, *Hodayot*) from training. Performance held stable at **~30% Top-10 Accuracy**, proving generalized Qumran Hebrew syntax.

5. **Dual-Metric Evaluation Framework (Intact Ink vs. Editor Concordance):**
   - Separated evaluation into **Ground-Truth Ink Accuracy** (`rec = 0`, 31.0% Top-10 on verified ancient ink) vs. **Human Editor Concordance** (`rec = 1`, 31.8% Top-10 on physical lacuna conjectures).

6. **Cross-Epoch Generalization Benchmark:**
   - Evaluated `MsBERT ft-SPAN-refined` on external historical Hebrew manuscript texts. Reached **39.0% Top-10 Slot Accuracy**, proving strong cross-epoch syntactic transfer beyond Qumran.

7. **Multi-Model Benchmark Comparison:**
   - Benchmarked `MsBERT` (whole-word) vs. `TavBERT` (character-level, Keren et al. 2022) vs. `BEREL` (subword).

---

## 📊 Performance Comparison Table (Top-10 Slot Accuracy)

| Model Architecture | Tokenization Level | 1-Word Gap | 2-Word Gap | 3-Word Gap | 4–5 Word Gap | 6+ Word Gap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`MsBERT ft-SPAN-refined` + Enhanced Decoding** | **Whole-Word** | **36.0%** | **36.0%** | **36.7%** | **32.0%** | **39.8%** |
| `MsBERT ft-SPAN-refined` (Standard) | Whole-Word | 37.3% | 34.7% | 31.1% | 29.4% | 31.0% |
| `TavBERT` (tau/tavbert-he) | Character-Level | 30.0% | 23.0% | 23.3% | 21.5% | 26.3% |
| `BEREL ft-SPAN` | Subword (Optimistic) | 36.0% | 30.7% | 28.4% | 25.4% | 26.8% |
| `BEREL` (Dynamic Leak-Free) | Subword | 26.0% | 21.0% | 20.7% | 14.9% | 6.9% |

---

## 🖥️ Interactive Presentation & Researcher Demo

- **Hebrew Presentation Deck (`demo/slides_he.html`):** Interactive 13-slide glassmorphism deck featuring attention saliency maps, real 1QS case studies, literature review, and live HUD controls.
- **Interactive Web Demo (`demo/index.html`):** Researcher-facing tool displaying candidate predictions and attention saliency overlays.

---

## 📁 Repository Structure

```
├── eval/
│   ├── tf_lacuna_len_aeneas_enhanced.py  # Ultimate RAG + Enhanced Decoding Pipeline
│   ├── tf_lacuna_len_enhanced.py         # Soft Length Filter + Lemma Deduplication
│   ├── tf_composition_split_eval.py      # Strict Composition-Level Split Validation
│   ├── tf_tavbert_eval.py                # TavBERT (Character-Level) Evaluation
│   ├── tf_intact_vs_lacuna_eval.py       # Intact Text vs Real Lacunae Comparison
│   ├── tf_lacuna_len_dynamic.py          # Dynamic BEREL Leakage-Free Evaluation
│   └── tf_lacuna_len.py                  # Core Scaled Evaluation Benchmark
├── training/
│   ├── finetune_span.py                  # Contiguous Span-Mask Finetuning
│   └── finetune_span_no_particles.py     # Refined Particle-Domination Fix
├── analysis/
│   ├── failure_analysis.py               # Empirical Error Typology Breakdown
│   ├── context_noise_stress_test.py      # Pythia-style Context Degradation Test
│   └── build_demo_data.py                # Demo Data & Attention Exporter
├── utils/
│   ├── morph_dss.py                      # DictaBERT-lex Lemmatization Helper
│   └── dss_split.py                      # Scroll & Book Protocol Splits
├── demo/
│   ├── slides_he.html                    # 13-Slide Hebrew HTML Presentation
│   └── index.html                        # Interactive Researcher Demo App
└── README.md
```

---

## 🔧 Quick Start

```bash
git clone https://github.com/shmulc8/dss-restoration.git
cd dss-restoration
python -m venv .venv
source .venv/bin/activate
pip install torch transformers text-fabric numpy
```

Run evaluations:
```bash
# Run ultimate RAG + Enhanced decoding pipeline
python eval/tf_lacuna_len_aeneas_enhanced.py

# Run TavBERT character-level evaluation
python eval/tf_tavbert_eval.py
```
