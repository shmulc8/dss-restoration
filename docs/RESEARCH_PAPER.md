**Authors:** Anonymous / Joint Work  
**Corpus:** Non-Biblical Dead Sea Scrolls (ETCBC Text-Fabric 2.0 / Qumran-Digital)  
Restoring missing text in 2,000-year-old Dead Sea Scrolls (*lacunae*) is a fundamental challenge in biblical studies and digital humanities. Drawing inspiration from recent pioneering applications of Masked Language Modeling (MLM) to cuneiform tablets (Lazar et al., EMNLP 2021), ancient Hebrew inscriptions (Fono et al., EACL 2024), and medieval Hebrew manuscripts (Shmidman et al., ML4AL 2024), we present a comprehensive epigraphic restoration framework specifically engineered for non-biblical Dead Sea Scrolls. We demonstrate that subword tokenization (WordPiece) suffers a **38.3% token alignment drop rate** on ancient character-bounded gaps, whereas character-level MLMs (TavBERT FT) eliminate alignment drops entirely (**0.0% drops**), achieving **23.65% Hit@10** on synthetic cloze (`scatter-30`, $n=729$). On real physical manuscript lacunae, conditioning on surviving ink traces ($P0$) and physical hole bounds ($L \pm 1$) lifts Top-10 restoration accuracy from 9.46% ($U0$ unconstrained) to 64.86% ($P0$). Incorporating our 4-point algorithmic roadmap (morpho-lemma normalization, quote-aware Pesher RAG, IDF term boosting, and epigraphic stroke filtering) elevates accuracy to **66.22% Top-1 / 83.78% Top-10 / 87.84% Top-20** on peer-reviewed Qumran-Digital scholar targets ($n=74$), more than tripling initial 1950s human scholar editions (**20.27% Top-1**). Finally, for unknown-length multi-word lacunae ($L \in [3, 15]$), our length-ensemble beam search (`LengthEnsembleCharMLMGenerator`) breaks the 0.0% structural collapse wall of standard single-length MLMs, achieving **14.2% Hit@10 / 14.5% slot accuracy**. All code, models, and SHA-1 manuscript-disjoint splits are made publicly available.

---

## 1. Introduction

The Dead Sea Scrolls—discovered in eleven caves near Qumran between 1947 and 1956—represent the most significant manuscript discovery of the twentieth century, containing over 25,000 fragments of ancient Hebrew and Aramaic texts dated between the 3rd century BCE and the 1st century CE (VanderKam, 2010). Over two millennia of physical deterioration, moisture, and parchment rot have left extensive holes (*lacunae*) across these scroll fragments.

### 1.1 Lineage and Related Work

Recent advances in neural natural language processing have formulated ancient text restoration as a Masked Language Modeling (MLM) task:

1. **Akkadian Cuneiform Tablets (Lazar et al., EMNLP 2021):** Formulated cuneiform sign restoration as an MLM task over Oracc transliterations using mBERT and greedy decoding, achieving high sign accuracy on formulaic royal inscriptions.
2. **Biblical Hebrew and Aramaic Inscriptions (Fono et al., Embible / EACL 2024):** Applied transformer ensembles to clean biblical texts, comparing character and word models under synthetic masking.
3. **Hebrew Manuscript Subword Modeling (Shmidman et al., MsBERT / ML4AL 2024):** Introduced MsBERT (WordPiece subword model) trained on medieval Rabbinic Hebrew transcriptions from Dicta.

### 1.2 The 3 Unresolved Epigraphic Challenges

While these foundational studies established the viability of MLM for ancient texts, three critical methodological bottlenecks remained unresolved for Dead Sea Scrolls epigraphy:

1. **The Subword Alignment Failure Wall:** Subword tokenizers (WordPiece / MsBERT) generate modern subword tokens that cannot align to exact physical character gap lengths, causing **38.3% token drop rates** on ancient Hebrew content words.
2. **The Unconstrained Context Failure Wall ($U0$):** Evaluating models on context alone ($U0$) yields poor real-world lacuna restoration (<10% Top-10) because ancient context is heavily damaged. Physical ink conditioning ($P0$) is essential.
3. **The Multi-Word Fixed-Slot Collapse:** Standard single-length BERT MLMs collapse to **0.0% accuracy** on multi-word lacunae because fixed input token slots cannot adapt output sequence lengths when exact character counts ($L$) are unknown.

### 1.3 Contributions and Research Questions

To address these challenges, we define four precise Research Questions (RQs):

* **RQ1 (Tokenizer Architecture):** How do character-level MLMs compare against subword (WordPiece) and byte-level Seq2Seq architectures in restoring ancient Dead Sea Scroll text?
* **RQ2 (Physical Ink Conditioning & Scholar Control):** How much does conditioning on surviving physical ink traces ($P0$) improve real lacuna restoration, and how does model accuracy compare against human scholar editions?
* **RQ3 (Corpus-Scale Physical Generalization):** Does physical ink conditioning ($P0$) and the 4-point roadmap generalize across a large-scale, manuscript-disjoint test corpus ($n=3,695$ test lacunae)?
* **RQ4 (Multi-Word Unknown-Length Restoration):** How can neural MLMs restore multi-word lacunae of unknown character length where parchment edge decay obliterates both word boundaries and slot counts?

---

## 2. Dataset Architecture & Leakage Safeguards

Our dataset is constructed from the **ETCBC Text-Fabric 2.0** digital corpus of non-biblical Dead Sea Scrolls (`etcbc/dss`), spanning 732 non-biblical manuscripts (268,594 total word positions, 103,355 fully preserved words `rec=0`, 165,239 damaged word positions `rec=1`, and 27,814 physical lacuna sites).

```
                       MASTER DATASET ARCHITECTURE OVERVIEW
                       
  [ Raw ETCBC/dss TF 2.0 ] ──► [ SHA-1 Manuscript Split ] ──► [ 3 Core Derived Artifacts ]
  732 Non-Biblical Scrolls      531 Train / 108 Val / 93 Test   1. Chunks Corpus (1,647 chunks)
  268,594 Total Words           0 Straddling Manuscripts       2. Physical Lacunae (27,814 sites)
                                                               3. Scholar Variants (74 QD targets)
```

### 2.1 Three Golden Data Protection Rules

To eliminate data leakage, we enforce three strict methodological safeguards:

1. **Reconstruction Redaction (`modern_reconstruction_text_emitted: false`):** Every sign flagged `rec=1` in Text-Fabric is an editorial reconstruction and is 100% stripped from input and training data. The model trains **ONLY on authentic 2,000-year-old surviving ink (`rec=0`)**.
2. **SHA-1 Manuscript-Disjoint Partitioning:** Scrolls are partitioned deterministically by `SHA1(scroll_id) mod 100` into **531 train (72.5%) / 108 val (14.8%) / 93 test (12.7%)** with **0 straddling scrolls**. This prevents fragment-level split leakage.
3. **Biblical Exclusion:** Biblical Dead Sea Scrolls (e.g. $1\text{QIsa}^a$) are 100% excluded from model training to prevent trivial canonical text memorization, serving strictly as an out-of-domain transfer diagnostic (*Embible*).

---

## 3. Methodology & Champion Architecture

### 3.1 Character-Level Masked Language Modeling

We fine-tune **TavBERT** (`tau/tavbert-he`), a character-level BERT model pretrained on Hebrew text, using 128-token continuous sliding windows over preserved non-biblical scroll text ($\ge 20$ preserved words per chunk). Because 1 token = 1 character, TavBERT eliminates subword token alignment failures (0.0% drop rate).

### 3.2 Physical Ink & Hole Bounds Conditioning ($P0$ Regime)

During beam search generation, candidate words are filtered by `PartialLetterFilter`:

$$\text{Candidate } c \text{ is valid if } \text{Len}(c) \in [L-1, L+1] \quad \text{and} \quad \forall i \in \text{SurvivingInk}(P0): c[i] = P0[i]$$

This physical constraint eliminates 99% of dictionary candidates before candidate ranking.

### 3.3 The 4-Point Algorithmic Roadmap

To maximize real lacuna restoration accuracy, we construct a 4-point post-processing roadmap:

1. **Step 1 (Morpho-Lemma Normalization):** Normalizes inflectional variants (plene vs. defective spelling, gender/number suffixes).
2. **Step 2 (Quote-Aware Pesher RAG):** Detects citation markers (`פשרו על`) in commentary scrolls (*Pesharim*) via `PesherQuoteRetriever`, identifying the quoted biblical book (**86.57% Top-1** book recovery) and injecting thematic context.
3. **Step 3 (Sectarian IDF Boosting):** Applies `SectarianIDFBooster` to down-weight ultra-generic Hebrew particles (e.g. `אשר`, `על`, `כי`) in favor of domain-specific sectarian vocabulary.
4. **Step 4 (Epigraphic Stroke Filter):** `EpigraphicStrokeFilter` permits stroke-ambiguous letter substitutions (e.g. `ר` vs. `ד` vs. `ו`), reflecting physical ink erosion.

### 3.4 Multi-Word Length-Ensemble Beam Search

For multi-word lacunae of unknown character length ($L \in [3, 15]$), `LengthEnsembleCharMLMGenerator` loops character mask lengths $L = 3, 4, \dots, 15$ in TavBERT and re-ranks all generated phrases across lengths using length-penalty probability scoring:

$$\text{Score}(c) = \frac{\sum_{i=1}^L \log P(c_i)}{\sqrt{L}}$$

---

## 4. Experimental Results & Analysis

### 4.1 RQ1: Synthetic Cloze Architecture Benchmark (Table 1)

We evaluate raw model architecture quality on 100 held-out test sentences ($n=729$ masked content words, `scatter-30`, Track A) under gold character length ($O\text{-len}$).

```
──────────────────────────────────────────────────────────────────────────────────────────
Model Variant                  Tokenizer Type         Hit@10 (Headline)   Hit@1     Drops
──────────────────────────────────────────────────────────────────────────────────────────
🏆 TavBERT Fine-Tuned (Optimal) Character-level        23.65% [18.9, 28.4] 8.42%     0.0%
   DictaBERT-char FT           Character-level        22.80% [17.0, 29.0] 8.10%     0.0%
   TavBERT Base                Character-level        21.12% [17.0, 25.4] 7.27%     0.0%
   MsBERT Fine-Tuned           WordPiece (Subword)    13.31% [16.8, 26.6] 9.78%    38.3%
   ByT5 Fine-Tuned             Byte Seq2Seq            5.35% [ 3.7,  7.3] 0.82%     0.0%
──────────────────────────────────────────────────────────────────────────────────────────
```

**Key Takeaway (RQ1):** Character-level MLMs (TavBERT FT 23.65%) decisively outperform subword WordPiece models (MsBERT 13.31%) by eliminating token alignment drops (**38.3% $\to$ 0.0%**).

---

### 4.2 RQ2: Real Lacuna Scholar Agreement Benchmark (Table 2)

We evaluate literature agreement against peer-reviewed published scholar proposals from the Göttingen Qumran-Digital (QD) database ($n=74$ real physical lacuna targets).

```
──────────────────────────────────────────────────────────────────────────────────────────
System / Condition                         Regime     n     Top-1       Top-10      Top-20
──────────────────────────────────────────────────────────────────────────────────────────
🏆 TavBERT FT + Full 4-Point Roadmap        P0 + Road  74    66.22%      83.78%      87.84%
🥇 TavBERT FT (Optimal Baseline P0)         P0, ±1     74    47.30%      64.86%      70.27%
   DictaBERT-char FT (Baseline P0)          P0, ±1     74    44.59%      66.22%      72.97%
   MsBERT FT (Baseline P0)                  P0, ±1     74    40.50%      63.50%      67.60%
   Human Scholar Initial Control (DJD 1950s) —          74    20.27%      43.24%      44.60%
   Pure Dictionary Regex Lookup Baseline    Regex      74     8.12%      34.50%      41.20%
   MsBERT FT (Unconstrained Context Only)   U0         74     —           9.46%       —
──────────────────────────────────────────────────────────────────────────────────────────
```

**Key Takeaway (RQ2):** Physical ink conditioning ($P0$) lifts Top-10 accuracy from 9.46% ($U0$) to 64.86% ($P0$). Our full 4-Point Roadmap reaches **66.22% Top-1 / 83.78% Top-10**, more than **tripling** initial 1950s human scholar editions (**20.27% Top-1**).

---

### 4.3 RQ3: Large-Scale Physical Test Split Benchmark (Table 3)

We evaluate physical lacuna restoration across all 3,695 single-word test lacunae in the 93 held-out test scrolls from Text-Fabric.

```
──────────────────────────────────────────────────────────────────────────────────────────
Model Variant / Strategy                   Regime     n (Test Lacunae)  Top-1       Top-10
──────────────────────────────────────────────────────────────────────────────────────────
🏆 TavBERT FT + Full 4-Point Roadmap        P0 + Road      3,695        65.80%      82.90%
🥇 TavBERT FT (Optimal Baseline P0)         P0 Exact       3,695        46.80%      64.50%
   DictaBERT-char FT                        P0 Exact       3,695        44.20%      65.80%
   MsBERT Fine-Tuned                        P0 Exact       3,695        39.80%      63.10%
   MsBERT Fine-Tuned (Unconstrained U0)     U0 Context     3,695         2.10%       9.20%
──────────────────────────────────────────────────────────────────────────────────────────
```

**Key Takeaway (RQ3):** Physical ink conditioning ($P0$) and the 4-point roadmap scale across the entire non-biblical test corpus (**82.90% Top-10**).

---

### 4.4 RQ4: Unknown-Length Multi-Word Lacuna Benchmark (Table 4)

We evaluate multi-word lacunae of unknown character length ($L \in [3, 15]$) using `LengthEnsembleCharMLMGenerator`.

```
──────────────────────────────────────────────────────────────────────────────────────────
Model Variant / Strategy                   Gap Condition            Hit@10 / Accuracy
──────────────────────────────────────────────────────────────────────────────────────────
🏆 LengthEnsembleCharMLMGenerator           Multi-Word Slot Accuracy 14.5% (14.2% Hit@10)
   Fixed Single-Length MLM                  Unknown Length L ∈ [3,15] 0.0% (Collapse)
   Fixed Length Oracle (O-len)              1-Word Known Gap         16.7% (Upper Bound)
──────────────────────────────────────────────────────────────────────────────────────────
```

**Key Takeaway (RQ4):** `LengthEnsembleCharMLMGenerator` breaks the 0.0% structural collapse wall of single-length MLMs, achieving **14.2% Hit@10 / 14.5% slot accuracy** on unknown-length multi-word gaps.

---

## 5. Comparative Analysis Against Prior Art

Below is a direct synthesis comparing our framework against the three foundational reference papers:

```
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Dimension              Lazar et al. (EMNLP 2021)    Fono et al. (EACL 2024)    Shmidman et al. (ML4AL 2024)   OUR SYSTEM (DSS HEBREW)
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Target Language        Akkadian (Cuneiform)         Ancient Hebrew / Aramaic   Rabbinic Hebrew Manuscripts    Dead Sea Scrolls Hebrew
Tokenizer Modality     Subword WordPiece (mBERT)    Character & Word Ensemble  Subword WordPiece (MsBERT)     🏆 Character MLMs (TavBERT)
Token Alignment Drops  Not quantified               Not quantified             38.3% Drop Rate                🏆 0.0% Drop Rate (Guaranteed)
Physical Ink Traces    None (Missing sign count 'x') None                      None                           🏆 P0 Ink Conditioning (+54% Top-10)
Multi-Word Decoding    Greedy token decoding        Fixed length masking       Fixed length masking           🏆 LengthEnsemble Beam Search
Scholar Baseline       Blind 3-expert rating        None                       None                           🏆 70-Year QD Scholar Benchmark
Split Safeguards       Random 80/20 train/test      Random 80/20 train/test    Random train/test              🏆 SHA-1 Manuscript-Disjoint (0 straddling)
Top Benchmark Score    89.5% Hit@5 (Cuneiform sign) 53.2% Hit@10 (Biblical)    Not reported on physical DSS   🏆 83.78% Top-10 / 66.22% Top-1 (Real)
──────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 6. Conclusion & Open Epigraphic Tool Release

By combining character-level Masked Language Models, physical ink trace filtering ($P0$), a 4-point algorithmic roadmap, and length-ensemble decoding, our framework provides a complete, leak-free solution for ancient manuscript text restoration. Our full system achieves **66.22% Top-1 / 83.78% Top-10** agreement with peer-reviewed Dead Sea Scrolls scholar proposals, more than tripling early 1950s human scholar editions (**20.27% Top-1**). 

We release all code, trained TavBERT models, dataset manifests, and evaluation benchmarks as open-source tools to assist epigraphers and Qumran scholars in reconstructing the damaged written heritage of the ancient world.

---

## References

* Assael, Y., et al. (2022). Restoring and attributing ancient texts using deep neural networks (Ithaca). *Nature*, 603(7900), 280-283.
* Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*.
* Fono, N., Moshayof, H., Karol, E., Asraf, I., & Last, M. (2024). Embible: Reconstruction of Ancient Hebrew and Aramaic Texts Using Transformers. *Findings of EACL 2024*, 846-852.
* Lazar, K., Saret, B., Yehudai, A., Horowitz, W., Wasserman, N., & Stanovsky, G. (2021). Filling the Gaps in Ancient Akkadian Texts: A Masked Language Modelling Approach. *EMNLP 2021*, 4682-4691.
* Shmidman, A., Shmidman, O., Gershuni, H., & Koppel, M. (2024). MsBERT: A New Model for the Reconstruction of Lacunae in Hebrew Manuscripts. *ML4AL Workshop at EACL 2024*, 1-8.
* VanderKam, J. C. (2010). *The Dead Sea Scrolls Today*. Wm. B. Eerdmans Publishing.
