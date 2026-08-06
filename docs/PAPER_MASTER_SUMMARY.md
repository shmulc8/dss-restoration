# 📜 Dead Sea Scrolls Text Restoration — Master Reference & Skeptical Reader Guide

> [!IMPORTANT]
> **Orientation.** This document explains what problem the project solves, where standard models break, how
> the pipeline works, what safeguards prevent leakage, and — section by section — exactly which numbers are
> measured and which are not.

> [!WARNING]
> **Evidence status.** Every figure below is traceable to a generated artifact, named at the point of use.
> Per [`docs/RESULTS.md`](RESULTS.md), **no result in this project is yet a frozen paper result**. The
> real-lacuna benchmark is a literature-agreement pilot at $n=74$; the cloze benchmark is a synthetic-damage
> diagnostic; several pipeline components are implemented but have never been benchmarked. Claims are tagged
> **[measured]**, **[pilot]**, or **[not yet measured]**.

---

## 🧭 1. Project Primer

### 1.1 What Problem Are We Solving?
The Dead Sea Scrolls, discovered in the caves around Qumran, are among the most important surviving
manuscripts of antiquity. After two thousand years the parchment and leather have decayed and torn, leaving
gaps — *lacunae* — in the text. Editors fill these gaps by scholarly conjecture, a process that has taken
decades and often leaves competing proposals: the Qumran-Digital variant database records **1,811 published
reading-proposal rows** across the sites we sampled.

The question this project asks is narrow and answerable: **how much does the surviving physical evidence at a
lacuna — the partial ink still on the parchment, and the size of the hole — improve automatic restoration?**

### 1.2 Where Do Standard Masked Language Models Break?
1. **Sub-word tokenization cannot express a character-bounded gap.** A WordPiece model predicts vocabulary
   entries, not characters. On the cloze benchmark, MsBERT cannot emit a token sequence matching the gap for
   **279 of 729 words (38.3%)**. **[measured]**
2. **Byte-level generation ignores the length budget.** A fine-tuned ByT5 decodes autoregressively with no
   gap-length constraint and reaches **5.35% Hit@10**. **[measured]**
3. **Random splits leak manuscripts.** Fragments of one scroll landing in both train and test let a model
   memorize rather than restore. This project partitions strictly by manuscript ID.

A character-level MLM avoids the first two problems structurally — 1 token = 1 character, so any gap length is
expressible and **0 of 729** words go unaligned. That is an *enabling* property, not a large accuracy win:
TavBERT base reaches 21.12% against MsBERT's 13.31% all-words score, with overlapping intervals.

### 1.3 What Is the Core Contribution?
- **Manuscript-disjoint partitioning** of 732 non-biblical scrolls, 0 straddling scrolls. **[measured]**
- **Reconstruction redaction**: every sign flagged `rec=1` in Text-Fabric is an editorial reconstruction and is
  never emitted into model input, so the model conditions only on `rec=0` physical ink. **[verified]**
- **Physical conditioning of a character decoder** on surviving ink plus approximate gap length, which on 74
  real lacunae raises Top-10 literature agreement from **9.5% to 63.5%** — a **+54.0 pp** effect. **[pilot]**

That last figure is the project's one large measured effect, and it should be framed for what it is: a
measurement of how much the *physical evidence* contributes, not of how good the language model is.

### 1.4 Do We Need a Neural Encoder At All?
A fair challenge: if the ink pattern already restricts candidates to a small set, why not just look them up?

The argument for the encoder is structural rather than empirical at this stage:

1. **Ranking among ink-compatible candidates.** A pattern such as `⬚⬚לעים` is satisfied by several real Hebrew
   words — in the observed test case the model's ranked output was `הקלעים`, `קלעים`, `סלעים`, `הסלעים`,
   `תולעים`, while the attributed reading was `מבלעים`. A lexicon lookup has no way to order these; an encoder
   can at least condition on the surrounding syntax.
2. **Gaps with no surviving ink.** **70.3% of damaged word positions retain no ink at all** (and 15.0% of
   lacunae contain no traced word anywhere). For those, pattern matching is vacuous and context is the only
   available signal.

> [!NOTE]
> **[not yet measured]** No dictionary/trie baseline has been run against the QD benchmark, so this section
> claims no comparative numbers. Building that baseline is a cheap and worthwhile addition — it is the natural
> control a reviewer will ask for, and the encoder's value is currently argued rather than demonstrated.

### 1.5 Why Does the Synthetic Track Mask Words and the Physical Track Mask Letters?
Two tracks measure two different things and must never be chained into one narrative.

- **Track A — synthetic cloze (`scatter-30`).** Hides 30% of complete content words in *preserved* text.
  Measures cloze capacity under heavy context loss. This is synthetic damage, not a real lacuna.
- **Track B — real lacunae (`lacuna-real`).** Real manuscript damage cuts *through* words, dropping some
  letters and leaving others. Measures restoration where partial ink survives.

The masking unit differs because the damage does. Track A numbers (≈21% Hit@10) and Track B numbers (63.5%
Top-10) are different tasks on different data and are not comparable.

---

## 🏛️ 2. Executive Summary

| # | Finding | Status |
|---|---|---|
| 1 | **Physical evidence dominates.** On 74 real lacunae, the same model and targets go from **9.5% Top-10 (U0, context only)** to **63.5% Top-10 (P0, ink + approximate length)** — **+54.0 pp**. 95% cluster bootstrap on the P0 figure: **51.4%–74.3%**. | **[pilot]** |
| 2 | **Sub-word tokenizers fail structurally on character-bounded gaps.** MsBERT cannot align on **279/729 (38.3%)** of masked words. Aligned-only scoring reports 21.56%; all-words scoring reports **13.31%**. | **[measured]** |
| 3 | **Fine-tuning does not currently help.** On the 729-word cloze benchmark TavBERT base scores **21.12%** against **20.58%** fine-tuned; at $n=250$, 24.40% base against 22.40% fine-tuned. | **[measured]** |
| 4 | **Scale does not help either.** DictaBERT-char at ~88M and DictaBERT-Large-char at ~307M both score **17.60%** Hit@10 at $n=250$. | **[measured]** |
| 5 | **Retrieval has produced no restoration gain.** On the 74 real lacunae, adding train-only RAG changes Top-1, Top-10 and Top-20 by **exactly 0.0 pp**. It does recover the quoted biblical book on Pesher passages at **86.6% Top-1**, which is a different task. | **[measured]** |
| 6 | **Unknown-length multi-word restoration is unsolved.** With the word-slot count supplied, whole-gap Top-10 is **7.0%** (9.0% with RAG). Without it, no benchmark has been run. | **[measured] / [not yet measured]** |

### 2.1 Publication Scope

**Main body**

| Table | Content | Status |
|---|---|---|
| 1 | Synthetic cloze, `scatter-30`, $n=729$ words | **[measured]** |
| 2 | Real lacuna literature agreement, QD $n=74$ | **[pilot]** |
| 3 | Multi-word lacunae, known slot count | **[measured]** |
| 4 | Physical lacuna corpus statistics | **[measured]** — counts only, no model evaluation |

**Appendix**

| Table | Content | Status |
|---|---|---|
| A1 | Model selection and pretraining scale, $n=250$ words | **[measured]** |
| A2 | Quote-aware Pesher source retrieval | **[measured]** |
| A3 | Canonical dataset split | **[measured]** |

### 2.2 The 12 Unification Decisions — Implementation vs. Evidence

"Implemented" means the code exists and is unit-tested. It does **not** mean the component has been evaluated
end-to-end. The two columns are deliberately separate.

| # | Topic | Chosen State | Code | Evidence |
|---|---|---|---|---|
| 1 | Dataset splits | Manuscript-disjoint SHA-1 partitioning, 531 / 108 / 93, 0 straddling | Implemented | **[verified]**, but see the two-registry conflict in §6 |
| 2 | Baseline model | TavBERT (`tau/tavbert-he`) as the cloze reference model | Implemented | **[measured]** — base leads fine-tuned |
| 3 | Masking protocol | Dual-track: `scatter-30` synthetic vs. `lacuna-real` physical | Implemented | **[measured]** on both tracks |
| 4 | Alignment metric | All-words headline metric ($unaligned = miss$) | Implemented | **[measured]** |
| 5 | Regime scoping | Explicit U0 / O-len / P0 regimes | Implemented | **[measured]** |
| 6 | Retrieval scope | Quote-aware Pesher source retrieval | Implemented | **[measured]** as source recovery; **no measured restoration gain** |
| 7 | Multi-word gaps | `LengthEnsembleCharMLMGenerator`, $L \in [3,15]$ | Implemented, unit-tested | **[not yet measured]** — never benchmarked |
| 8 | Redaction | `rec=1` signs never emitted into model input | Implemented | **[verified]** in the corpus manifest |
| 9 | Fine-tuning strategy | Validation early stopping, $1\times10^{-5}$ LR | Implemented | **[not yet measured]** — no early-stopped run has been scored |
| 10 | Uncertainty | Percentile cluster bootstrap, $B=1000$; permutation tests for the retrieval ablation | Implemented | **[measured]**. `mcnemar_test()` exists but has never been applied to a reported comparison |
| 11 | Data licensing | `ETCBC/dss` via Text-Fabric (**CC BY-NC 4.0**) + cached QD snapshot | Implemented | **[verified]** — note the non-commercial term |
| 12 | Reporting | Evidence register + presentation deck, both guarded by `tests/test_public_claims.py` | Implemented | **[verified]** |

### 2.3 What Is the Pesher Source Retrieval Module?

A **Pesher** (פֶּשֶׁר) is an ancient sectarian commentary following a fixed formula: quote a biblical verse,
introduce the interpretation with `פשרו על` ("its interpretation concerns…"), then read the verse as a
prophecy about the Qumran community.

Generic retrieval across the whole corpus injects biblical language into non-commentary texts, so retrieval is
restricted to Pesher passages. The module detects the citation formula and matches the quotation against the
24 books of the Hebrew Bible by character n-gram similarity.

**Result [measured]:** **86.6% Top-1 / 99.1% Top-3** book recovery, macro-averaged by Pesher manuscript over
35 known-source passages from eight mapped Pesharim. See §8, Table A2 for the leakage control.

---

## 📜 3. Real Lacunae & Error Analysis

### 3.1 Five Real Targets from the Canonical Test Split

All five scrolls below sit in the **canonical test split**, and every prediction is the model's actual ranked
output from [`analysis/reports/qd_researcher_comparison.json`](../analysis/reports/qd_researcher_comparison.json).
Predictions come from `ft_msbert_span_preserved_nonbib` under P0 (±1 character). Two of the five are not Top-1
hits — this is a representative sample, not a highlight reel. **[pilot]**

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. 4Q303 frg. 1, l. 10  (Meditation on Creation A)                                               │
│    Context  : "... לו עזר כ⬚⬚⬚⬚"          Surviving ink: 'כ',  estimated length 5                 │
│    Top-1    : כנגדו   (ke-negdo, "corresponding to him")                    ✅ Top-1 hit          │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. 4Q185 frg. 1-2 i, l. 14  (Sapiential Work)                                                    │
│    Context  : "... תמו מן ⬚בורת אלהים וזכרו"   Surviving ink: 'בורת', estimated length 5           │
│    Top-1    : גבורת   (gevurat, "the might of")                             ✅ Top-1 hit          │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. 4Q185 frg. 1-2 i, l. 9                                                                        │
│    Context  : "... באש להבה ישפט⬚"          Surviving ink: 'ישפט', estimated length 5              │
│    Top-1    : ישפט    (yishpot, "he will judge")                            ✅ Top-1 hit          │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. 1QpHab col. IV, l. 11  (Pesher Habakkuk)                                                      │
│    Context  : "... בעצת בית אשמ⬚⬚"          Surviving ink: 'אשמ',  estimated length 5              │
│    Top-1    : אשמות                       Attributed: אשמים / אשמתם                               │
│                                                                    ⚠️ rank 2, wrong inflection    │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. 4Q163 frg. 4-7 i, l. 7  (Pesher Isaiah C)                                                     │
│    Context  : "... ⬚⬚לעים על כנ ..."        Surviving ink: 'לעים', estimated length 6             │
│    Top-1    : הקלעים                      Attributed: מבלעים                                      │
│                                             ❌ miss — קלעים / סלעים / תולעים all fit the trace     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!NOTE]
> "Attributed" means a *modern published proposal* recorded by Qumran-Digital, not physical ground truth — the
> original wording is lost. A hit means the model agreed with a bibliographically attributed restoration.

### 3.2 Where Do the Remaining Errors Come From?

Under the P0 baseline, **36.5%** of the 74 targets miss Top-10. Four failure modes are identifiable by manual
inspection of held-out misses:

1. **Orthographic and morphological variants** — plene/defective spelling (`לוא` vs `לא`) or a different
   inflection of the right root (`אשמות` vs `אשמים`, example 4 above).
2. **Severe context degradation** — adjacent words also missing. Confirmed as first-order by the ablation in
   §3.3.
3. **Rare sectarian vocabulary and hapax legomena** — targets occurring once in the corpus, where the model
   prefers common words.
4. **Ambiguous ink traces** — several real words satisfy the surviving strokes and the context cannot
   discriminate (example 5 above).

> [!WARNING]
> **[not yet measured]** The per-category *shares* are not known. The diagnostic CSV that once supported a
> breakdown is no longer in the repository, so no percentages are claimed. Re-deriving this taxonomy on the
> current model is a prerequisite for the paper's error-analysis section.

### 3.3 Context Degradation Stress Test **[measured]**

Real lacunae rarely sit in clean text. This ablation randomly masks a fraction of the surrounding context
words on 120 held-out cases (≤30 per gap-length bucket).
Artifact: [`analysis/reports/context_noise_ablation.md`](../analysis/reports/context_noise_ablation.md).

| Context noise | Slot-level Top-1 | Sequence-level Top-1 |
|---|---|---|
| 0% | 12.2% | 7.5% |
| 10% | 11.9% | 8.3% |
| 25% | 8.7% | 5.8% |
| 40% | 7.4% | 2.5% |

From 0% to 40% noise, slot accuracy falls ~40% relative and sequence accuracy ~67%. Context degradation is a
measured, first-order limitation — which is precisely why the physical ink channel matters: it survives when
the context does not.

### 3.4 Four-Point Roadmap — Implemented, Not Yet Evaluated

> [!WARNING]
> **[not yet measured]** All four components exist in `eval/candidate_generator.py` and
> `eval/metrics_runner.py` with unit tests, but **none has been run against a benchmark**, so no accuracy is
> attributable to any of them. Two of the four need rework before they could support a published claim.

1. **Lemma-normalised scoring** (`normalize_hebrew_lemma()`) — collapses final letter forms, common plene
   spellings and some affixes. This **relaxes the metric**; it does not improve the model, and must be
   reported as secondary lemma-level agreement alongside exact match.
2. **Retrieval-augmented context injection** — adds column headers, citation formulas and retrieved keywords
   when local context is destroyed. The one retrieval variant already measured on real lacunae produced a
   **0.0 pp** change, so a positive result cannot be assumed.
3. **`SectarianIDFBooster`** — **needs rework.** It is a hand-written lexicon of 15 terms with hand-assigned
   weights (2.0–3.5) matched by substring. It computes no inverse document frequency and reads no corpus
   counts, so the name is inaccurate. A publishable version needs genuine IDF over the training split, and its
   lexicon must not overlap the evaluation targets.
4. **`EpigraphicStrokeFilter`** — **needs rework.** Four hand-written letter groups and a single similarity
   constant of 0.85, applied with a 0.85 threshold, so *any* substitution within a group passes
   unconditionally. It is a relaxed matcher, not a probability matrix, and it will admit readings the surviving
   ink excludes. See §7.8.

---

## 🔬 4. Walkthrough & Reviewer Traps

### 4.1 One Target, End to End

Traced through the pipeline using a real canonical-test target (4Q303 frg. 1, l. 10):

```
STEP 1: DIGITAL EDITION AS PRINTED
  Context      : "... אעשה לו עזר"
  Damaged word : "כ[נגדו"        Square brackets mark the editor's reconstruction (rec=1)

STEP 2: RECONSTRUCTION REDACTION
  Rule         : signs with rec=1 are never emitted; the gap length is kept as metadata
  Model input  : "כ⬚⬚⬚⬚"         Surviving ink 'כ' at position 1; estimated length 5

STEP 3: CONSTRAINED CANDIDATE GENERATION
  Position 1   : forced to 'כ' by the surviving ink
  Positions 2-5: ranked by the model, filtered to length 5 ± 1

STEP 4: RANKED OUTPUT
  1. כנגדו     <- matches the attributed reading (Gen 2:18 עזר כנגדו)
  2. כנגדה
  3. כמהו
  4. כנגד
  5. כאשר
```

Note what the physical channel does here: one surviving letter plus a length estimate reduces the space enough
that the context can pick the right word. Note also what it does not do — for the 70.3% of damaged positions
with no surviving ink, step 3 has nothing to constrain.

### 4.2 Reviewer Traps

| Question | Answer |
|---|---|
| **"Did the model read the editors' guesses?"** | No. Text-Fabric flags editorial reconstructions as `rec=1` and the corpus builder never emits that text (`modern_reconstruction_text_emitted: false`). Fine-tuning data contains no bracketed restorations. For Table 2 the *reference* readings come from a separate source (QD publication rows), not from the input text. |
| **"Did training scrolls leak into test?"** | Not within the canonical registry: all 732 scrolls are assigned by manuscript ID, 0 straddling. **But** a second, incompatible registry exists and was used for the QD run, putting 40 of its 74 targets in the canonical train split. §6 and Table 2 report this and the clean subsets. |
| **"Why is MsBERT's aligned score 21.56% but its headline 13.31%?"** | Because aligned-only scoring silently drops the 279/729 words (38.3%) where WordPiece cannot express the gap. An editor needs a proposal for every gap, so unaligned counts as a miss. |
| **"Is ~21% Hit@10 good on single-word cloze?"** | It is weak, and it is presented as a diagnostic rather than a headline. Roughly a third of the sentence is missing at once and Hebrew morphology admits many valid inflections per slot. But no model in Table 1 is separated from its neighbour at 95%, and fine-tuning does not beat the base checkpoint. The result worth defending is the physical-conditioning effect, not the cloze number. |
| **"Can you handle multi-word gaps of unknown length?"** | Not yet. With the slot count supplied, whole-gap Top-10 is 7.0% (9.0% with RAG); on the synthetic-span diagnostic every system scores 0% on two- and three-word spans. `LengthEnsembleCharMLMGenerator` is implemented and unit-tested but never benchmarked. |

---

## 📂 5. Data Provenance

| Data layer | Origin & attribution | Version / snapshot | Path | Role |
|---|---|---|---|---|
| **ETCBC Dead Sea Scrolls corpus** | ETCBC (Vrije Universiteit Amsterdam), distributed via Text-Fabric. Transcription and features by **Martin G. Abegg Jr., James E. Bowley, Edward M. Cook**. **Licence: CC BY-NC 4.0.** | Corpus `ETCBC/dss`, data version **2.0** | `DSS_TF_DIR` | Source text for 732 non-biblical scrolls with `rec` flags (0 = ink, 1 = reconstruction). |
| **Qumran-Digital variant database** | Qumran Digital: Text and Lexicon — Göttingen Academy of Sciences and Humanities | Cached snapshot **2026-05-21**; the scorer makes no network requests | `data/derived/qd_researcher_variants.jsonl` | 1,811 proposal rows → 74 eligible targets with attributed restorations. |
| **Derived lacuna dataset** | This project's Text-Fabric parser | SHA-256 `41206bc6…f7caff6b8` | `data/derived/nonbib_lacunae.jsonl` | 27,814 lacunae; 165,239 damaged and 103,355 preserved word positions. |
| **Preserved-text chunk corpus** | `data/build_preserved_nonbib_corpus.py` (seed 0, ≥20 preserved words per chunk) | SHA-256 `4c73c58c…044a517f` | `data/derived/preserved_nonbib_chunks.jsonl` | 1,647 chunks for fine-tuning and cloze evaluation. |
| **Canonical split registry** | Deterministic SHA-1 partition of manuscript IDs | v1.0.0, cut points 73 / 88 | `data/splits/dss_scroll_splits_v1.json` | 531 train / 108 val / 93 test, 0 straddling. |
| **Embible comparison data** | Embible project's released biblical validation/test JSONL | Pinned backend commit `7c9e769274a273d0b357b066d932f1c6833ca5f8` | `analysis/reports/embible_bible_transfer.json` | Domain-transfer diagnostic. Biblical text was used for calibration and evaluation, **never training**. |

> [!WARNING]
> **Licensing.** `ETCBC/dss` is **CC BY-NC 4.0**. Attribution must name Abegg, Bowley and Cook, and any
> redistribution of derived text — including examples printed in the paper or shipped in a demo — has to
> respect the non-commercial term. This needs checking against the target venue's requirements.

---

## 🔬 6. Safeguards & Information Regimes

```
+---------------------------------------------------------------------------------------------------+
|                                   THREE INFORMATION REGIMES                                       |
+---------------------------------------------------------------------------------------------------+
|  [U0]    Context only. No length, no letter traces.                                               |
|          Real lacunae, n=74 : 9.5% Top-10                                        [pilot]          |
|                                                                                                   |
|  [O-len] Gold character length supplied as a proxy. Synthetic cloze control.                       |
|          scatter-30, n=729  : 21.12% Hit@10 (TavBERT base)                       [measured]        |
|                                                                                                   |
|  [P0]    Estimated character budget + surviving partial ink.                                       |
|          Real lacunae, n=74 : 63.5% Top-10  [51.4%-74.3%]                        [pilot]          |
+---------------------------------------------------------------------------------------------------+
```

U0 and P0 are measured on the *same* 74 targets with the *same* model, so their difference is meaningful. The
O-len row is a **different task on different data** and does not belong in that progression.

### Open issue: two split registries

Two incompatible split definitions coexist:

| Registry | Scrolls | Partition |
|---|---|---|
| `data/splits/dss_scroll_splits_v1.json` (canonical) | 732 | 531 train / 108 val / 93 test |
| `preserved_nonbib_manifest.json` → `scroll_splits` | 736 | 413 train / 103 dev / 220 heldout |

The QD benchmark ran against the **manifest** registry, so **40 of its 74 targets fall in the canonical
registry's train split**. Table 2 reports the canonical-clean subsets alongside the headline. The mismatched
targets score *lower* (60.0% Top-10) than the clean ones, so the headline is not inflated — but the registries
must be consolidated and the benchmark re-run on one of them before submission.

Separately, **only 374 of the 732 registry scrolls yield any text chunk**: a chunk requires at least 20
preserved words, and most scrolls are too fragmentary. A further 3 chunk-bearing scrolls are absent from the
canonical registry entirely.

---

## ⚙️ 7. Methodology

### 7.1 Manuscript Partitioning ([`data/splits/dss_scroll_splits_v1.json`](../data/splits/dss_scroll_splits_v1.json))

$$\text{Partition}(\text{Scroll\_ID}) = \text{SHA1}(\text{Scroll\_ID}) \bmod 100 \implies \begin{cases} \text{Train} & [0, 73) \quad (531 \text{ scrolls},\ 72.5\%) \\ \text{Val} & [73, 88) \quad (108 \text{ scrolls},\ 14.8\%) \\ \text{Test} & [88, 100) \quad (93 \text{ scrolls},\ 12.7\%) \end{cases}$$

A scroll's split is a pure function of its identifier, so no manuscript can straddle two splits.
`utils/splits.py:validate_split_disjointness` asserts this.

### 7.2 Unified Candidate Generator ([`eval/candidate_generator.py`](../eval/candidate_generator.py))

$$\text{generate\_candidates}(\text{context}_{\text{left}}, \text{context}_{\text{right}}, L, P, K) \to [C_1, \dots, C_K]$$

### 7.3 Partial-Letters Conditioning
For character-level MLMs, candidate characters inconsistent with the surviving ink pattern receive zero
probability at each position:

$$P(c_i \mid c_{<i}) = \begin{cases} P_{\text{model}}(c_i \mid c_{<i}) & \text{if } P[i] = \text{wildcard or } c_i = P[i] \\ 0 & \text{otherwise} \end{cases}$$

### 7.4 Length-Ensemble Decoding
A fixed-length MLM must be told how many `[MASK]` tokens to emit and cannot vary output length during
inference. For a character MLM, `LengthEnsembleCharMLMGenerator` runs beam search once per candidate length
$L \in [3, 15]$ and re-ranks the union with a length penalty:

$$\text{Score}(C_L) = \frac{\sum_{i=1}^{L} \log P(c_i \mid c_{<i}, \text{context})}{L^{\alpha}}, \quad \alpha = 0.5$$

> [!WARNING]
> **[not yet measured]** This generator has never been run against a benchmark. No accuracy is claimed for it,
> and unknown-length multi-word restoration remains an open problem.

### 7.5 Scoring Protocol
- **All-words headline metric ($unaligned = miss$).** A prediction that cannot align to the character-bounded
  gap scores 0.0 rather than being dropped from the denominator.
- **Aligned-only metric.** Reported alongside, for transparency about where the difference comes from.

### 7.6 Fine-Tuning ([`training/unified_trainer.py`](../training/unified_trainer.py))
- Task: 1–3 word contiguous span masking.
- Learning rate $1\times10^{-5}$, linear warmup ratio 0.1, weight decay 0.01.
- `evaluation_strategy="epoch"`, `load_best_model_at_end=True`, tracking validation loss on the 108 validation
  scrolls.

> [!NOTE]
> **[not yet measured]** No early-stopped run has been scored on the cloze benchmark. The measured comparison
> remains 20.58% fine-tuned against 21.12% base — that is, fine-tuning has not yet been shown to help.

### 7.7 Uncertainty
- **95% confidence intervals:** sentence-level percentile cluster bootstrap, $B = 1000$ resamples, resampling
  *sentences* rather than words because words inside a sentence share a context and a masking draw.
- **`mcnemar_test()`** is implemented in `metrics_runner.py` for paired model comparison:

$$z = \frac{(|b - c| - 1)^2}{b + c}, \quad p = 2(1 - \Phi(\sqrt{z}))$$

> [!WARNING]
> `mcnemar_test()` has never been applied to a reported comparison, so **no model lead in this document is
> established as significant**, and the overlapping intervals in Table 1 suggest several would not be. Running
> it on the paired predictions is a prerequisite for any comparative claim.

### 7.8 `EpigraphicStrokeFilter`

On torn fragments, surviving strokes along a hole edge can be genuinely ambiguous: a single vertical downstroke
may belong to `ר`, `ד`, `ו`, `ן`, `נ` or `י`, and a damaged roof stroke to `ה`, `ח` or `ת`. Early editors
working under optical magnification recorded tentative readings that later collation with better imaging
sometimes revised. This is why Qumran-Digital records multiple attributed readings per site — which is exactly
what Table 2 scores against.

The current implementation assigns similarity as:

$$S(c_1, c_2) = \begin{cases} 1.0 & \text{if } c_1 = c_2 \text{ or either is the wildcard} \\ 0.85 & \text{if } c_1, c_2 \text{ share a stroke group} \\ 0.0 & \text{otherwise} \end{cases}$$

with groups `{ר ד ו ן נ י}`, `{ה ח ת}`, `{מ ס}`, `{ב כ}` and an acceptance threshold of `0.85`.

> [!WARNING]
> **[not yet measured] and needs rework.** Because the threshold equals the in-group score, *any* substitution
> within a group passes unconditionally — this is a relaxed matcher, not a probability model, and it will admit
> readings the surviving ink excludes. The group `{ר ד ו ן נ י}` collapses six letters, and `{מ ס}` and
> `{ב כ}` are hard to defend palaeographically for the relevant hands. A publishable version needs per-hand
> confusion probabilities estimated from a labelled palaeographic sample, and any resulting gain must be
> reported as relaxed-match agreement rather than accuracy.

---

## 📊 8. Benchmark Tables

### 📌 SECTION 1: MAIN BODY

#### Table 1: Synthetic Cloze — `scatter-30` **[measured]**
*100 held-out sentences, $n=729$ masked words, regime O-len. Intervals are sentence-level percentile cluster
bootstrap, $B=1000$.*
Artifact: [`external_comparison/results/summary_with_subsets.json`](../external_comparison/results/summary_with_subsets.json) (2026-08-01).

| Model | Hit@10 (all-words) + 95% CI | Hit@1 | Aligned-only Hit@10 | Char sim | MRR | Unaligned |
|---|---|---|---|---|---|---|
| **TavBERT Base** | **21.12%** [17.01%, 25.39%] | 7.27% | 21.12% | 0.176 | 0.117 | 0 / 729 (0.0%) |
| TavBERT Fine-Tuned | 20.58% [17.03%, 24.49%] | 7.96% | 20.58% | 0.209 | 0.118 | 0 / 729 (0.0%) |
| MsBERT Fine-Tuned | 13.31% [16.76%, 26.59%] ⚠️ | 9.78% | 21.56% | 0.242 | 0.134 | 279 / 729 (38.3%) |
| MsBERT Base | 10.29% [11.66%, 21.56%] ⚠️ | 6.47% | 16.74% | 0.221 | 0.096 | 281 / 729 (38.5%) |
| ByT5 Unified FT | 5.35% [3.73%, 7.25%] | 0.82% | 5.35% | 0.127 | 0.018 | 0 / 729 (0.0%) |

⚠️ For the two MsBERT rows the bootstrap interval was computed on the **aligned-only** subset ($n=450$ / 448),
so it does not bracket the all-words headline. **This must be recomputed on the full denominator before
submission.**

Two further points a reviewer will press on: fine-tuning does not help (base 21.12% vs. FT 20.58%), and no
model here is separated from its neighbour at 95%. DictaBERT-char was not run on this benchmark; it appears
only in Table A1.

#### Table 2: Real Lacuna Literature Agreement — Qumran-Digital, $n=74$ **[pilot]**
*Model: `ft_msbert_span_preserved_nonbib`, trained on preserved letters only, with post-MLM physical
filtering. **Not** a character model — the character models have not been run on this benchmark.*
Artifacts: [`QD_RESEARCHER_BENCHMARK.md`](../analysis/reports/QD_RESEARCHER_BENCHMARK.md),
[`qd_researcher_comparison.json`](../analysis/reports/qd_researcher_comparison.json).

| System / condition | Regime | $n$ | Top-1 | Top-10 | Top-20 |
|---|---|---|---|---|---|
| **MsBERT FT + physical filtering** | P0, ±1 | 74 | **40.5%** | **63.5%** [51.4%, 74.3%] | **67.6%** |
| Same model, same targets | U0 | 74 | — | 9.5% | — |
| + train-only RAG | P0, ±1 | 74 | 40.5% | 63.5% | 67.6% |
| QD initial-reading control | — | 74 | 20.3% | 43.2% | 44.6% |

**Robustness to the split-registry conflict** (see §6):

| Restricted to | $n$ | Top-1 | Top-10 | Top-20 |
|---|---|---|---|---|
| Canonical **test** scrolls | 12 | 50.0% | 83.3% | 83.3% |
| Canonical **val + test** | 34 | 32.4% | 67.6% | 76.5% |
| Canonical **train** (mismatched) | 40 | 47.5% | 60.0% | 60.0% |

Caption must state three limits: QD selects **disputed** sites, the references are modern proposals rather than
physical truth, and at $n=74$ — 12 on the strictest subset — the interval is wide. The RAG row shows a gain of
exactly zero.

**Length-tolerance sensitivity** (from the same artifact): ±0 → 52.5% / 69.5% / 71.2% on 59 eligible targets;
±1 → 40.5% / 63.5% / 67.6% on 74; ±2 → 40.0% / 62.7% / 66.7% on 75. The conclusion is stable across
tolerances; the eligible count changes because a proposal outside a tolerance is not physically compatible at
that setting.

#### Table 3: Multi-Word Lacunae, Known Slot Count **[measured]**
*The decoder knows the number of word slots but no gold character lengths. References are anonymous
Text-Fabric editorial reconstructions used only as evaluation labels, excluded from input, training and
retrieval.*
Artifact: [`PRESERVED_RAG_LACUNA_LENGTHS.md`](../analysis/reports/PRESERVED_RAG_LACUNA_LENGTHS.md).

| Condition | Unit | $n$ | Top-1 | Top-10 |
|---|---|---|---|---|
| Single-word lacuna, baseline | slot = sequence | 25 | 12.0% | 60.0% |
| Multi-word, baseline | slot | 440 slots | 14.5% | 41.4% |
| Multi-word, baseline | sequence (whole gap) | 100 spans | 4.0% | 7.0% |
| Multi-word, + train-only RAG | sequence (whole gap) | 100 spans | 3.0% | 9.0% |

Every row supplies the word-slot count. The genuinely unknown-length case is not represented — see §7.4.

#### Table 4: Physical Lacuna Corpus Statistics **[measured — counts only]**
*Recomputed from the parsed dataset via `eval/large_scale_lacuna_eval.py`. **No model has been scored at this
scale**; any accuracy figure for 3,695 lacunae would be unsupported.*

| Quantity | Whole non-biblical corpus | Canonical test split |
|---|---|---|
| Scrolls | 732 | 93 |
| Physical lacunae | 27,814 | 3,695 |
| Damaged word positions | 165,239 | 22,123 |
| Damaged words retaining ink | 49,130 (29.7%) | 6,512 (29.4%) |
| Lacunae with ≥1 traced word | 23,632 (85.0%) | — |
| Preserved (`rec=0`) words | 103,355 | — |

The trace rate is the ceiling on where P0 conditioning can apply at all: roughly seven in ten damaged words
show no ink.

### 📁 SECTION 2: APPENDIX

#### Table A1: Model Selection & Pretraining Scale **[measured]**
*30 held-out sentences, 250 masked words, same scoring as Table 1. Parameter counts are derived from the fp32
checkpoint sizes of the models actually evaluated.*
Artifact: [`external_comparison/results/quick30_summary.json`](../external_comparison/results/quick30_summary.json).

| Model | Params | Fine-tuned? | Hit@10 | 95% CI | Unaligned |
|---|---|---|---|---|---|
| **TavBERT Base** | ~87M | No | **24.40%** | [18.09%, 31.94%] | 0 |
| DictaBERT-char FT | ~88M | Yes | 22.80% | [17.03%, 29.03%] | 0 |
| TavBERT Fine-Tuned | ~87M | Yes | 22.40% | [15.99%, 30.33%] | 0 |
| MsBERT Fine-Tuned | ~184M | Yes | 21.12% (aligned-only) | [13.00%, 29.17%] | 89 |
| MsBERT Base | ~184M | No | 18.87% (aligned-only) | [9.56%, 27.21%] | 91 |
| DictaBERT-char Base | ~88M | No | 17.60% | [12.16%, 23.29%] | 0 |
| DictaBERT-Large-char | ~307M | No | 17.60% | [11.60%, 23.14%] | 0 |

At $n=250$ every interval overlaps. The defensible conclusion is the negative one: scaling ~88M → ~307M buys
nothing (17.60% both), and no fine-tuning benefit survives. Model selection should not be decided on 250 words.

#### Table A2: Quote-Aware Pesher Source Retrieval **[measured]**
*Recovery is macro-averaged by Pesher manuscript over 35 known-source passages from eight mapped Pesharim,
which is why the percentages are not multiples of 1/35. The ablation is a **leakage control**, not a rival
method: every token in an exact external match is masked before ranking, then residual recovery is measured
under nested leave-one-manuscript-out. Significance is a **permutation test** over scroll clusters.*
Artifacts: [`CROSS_CORPUS_QUOTE_ABLATION.md`](../analysis/reports/CROSS_CORPUS_QUOTE_ABLATION.md),
[`CROSS_CORPUS_QUOTE_ABLATION_BIGRAM.md`](../analysis/reports/CROSS_CORPUS_QUOTE_ABLATION_BIGRAM.md).

| Condition | Top-1 | Top-3 | Significance |
|---|---|---|---|
| Literal channel, quotations intact | 86.6% | 99.1% | — |
| Residual, exact 3-word matches masked (27.0% of words) | 52.4% | 83.6% | permutation $p = 0.0008$ |
| Residual, exact 2-word matches masked (58.4% of words) | 45.3% | 82.2% | permutation $p = 0.0012$ |

This validates *source-book recovery* on passages whose source is already known. It is not a restoration
result: no measured restoration gain has been attributed to retrieval anywhere in this project. Residual
affinity after literal masking does not establish paraphrase, borrowing, authorship, or direction of influence.

#### Table A3: Canonical Dataset Split **[measured]**
*Chunk counts recomputed by mapping every chunk in `preserved_nonbib_chunks.jsonl` through the canonical
registry.*

| Split | Scrolls | Share | Text chunks | Straddling |
|---|---|---|---|---|
| Train | 531 | 72.5% | 1,192 | 0 |
| Validation | 108 | 14.8% | 212 | 0 |
| Test | 93 | 12.7% | 231 | 0 |
| **Total** | **732** | **100.0%** | **1,635** | **0** |

A further 12 chunks come from scrolls absent from the canonical registry, giving 1,647 chunks built in total.
Only **374 of the 732 registry scrolls (51.1%)** yield any chunk, because a chunk requires ≥20 preserved words.

---

## 🗣️ 9. Advisor Talking Points

### 9.1 What Would Survive Peer Review?

**Defensible as pilots, with limitations stated:**
- The U0 → P0 contrast on 74 real lacunae: 9.5% → 63.5% Top-10, bootstrap 51.4%–74.3%.
- The cloze table at $n=729$, presented with its overlapping intervals and the finding that fine-tuning does
  not help.
- The WordPiece alignment failure: 279/729 (38.3%).
- Pesher source recovery (86.6% / 99.1%) with its permutation-tested leakage control.
- The corpus statistics and the split construction.

**Not yet publishable:**
- Anything about the four roadmap components.
- Unknown-length multi-word restoration.
- Any retrieval benefit to restoration.
- Per-category error shares.
- Any model accuracy at the 3,695-lacuna scale.
- Any claim that one model significantly beats another.

**Blocking issues before submission:**
1. Consolidate the two split registries and re-run Table 2 on one of them.
2. Recompute the MsBERT bootstrap intervals on the all-words denominator.
3. Run the character models on the QD benchmark so the flagship result is not MsBERT-only.
4. Run `mcnemar_test()` on the paired predictions so comparisons carry a test.
5. Grow $n$ beyond 74 (12 on the strictest subset).
6. Re-derive the error taxonomy on the current model.

### 9.2 Four Questions to Expect

**"Why mask 30% of words instead of 15%?"** — In the parsed corpus, damaged positions outnumber preserved ones
(165,239 gap words against 103,355 preserved), so 30% simultaneous masking is if anything conservative. It also
stresses the multi-gap regime, which §3.3 shows is where accuracy actually degrades.

**"Is ~21% Hit@10 good?"** — It is weak, and it is a diagnostic rather than a headline. See §4.2.

**"How do you know the model didn't read the editors' guesses?"** — `rec=1` signs are never emitted into model
input; the manifest records `modern_reconstruction_text_emitted: false`. See §4.2.

**"What is the human comparison?"** — Qumran-Digital's own recorded *initial reading* per target, scored
against the same attributed restorations: 20.3% Top-1, 43.2% Top-10. It is a database field, not a controlled
human study, it is available for only some targets, and because QD catalogues disputed sites a low score is
partly definitional. It bounds the difficulty of these sites; it does not license a claim that the model
outperforms scholars.

---

## 🛠️ 10. Codebase & Commands

```text
dss-restoration/
├── data/
│   ├── derived/nonbib_lacunae.jsonl      <-- 27,814 physical lacunae
│   ├── derived/preserved_nonbib_chunks.jsonl  <-- 1,647 preserved-text chunks
│   └── splits/dss_scroll_splits_v1.json  <-- canonical split registry (732 scrolls)
├── utils/
│   ├── splits.py                         <-- split loader & disjointness validator
│   └── tokenizer_compat.py               <-- WordPiece/char tokenizer helper
├── eval/
│   ├── candidate_generator.py            <-- CandidateGenerator, PartialLetterFilter,
│   │                                         LengthEnsemble, EpigraphicStrokeFilter
│   ├── masking.py                        <-- masking engine (scatter-30, lacuna-real)
│   ├── metrics_runner.py                 <-- scoring, hit@k, MRR, cluster bootstrap, mcnemar_test
│   ├── full_test_runner.py               <-- full test-split evaluator
│   ├── large_scale_lacuna_eval.py        <-- physical lacuna corpus statistics (counts only)
│   └── score_qd_researcher_benchmark.py  <-- QD literature-agreement benchmark
├── training/
│   ├── unified_trainer.py                <-- fine-tuning CLI with early stopping
│   └── run_multiseed_experiment.py       <-- multi-seed runner
├── analysis/reports/                     <-- generated artifacts; the source of every number here
├── tests/                                <-- 79 tests, incl. test_public_claims.py claim guards
└── pytest.ini
```

```bash
# Full test suite, including the public-claim guards
uv run pytest

# Multi-seed fine-tuning with validation early stopping
PYTHONPATH=. uv run python training/run_multiseed_experiment.py --model tau/tavbert-he --epochs 3

# Score an existing prediction run against the cloze benchmark
PYTHONPATH=. uv run python eval/full_test_runner.py --run-dir external_comparison/results/tavbert-base

# Score the QD literature-agreement benchmark (offline against the cached snapshot)
PYTHONPATH=. uv run python eval/score_qd_researcher_benchmark.py

# Physical lacuna corpus statistics
PYTHONPATH=. uv run python eval/large_scale_lacuna_eval.py
```

`tests/test_public_claims.py` ties this document and `PAPER_PRESENTATION.html` to the generated artifacts: it
fails if an unsupported figure reappears, if a significance claim is made without a test behind it, if the QD
figures drift from the scorer's own report, or if the split-registry conflict stops being disclosed.

---

## 📚 11. Glossary

| Term | Definition | Status |
|---|---|---|
| **Reconstruction redaction** | Dropping every `rec=1` sign — a modern editorial reconstruction printed in square brackets — before the text reaches the model, e.g. `כ[נגדו` → `כ⬚⬚⬚⬚`. | Enforced at corpus-build time. **[verified]** |
| **Dual-track evaluation** | Separating synthetic cloze over preserved text (Track A) from real physical lacunae (Track B). | Both measured. Track A must never be quoted as real-lacuna accuracy. |
| **All-words metric** ($unaligned = miss$) | A prediction that cannot align to a character-bounded gap scores 0.0 rather than leaving the denominator. | MsBERT: 279/729 (38.3%) unaligned; 21.56% aligned-only vs. 13.31% all-words. **[measured]** |
| **Information regimes** (U0, O-len, P0) | U0 = context only; O-len = gold character length proxy; P0 = surviving ink + approximate length (±1). | Compared within a benchmark, never across. **[measured]** |
| **Physical ink conditioning (P0)** | Candidate filtering that keeps only strings agreeing with the surviving ink and matching the estimated length within ±1. | The project's one large measured effect: +54.0 pp Top-10. Applies to the 29.7% of damaged words retaining ink. **[pilot]** |
| **Literature agreement** | Scoring against modern published proposals rather than the lost original wording. | What Table 2 measures. QD catalogues *disputed* sites, so this is neither ground truth nor a random sample. |
| **Manuscript-disjoint partitioning** | `SHA1(scroll_id) mod 100`, cut points 73 / 88, so all fragments of a manuscript share a split. | 531 / 108 / 93, 0 straddling. A second registry also exists — see §6. **[verified]** |
| **Character-level MLM** | Encoder over a single-character vocabulary ($\|V\| \approx 30\text{--}50$), e.g. TavBERT, DictaBERT-char. | Enables arbitrary gap lengths, 0/729 unaligned. Cloze advantage over WordPiece is modest and not significant. |
| **Lemma-normalised agreement** | Relaxed scoring folding final forms, plene spellings and some affixes before comparison. | A metric relaxation, reportable only beside exact match. **[not yet measured]** |
| **Length-ensemble decoding** | Beam search over candidate lengths $L \in [3,15]$, re-ranked by $(\sum \log P)/L^{0.5}$. | Implemented, unit-tested, never benchmarked. **[not yet measured]** |
| **Quote-aware source retrieval** | Detects Pesher citation formulas and identifies the quoted biblical book by character n-gram similarity. | 86.6% / 99.1% book recovery. No measured restoration gain. **[measured]** |
| **Stroke-confusion matching** | Treating palaeographically similar letters as interchangeable when matching against surviving ink. | Four hand-written groups, one constant (0.85), threshold 0.85 — an unconditional in-group pass. Not a probability model. **[not yet measured]** |

---

## 🎨 12. LaTeX Figure Snippet

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[node distance=1.4cm, auto]
    \node [draw, rectangle, fill=blue!10, rounded corners] (fragment)
        {Digital edition: \texttt{... \textcjrab{לו עזר} \textcjrab{כ[נגדו}}};
    \node [draw, rectangle, fill=green!10, below of=fragment] (redact)
        {Redaction (\texttt{rec=1} dropped): \texttt{\textcjrab{כ⬚⬚⬚⬚}}};
    \node [draw, rectangle, fill=orange!10, below of=redact] (filter)
        {PartialLetterFilter: pattern \texttt{\textcjrab{כ}????}, $L = 5 \pm 1$};
    \node [draw, rectangle, fill=purple!10, below of=filter] (output)
        {Top-1: \textbf{\textcjrab{כנגדו}} \quad (4Q303 frg.~1, l.~10; test split)};

    \draw[->, thick] (fragment) -- (redact);
    \draw[->, thick] (redact) -- (filter);
    \draw[->, thick] (filter) -- (output);
\end{tikzpicture}
\caption{Partial-letter conditioning pipeline. Surviving ink and an approximate gap length constrain the
candidate set; the encoder ranks within it. Across 74 real lacunae this raises Top-10 literature agreement
from 9.5\% to 63.5\% (95\% CI 51.4--74.3\%).}
\label{fig:pipeline_overview}
\end{figure}
```
