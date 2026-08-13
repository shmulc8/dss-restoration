# DSS Text Restoration — Unification Plan: Decision Points & Evidence

> **Historical decision record (not a current results source).** Use
> `docs/RESULTS.md` and `experiments/results/paper/paper_results_snapshot.json`
> for current numbers and promotion status.

**Purpose.** Shmulik and Itay have two independent codebases for the same research goal. This document is the single reference for merging them into one repository with one benchmark protocol: the background (§0), the twelve decisions to agree on (§1–§12), the empirical evidence gathered so far (§R), and the numbers each side stands behind (Appendix). All statements about the codebases were verified against the code itself on 2026-07-31; the evaluation paths were audited for gold-information leaks on 2026-08-02 (§R5).

---

## 0. Background: the task and the two pipelines

**The problem.** The Dead Sea Scrolls are ~2,000-year-old Hebrew manuscripts with large physically lost stretches — holes, abrasions, torn edges (a missing stretch is a *lacuna*). Scholars propose restorations by hand; both projects train language models to do it from the surviving context. Both face the same trap: modern editions print the editors' reconstructions inside the text, so a model trained naively on an edition learns modern guesses, not ancient language — and its "predictions" become circular. Each pipeline therefore starts by deciding which words are authentic enough to train on.

**The shared source.** Both read the same machine-readable corpus: the ETCBC transcription of the scrolls (Text-Fabric format), which records sign-by-sign whether each character physically survives or was supplied by an editor.

**Itay's pipeline (`new_dead_sea_scrolls`):**
1. *Curation* — every word classified into one of 7 preservation categories; partially-preserved words kept if ≥ half their characters are authentic; reconstructed words dropped.
2. *Dataset* — sentences of ≥ 7 words (non-biblical scrolls), one xlsx with train/val/test labels; split is scroll-disjoint via a deterministic hash of the scroll name.
3. *Models* — BERT-style masked language models (MLMs) that fill blanks in place: MsBERT (subword pieces) and TavBERT (character-level).
4. *Evaluation* — synthetic damage: hide 30% of each test sentence's words, beam-search fill each blank, score hit@k / character-similarity / rank metrics with bootstrap CIs. Every run writes a durable `predictions.jsonl`; an HTML viewer compares runs and flags incomparable protocols and train/test contamination.

**Shmulik's pipeline (`dss-restoration`):**
1. *Curation* — stricter: a word is discarded if *any* of its characters is editor-supplied or damage-marked; a validator hard-fails if an editorial character survives. Non-biblical only.
2. *Dataset* — contiguous preserved-text chunks; seeded manuscript-level (scroll-disjoint) split.
3. *Models* — mainly ByT5, a sequence-to-sequence generator over raw bytes (no tokenizer) that writes the missing span as free text; MsBERT/TavBERT/BEREL fine-tuned variants as baselines.
4. *Evaluation* — two tracks: (a) synthetic contiguous 1–3-word lacunae with 8 words of context per side; (b) real lacunae: agreement with published scholarly restorations (Qumran-Digital targets), including lemma-level matching via a Hebrew lemmatizer.

**The four substantive differences** (everything else is engineering that merges cleanly):

| Difference | Itay | Shmulik | Decided in |
|---|---|---|---|
| Clean training text | keep half-authentic words | strict whole-word stripping | §1 |
| Shape of the test | scattered sentence masks | contiguous lacunae + real gaps | §7, §10 |
| Architecture | MLM fill-in | seq2seq generation | §4 |
| Length info given to model | gold token count (implicit) | gold length ±2 filter (explicit) | §5 |

---

## The 12 decision points

### 1. Data curation — which words enter the corpus
- **Itay:** 7-marker scheme; `PPP` (partially preserved) words kept at authentic-character ratio ≥ 0.5. Dataset: `ppp_nonbib`, 2,424 sentences.
- **Shmulik:** whole-word stripping if any sign has `rec == 1` or `rem == 1` or `#` (OR over three sign-level conditions); `biblical == 0`; hard validator.
- **Proposal:** test set strict-preserved, full stop. Training uses the `ppp` (≥ 0.5) corpus — more data, cannot leak into a strict test set. One strict-trained control per model family; revisit only if it flips conclusions.

### 2. Corpus unit — sentences vs. chunks
- **Itay:** ETCBC sentences ≥ 7 words, with lemma/POS metadata. **Shmulik:** preserved chunks, no sentence-segmentation dependency.
- **Proposal:** sentence xlsx is the benchmark unit (the eval infrastructure runs on it); chunks stay as a training-data option. Document the ETCBC-segmentation dependency as a limitation.

### 3. Train/val/test split
- **Itay:** `sha1(scroll) % 100`, cut-points 73/88 — realized ~74/12/14 on `ppp_nonbib` (the cut-points approximate 70/15/15, they don't guarantee it).
- **Shmulik:** seeded RNG, book-level — realized ≈ 61/14/25 by chunk; two split implementations exist and need consolidating regardless.
- **Proposal:** generate once with the `sha1` bucketing, **commit the realized scroll→split assignment as a frozen JSON file**, and make every code path load that file. One test asserts scroll-disjointness and file↔corpus consistency.

### 4. Model architecture and roster for the unified system
- **Evidence-based center:** the final restoration engine should be a character-level MLM. Physical gap length, surviving letters and their positions, and spaces inside multiword gaps are character-shaped evidence; a char model covers every target and can condition on that evidence directly (§R2b). The final checkpoint does not exist yet: train the same realistic DSS restoration objective from two initializations, `dicta-il/dictabert-char` and `tau/tavbert-he`, then select on frozen dev criteria and paired full-test results.
- **Current leaders — no single overall winner yet:**
  - **TavBERT base** wins the headline synthetic comparison (21.1% hit@10 on `scatter-30`) and QD Top-1 (45.9%).
  - **Fine-tuned dictabert-char** wins QD Top-10/20 (66.2%/73.0%) and is close to TavBERT on the 30-sentence synthetic pilot (22.8% vs 24.4% hit@10).
  - **Fine-tuned MsBERT** wins only when unaligned targets are excluded (21.6% synthetic hit@10); under the headline all-target scoring it falls to 13.3%, so it remains a baseline rather than the system default.
- **Primary candidates:** `dictabert-char` is the likely distributable default (88M parameters, safetensors, explicit CC BY 4.0); TavBERT is the mandatory competing initialization because it is strongest zero-shot and currently has the best QD Top-1. TavBERT's published model card does not state a license, so clarify redistribution terms before making it the released default. Current pilots do not statistically separate the two (§R2b, §R6).
- **Baseline, not product core:** keep `dicta-il/MsBERT` as the manuscript-specific WordPiece baseline and for the vocabulary-filter comparison. It is not the default engine because its tokenization leaves targets unaligned and its fixed single-WordPiece candidate set cannot generate the partial-letter restorations directly (§8, §R1, §R2b).
- **Conditional experiments:** run `dicta-il/dictabert-large-char` only as a one-seed scale ablation after the base pipeline is stable; it showed no zero-shot gain in §R6. Preserve ByT5-small as the negative seq2seq result in §R1, but remove it from the headline roster and main GPU rerun. Revisit a larger seq2seq model only if a separately defined unknown-length/multiword track demonstrates that the char engine cannot cover the task. BEREL and NeoDictaBERT leave the active roster.
- **Unified software boundary:** retain the common `predictions.jsonl`/manifest/metric contract, not an MLM-specific runner as the permanent core. A model-neutral `CandidateGenerator` interface feeds one shared constraint, normalization, scoring, and artifact layer; implementations are `CharMLMGenerator` (DictaBERT-char and TavBERT), `WordMLMGenerator` (MsBERT), and an optional future `Seq2SeqGenerator`. Model IDs, pinned revisions, tokenizer family, and license live in a registry. Tokenizer compatibility is validated centrally rather than by manually patching downloaded model files.

### 5. Length information — the biggest comparability risk
What the code does today: **Itay's protocol leaks gold length** — the number of `[MASK]` tokens equals the gold word's token count, which for char-level TavBERT is the *exact character count of the answer* and for MsBERT its WordPiece count: two different oracles. **Shmulik's length filter is also gold-derived** (candidates outside gold-length ± 2 chars dropped). Neither is "unconstrained," and no current protocol uses physically measured gaps.
- **Proposal:** define three regimes, label every reported number, never mix silently:
  - **U0** — no length information;
  - **O-len** — gold-derived length (both current protocols);
  - **P0** — length budget from the physically recorded gap (available today — §6).
  **Headline = length-informed (P0):** a real lacuna always comes with a measurable physical extent, so the researcher-relevant number includes it — §R2's 63.5% is this regime. Until the benchmark carries real gap measurements for every item, O-len serves as the labeled interim proxy for P0 (for char-level models an accurate measurement converges to it — §R2's exact-length row, 69.5%). U0 is a diagnostic ablation showing how much of the performance is the constraint. §R1 quantifies the regime gaps.

### 6. Ground the benchmark in real lacuna statistics
From 12,971 real damaged-word runs on held-out scrolls (Shmulik's derived data, computed 2026-07-31):

| Gap length (words) | share | · | Missing chars per damaged word | share |
|---|---|---|---|---|
| 1 | 77.6% | | 1 | 41.0% |
| 2 | 17.4% | | 2 | 36.6% |
| 3 | 3.7% | | 3 | 10.4% |
| 4 | 0.9% | | 4 | 6.7% |
| 5+ | 0.4% | | 5+ | 5.3% |

Word-level: p90 = 2, p95 = 3, p99 = 4. Char-level (16,769 recorded patterns): median 2, p95 = 5, p99 = 7. **82.5% of damaged words retain at least one legible letter** (patterns like `סר⬚⬚ך` with known gap positions).
- **Proposal:** (a) benchmark span lengths sampled from this distribution instead of uniform-K or a fixed mask ratio; (b) **real P0** — character budgets from the recorded `⬚` counts, never from the gold answer; (c) a **partial-letters regime** — the model also sees surviving letters and their positions. No prior benchmark evaluates (c); it is both the most faithful setting and a novel contribution. §R2 shows constraints of this kind are worth +54 points, and §R2b shows that character-level conditioning is competitive with vocabulary filtering while removing its candidate ceiling.

### 7. Eval masking policy
- **Itay:** 30% of sentence words masked, span-concentration 0.5, per-sentence deterministic seeding. **Shmulik:** contiguous 1–3-word lacunae, 8-word context.
- **Proposal:** two named tracks in one runner — `scatter-30` (Itay's; stresses global context) and `lacuna-real` (contiguous spans from the §6 distribution; subsumes Shmulik's K ∈ {1,2,3}). Same artifact format; keep per-sentence seeding (it makes paired statistics valid).

### 8. Scoring fairness — the unaligned-word exclusion
Itay's scorer drops words whose predictions can't be word-aligned instead of counting them as misses; MsBERT loses ~38% of words this way, TavBERT 0%. §R1 shows this single rule decides the leaderboard winner.
- **Proposal:** unaligned words count as misses in the headline metric; the exclusion-based number is reported secondarily as `hit@k_aligned`.

### 9. Metrics, statistics, and artifacts
- **Adopt from Itay:** `predictions.jsonl` durable artifact; generation/scoring separation with rescore; hit@{1,3,5,10}, char_sim, MRR; cluster bootstrap B=1000; McNemar pairing; `protocol_id`/`decode_id` comparability gating; train/eval contamination check.
- **Adopt from Shmulik:** DictaBERT lemma-level matching; real-lacuna literature-agreement track; leakage/protocol validators; the forbidden-claims test (greps docs for retired numbers so they can't resurface).
- **Proposal:** union of both; freeze the artifact schema version at unification.

### 10. Real-lacuna evaluation (literature agreement)
Shmulik-only today; Itay's evaluation is fully synthetic. The Qumran-Digital snapshot has 1,811 raw rows → **74 eligible targets / 99 compatible target-reading pairs** currently scored (sources incl. Qimron 2013/2020, DJD XXIX).
- **Proposal:** include as the second benchmark track — the only evaluation with genuinely lost text; the paper's humanities-relevance argument rests on it. Report with the small-n caveat; grow the target set jointly; combine with §6(c), since real lacunae carry their own char budgets and surviving letters.

### 11. Training protocol for fine-tuned models
- **Checkpoint selection:** probe-selected "best", always; no manually picked epochs.
- **Seeds:** 3 per headline model with mean ± sd, or 1 seed with the claim scoped accordingly — decided per model before running.
- **Tune scope:** full fine-tune for the paper; LoRA as ablation.
- **Recipe:** each model's masking-mixture recipe is part of its identity but must be recorded in `training/tuning_config.json` inside every published checkpoint (Itay's convention — keep).

### 12. Repository mechanics
- One repo with neutral core modules: Shmulik's corpus building, validators, physical constraints, lemma/literature benchmarks, and demo; Itay's durable artifact schema, scoring, bootstrap statistics, comparison viewer, and training conventions. Replace model-specific orchestration with the `CandidateGenerator` boundary in §4 rather than treating either existing runner as the permanent backbone.
- Source of truth is `.py` modules; notebooks import from modules (invert the current notebook-is-source contract).
- Results storage: Itay's HF-dataset sync, only `manifest.json` + `metrics.json` in git.
- Tests: union of both suites (masking-regression + scoring; leakage validators + forbidden-claims guard).
- Cleanup at merge: delete or implement placeholder code before it can be cited (Shmulik has a stub benchmark runner that simulates hits); fix dangling doc references (Itay's docs point to two files absent from the shared archive).

---

## §R. Evidence gathered for this document (2026-07-31)

### R1. First unified-protocol comparison — paired sample, all model families

Protocol: Itay's `eval_runner`, `ppp_nonbib` test split, mask_ratio 0.3 / span_concentration 0.5 / seed 42, beam 10×6. TavBERT and MsBERT were fine-tuned fresh with Itay's exact `TuningConfig` (full FT, probe-best checkpoint; see R4). ByT5 is Shmulik's existing preserved-corpus checkpoint entering through the new adapter (§4), **receiving no length information (U0)** while the MLMs implicitly receive gold token counts (O-len) — kept deliberately to quantify §5.

**Paired sample: same 100 test sentences, 729 masked words, every model.** Run on a laptop; the full 338-sentence split is reserved for the GPU pass.

Aligned-only scoring (Itay's current metric):

| Model | Regime | hit@1 | hit@10 | hit@10 CI | char_sim | MRR | unaligned |
|---|---|---|---|---|---|---|---|
| TavBERT base | O-len | 7.3% | 21.1% | [17.0, 25.4] | 0.176 | 0.117 | 0 / 729 |
| TavBERT fine-tuned | O-len | 8.0% | 20.6% | [17.0, 24.5] | 0.209 | 0.118 | 0 / 729 |
| MsBERT base | O-len | 6.5% | 16.7% | [11.7, 21.6] | 0.221 | 0.097 | 281 / 729 |
| MsBERT fine-tuned | O-len | 9.8% | 21.6% | [16.8, 26.6] | 0.242 | 0.134 | 279 / 729 |
| ByT5 preserved (adapter) | U0 | 0.5% | 3.3% | [2.1, 4.7] | 0.123 | 0.013 | 0 / 729 |
| ByT5 preserved (adapter) | O-len (±2) | 0.5% | 4.0% | [2.6, 5.5] | 0.127 | 0.015 | 0 / 729 |
| **ByT5 unified-trained** (Itay's data + task, T4, best epoch 5/7) | U0 | 0.8% | 5.4% | [3.7, 7.3] | 0.127 | 0.018 | 0 / 729 |
| **ByT5 unified-trained** | O-len (±2, 30 beams) | 0.4% | 5.4% | [3.8, 6.9] | 0.137 | 0.016 | 0 / 729 |

Headline scoring per §8 (unaligned = miss), hit@10: TavBERT base **21.1%**, TavBERT FT 20.6%, MsBERT base 10.3%, MsBERT FT 13.3%, ByT5 3.3%.

**Contamination note:** 3 of the sample's 6 scrolls (1Q16, 1Q25, 1QM) are in ByT5's training split. On the clean 30-sentence subset (11Q17, 1Q27, 11Q5; 338 words) the picture is unchanged: TavBERT base 25.2%, TavBERT FT 23.1%, MsBERT base 19.2% aligned-only (12.1% headline), MsBERT FT 24.3% aligned-only (15.4% headline), ByT5 2.4% (O-len 4.4%).

What this sample establishes:

1. **The scoring rule decides the winner (§8 is not cosmetic).** Aligned-only crowns MsBERT-FT (21.6%); counting its 38% unaligned words as misses crowns TavBERT (21.1% vs 13.3%).
2. **The seq2seq question is now answered — ByT5-small is not competitive on this benchmark.** We eliminated both confounds in turn. Distribution shift: a fresh `google/byt5-small` was fine-tuned on Itay's exact train split with the exact eval task format (per-sentence-seeded scatter masking, `restoration:` prompt; 15,294 examples, GPU, early-stopped at epoch 7, best val_loss 1.046 at epoch 5) — hit@10 rose only 3.3% → 5.4%. Length information: giving it the MLMs' gold-length diet (O-len ±2, 30 beams) changed nothing (5.4%). With training matched, data matched, and information matched, ByT5-small sits at ~5% vs TavBERT's ~21%: on scattered single-word-slot restoration, a byte-level generator at this scale is architecturally behind character MLMs, full stop. The remaining open questions for seq2seq are scale (byt5-base+) and the tracks where generation is structurally necessary — multiword unknown-length lacunae (R3 rows D–F, where MLMs are at ~0% too) and the partial-letters regime (§6c). One asymmetry to disclose: the ByT5 adapter predicts one span at a time, so the other masked spans' true words are visible in its context (in training and eval, consistently), while the MLMs see `[MASK]` there. This favors ByT5, and it still loses — so the verdict only strengthens.
3. **Fine-tuning helps MsBERT clearly** (hit@1 6.5% → 9.8%, MRR 0.097 → 0.134) **but barely moves TavBERT** on this recipe (early-stopped at epoch 4; char_sim did rise 0.176 → 0.209) — revisit the character-model schedule before the GPU run.
4. **Nothing is statistically separated at n=100** — CIs overlap; the full-split GPU run with McNemar pairing produces the quotable ranking.

Artifacts (Shmulik's machine, shareable on request): `scratch/external_finetune/merged_results/` — per model: `predictions.jsonl`, `metrics.json`, `word_scores.csv`, plus `summary_with_subsets.json`. All in Itay's run-artifact format, so they load directly into the comparison viewer.

### R2. The same question on real lacunae (Shmulik's benchmark)

Evaluation design is itself an open decision (§5–§8), so R1 is only one lens. This benchmark evaluates at genuinely damaged locations against published scholarly restorations, giving the model the physically recorded evidence (visible letters + length ±1 — the real-P0 + partial-letters regimes of §6):

| Setup (74 QD targets) | Top-1 | Top-10 |
|---|---|---|
| Constrained MLM (visible letters + length ±1) | 40.5% | **63.5%** (scroll-cluster CI 52.4–74.4) |
| Same, exact length (±0; the 59 length-certain targets) | 52.5% | 69.5% |
| Same targets, no physical constraints | — | 9.5% |
| + train-only RAG (α fit on dev) | 40.5% | 63.5% (no change) |
| Control: QD "initial reading" (the human workflow's starting point) | 20.3% | 43.2% |

**How it works** (full spec: `experiments/run_qd_benchmark.py` + `comparison/reports/QD_RESEARCHER_BENCHMARK.md` in Shmulik's repo): the model is MsBERT fine-tuned on the strict-preserved non-biblical corpus (editorial reconstructions redacted, so it has provably never seen a modern guess). At each real lacuna the target is masked; the MLM proposes vocabulary candidates; a filter keeps only candidates that contain the visibly surviving letters in order (with left/right anchoring) and fit the estimated gap length; survivors are ranked by MLM score. A target counts as a hit when any bibliographically attributed, physically compatible restoration appears in Top-K. Stable across ±0/±1/±2 tolerance; fully offline against a cached QD snapshot.

**63.5% vs 9.5% on identical targets is the central finding so far: what the model is told about the gap matters far more than which model fills it** (every model-vs-model gap measured in R1 is ≤ 8 points). Note also 69.5% under exact length — the method *improves* as physical measurement improves, which is the practical case for real P0 data (§6). Known limits: single-word lacunae (78% of real gaps, per §6); candidates limited to single-WordPiece vocabulary words (a ceiling a byte-level generator could remove); agreement with scholars, not physical truth; n = 74.

**R2b. Partial-letters conditioning — the §6c regime, first implementation (2026-08-02).** The MsBERT engine above uses the surviving letters only as a post-hoc *filter* on vocabulary candidates. A character-level engine can instead *condition* on them: pre-fill the surviving letters into the masked character slots (every placement compatible with anchoring and order) and beam-search only the unknown positions, ranking by mean log-probability of the generated characters. Same benchmark, same physical evidence, same scoring:

| Engine (same QD setup, length ±1; n = 74) | Top-1 | Top-10 | Top-20 |
|---|---|---|---|
| MsBERT FT — vocab-rank, letters as filter (R2 headline) | 40.5% | 63.5% | 67.6% |
| dictabert-char FT — letters as conditioning | 44.6% | **66.2%** | **73.0%** |
| TavBERT base — same engine | **45.9%** | 62.2% | 67.6% |
| TavBERT FT (Itay's recipe) — same engine | 43.2% | 59.5% | 63.5% |

Findings: (a) character-level conditioning is competitive with filtered MsBERT: the best observed Top-1 is TavBERT base at 45.9% (+5.4 points), while dictabert-char FT gives the best Top-10/20 at 66.2%/73.0%; (b) every model remains far above the 20.3% human first-pass Top-1 control; (c) Top-10 stays in a narrow 59.5–66.2% band, so these pilot differences are not strong model-ranking evidence; (d) the char engine removes the single-WordPiece candidate ceiling and directly implements the physically faithful partial-letters regime. Caveats: single run, n = 74, and a new mean-log-probability ranking heuristic. Implementation and result artifacts: `experiments/external_finetune/qd_char_engine.py` and `experiments/results/runs/qd_char/`.

### R3. Shmulik's original evidence register (pre-unification)

Full detail: `docs/RESULTS.md` (evidence register, 2026-07-25); every row has a checked-in artifact; none is a frozen paper result.

| # | Benchmark (unit, scope) | Key numbers | Class |
|---|---|---|---|
| A | Preserved-word recovery — 300 intact words, held-out non-bib, editorial text redacted | MsBERT-preserved 13.7% Top-1, 30.7% Top-5, **36.3% Top-10**, 43.7% Top-20 | synthetic diagnostic |
| B | QD real lacunae — 74 targets, visible letters + length ±1 | **40.5% / 63.5% / 67.6%** (Top-1/10/20); unconstrained 9.5% Top-10 | literature-agreement pilot |
| C | Train-only RAG ablation | QD 63.5% → 63.5%; TF single-word 60.0% → 64.0% (25 spans); multiword slots 41.4% → 41.8% | pilot |
| D | Embible-style synthetic spans — 30 held-out, K∈{1,2,3}, unknown length | word-only 16.7% Top-10; base TavBERT 6.7%; **all systems 0% on 2–3-word strata**; oracle-boundary ceiling 33.3% | synthetic diagnostic |
| E | Bible domain transfer — same decoder on Embible verses | Bible 50.0% vs DSS 16.7% balanced Top-10 (1-word: 80.0% vs 50.0%) → the gap is the domain, not the decoder | transfer diagnostic |
| F | Expanded model selection — 300 held-out spans | word-only 15.0% Top-10 (best); TavBERT-preserved 6.0%; fusion/RAG not promoted | selection pilot |
| G | Quote-aware cross-corpus source recovery — 35 known-source Pesher passages | 86.6% Top-1 / 99.1% Top-3 book recovery; survives trigram ablation (52.4% Top-1, p=0.0008) | method validation (paper-facing) |

The register also lists claims the repo explicitly does **not** make (no end-to-end accuracy, no RAG gain, no SOTA, no unknown-length multiword success). Rows D–F pin the honest open problem for the unified benchmark: **2–3-word unknown-length restoration is at ~0% for every current system.**

### R4. Fresh fine-tunes behind R1 (this document's runs)

| Model | Recipe | Stopped | Best checkpoint | probe_exact |
|---|---|---|---|---|
| TavBERT full-FT | Itay's TuningConfig, `ppp_nonbib` | epoch 4 | epoch 3 | 0.1425 → 0.1471 |
| MsBERT full-FT | same | epoch 14 | epoch 12 | 0.1048 → 0.1572 |

### R5. Evaluation-integrity audit (2026-08-02)

A line-by-line audit of both scoring paths for gold-information leaks, so every number above can be trusted at face value:

| Surface checked | Verdict |
|---|---|
| QD real-lacuna scorer: gold text reaching model input | **Clean.** Any context word containing a bracketed reconstruction is redacted to `<GAP>`; the target is always masked; no code path can insert a reading into the prompt. Visible-segment anchoring uses only letters outside editorial brackets — genuinely preserved characters. |
| QD length constraint provenance | **Clean.** Estimated length comes from editorial slot counts (`○` markers / display slots), not from the specific attributed reading being scored. The one near-self-referential case — the "initial reading" — is reported only as the labeled human-baseline control, never in the headline metric. |
| Itay's masking: `[MASK]` count = gold token count | **Known oracle, labeled.** This is exactly the O-len regime of §5 (exact char count for TavBERT, WordPiece count for MsBERT); every affected number in this document carries the label. |
| Unaligned-word exclusion inflating hit@k | **Known, labeled.** §8; both scoring variants reported in R1. |
| ByT5 adapter: other masked spans visible in context | **Asymmetry favoring ByT5, disclosed in R1.** Consistent between its training and eval, so no train/eval mismatch — and ByT5 loses despite the advantage. |
| Split hygiene (`ppp_nonbib` xlsx + R1 sample) | **Clean, verified directly:** 0 scrolls appear in more than one split; all 100 R1 sample sentences are test-split; 0 test sentences share exact text with any train sentence. |
| ByT5 unified training data | **Clean.** Trained on the train split only; the R1 sample is entirely test-split (above). |

Standing rule going forward: any new evaluation path gets this audit before its numbers enter the document, and every length-informed number carries its §5 regime label. Applied to §R2b's char engine: its conditioning input is only physically preserved letters and editorial slot counts — the same provenance as the R2 engine, no gold-derived information.

### R6. Model-lineup quick checks — dicta char family (2026-08-02)

Zero-shot and fine-tuned checks on a fixed 30-sentence slice of the R1 sample (250 masked words, Itay's runner, hit@10; aligned-only = headline for char models since they align 100% of words):

| Model | hit@10 |
|---|---|
| TavBERT base | 24.4% [18.1, 31.9] |
| **dictabert-char, fine-tuned (Itay's recipe, epoch-6 stop, probe-best)** | **22.8%** [17.0, 29.0] |
| TavBERT fine-tuned | 22.4% |
| dictabert-char base | 17.6% |
| dictabert-large-char base (0.4B) | 17.6% |

Takeaways: modern-Hebrew pretraining costs the dicta models ~7 points zero-shot, and fine-tuning on 1,783 scrolls sentences recovers all of it (probe_exact 0.113 → 0.140, the largest fine-tune gain of any char model measured); large-char at 4× parameters gains nothing zero-shot, so scale alone does not transfer to Qumran Hebrew — large-char *fine-tuned* is the open GPU-pass question; nothing separates the fine-tuned dicta model from TavBERT at n=250. Artifacts: `experiments/results/runs/quick30_summary.json`.

---

## Proposed next steps

1. **Meeting:** walk the 12 decision points (Appendix A is the checklist); most have a concrete proposal to accept, amend, or reject.
2. **Freeze the split** (§3): generate the sha1 scroll assignment once, commit the JSON, both codebases load it.
3. **GPU pass** (supersedes R1/R6): train the same realistic DSS char-restoration objective from TavBERT and dictabert-char, 3 seeds each, and evaluate base + fine-tuned checkpoints on the full 338-sentence test split and `lacuna-real`; include MsBERT base + fine-tuned as the WordPiece baseline. Both scoring variants (§8), paired McNemar tests, and dev-only checkpoint/model selection. Run dictabert-large-char afterward as a one-seed scale ablation only if the base pipeline is stable; do not rerun ByT5-small.
4. **Build the `lacuna-real` track** (§6–§7): span lengths from the empirical distribution, real P0 char budgets from recorded gap positions, and the partial-letters regime — the novel contribution candidate.
5. **Grow the real-lacuna target set** (§10) beyond the current 74 targets, jointly.
6. **Merge repositories** per §12, including the tests and cleanup items.

---

## Appendix

### A. Decision summary

| # | Decision | Proposal |
|---|---|---|
| 1 | PPP words in training | yes (ratio ≥ 0.5); test strict-preserved; one strict control |
| 2 | Benchmark unit | sentences (Itay's xlsx); chunks = training option |
| 3 | Split | sha1 buckets → frozen scroll-list JSON, loaded everywhere |
| 4 | Model roster | char-MLM core: dictabert-char vs TavBERT; MsBERT baseline; large-char scale ablation; ByT5 retained as negative result only |
| 5 | Length regimes | label all numbers U0 / O-len / P0; headline = P0 (O-len as labeled interim proxy); U0 = ablation |
| 6 | Mask statistics | sample from real lacuna distribution; real char budgets; partial-letters regime |
| 7 | Masking tracks | `scatter-30` + `lacuna-real` |
| 8 | Unaligned words | miss in headline; aligned-only secondary |
| 9 | Metrics & artifacts | union of both stacks; freeze schema |
| 10 | Real-lacuna track | include (74 targets now; grow jointly) |
| 11 | Checkpoints & seeds | probe-best; 3 seeds where compute allows; full FT; recipe recorded |
| 12 | Repo backbone | neutral `CandidateGenerator` core; shared constraints/scoring/artifacts; modules over notebooks |

### B. Already agreed (no decision needed)
Scroll-disjoint splitting · non-biblical main corpus · zero editor-supplied letters in the test set · durable per-run artifacts with config fingerprints.

### C. Numbers we stand behind
Rule: no figure enters the joint paper unless it traces to a checked-in artifact we generated or verified ourselves. That is: everything in §R1 (fresh paired runs, artifacts in `merged_results/`), §R2/R3 (evidence register with artifacts), §6 (lacuna distributions from 12,971 gap runs / 16,769 char patterns), and §R4. Previously circulated figures not present in this document are retired; the forbidden-claims test (§9) enforces the retirement at unification. The full 338-sentence GPU run supersedes §R1 when it lands.
