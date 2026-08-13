# Full methods and results

Date: 25 July 2026  
Project: reconstruction-free Dead Sea Scroll restoration and cross-corpus
source detection

## Executive summary

This project began with a restoration question:

> Can language models and retrieval from related Hebrew books recover missing
> Dead Sea Scroll text?

The experiments did not find a reliable restoration improvement from retrieval,
embeddings, candidate injection, or the current sequence model. Those negative
results are important and are retained below.

The work did identify a stronger and more defensible paper contribution:

> A quote-aware cross-corpus method can recover independently known biblical
> source books from preserved Pesher passages, even after exact shared phrases
> have been removed.

The final method has two channels:

1. **Literal source recovery:** rank external books with word and character
   TF-IDF.
2. **Residual source affinity:** remove every DSS word participating in an exact
   external phrase, then rank the remaining text using a feature family selected
   without access to the manuscript being evaluated.

The strongest validation results are:

| Evaluation | Text masked | Surviving exact matches | Nested Top-1 | Nested Top-3 | Permutation p, Top-3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Three-word phrase removal | 27.0% | 0 trigrams | 52.4% | 83.6% | 0.0008 |
| Two-word phrase removal | 58.4% | 0 bigrams | 45.3% | 82.2% | 0.0012 |

These are manuscript-macro results on Pesharim whose source books are already
known. They validate the detection method; they do not establish new historical
source relationships.

No unknown composition/source pair survives the corrected exploratory tests.
Consequently, the paper can make a strong methodological claim while keeping
specific historical connections as candidates for philological review.

## 1. Research questions

The project addresses two related but distinct questions.

### 1.1 Restoration

Given preserved words on both sides of a synthetically hidden span, can a model
recover the complete hidden sequence without being told:

- its character length;
- its word count; or
- its word boundaries?

This is evaluated only where the hidden answer is physically preserved text.
It is a synthetic-damage diagnostic, not a claim about unknowable original text
at a real lacuna.

### 1.2 Cross-corpus source connection

Given a physically preserved DSS passage, can a transparent retrieval method:

- recover a known scriptural source;
- distinguish literal quotation from residual affinity;
- search beyond the Qumran collection in the Hebrew Bible, inscriptions, and
  later Hebrew comparators; and
- avoid converting text similarity into unsupported claims about authorship,
  borrowing, chronology, or lost sources?

The second question produced the useful paper method.

## 2. Corpus and provenance

### 2.1 Preserved DSS corpus

The training and query corpus is:

`curation/derived/preserved_nonbib_chunks.jsonl`

The raw `dss_chunks.csv` is used for non-textual labels such as composition and
genre, but it is not itself treated as reconstruction-free training text.

The corpus validator confirms:

- 1,647 preserved non-biblical chunks;
- 27,814 lacuna records;
- 736 scrolls;
- disjoint train, development, and held-out scroll sets;
- training tokens contain only preserved Hebrew or the anonymous `<GAP>`
  marker; and
- no brackets, unknown-letter markers, or reconstructed letters occur in
  training text.

Modern editorial reconstructions are not used as training labels, restoration
answers, retrieval records, or source-query text.

### 2.2 External comparison shelf

The external shelf contains 5,775 windows from 49 books or sources:

| Corpus | Material | Role |
| :--- | :--- | :--- |
| ETCBC BHSA | Consonantal Hebrew Bible | Scriptural source and parallel search |
| ETCBC extrabiblical | Early Hebrew inscriptions | Diachronic comparators |
| ETCBC extrabiblical | Pirqe Avot | Later rabbinic comparator |
| ETCBC extrabiblical | Mekhilta Shirata | Later reception/shared-source comparator |

Qumran works included in the ETCBC extrabiblical package are excluded from the
external shelf. The connection analysis therefore searches books outside the
Qumran collection rather than matching one scroll against another copy in the
same shelf.

External texts are divided into 100-word windows with 15-word overlap. A
40-word sensitivity was also run, but it changes which DSS passages are
eligible and must not be interpreted as a clean window-size treatment effect.

## 3. Evaluation principles

The following rules are used throughout:

1. **Preserved-only labels:** synthetic restoration targets are physically
   preserved words.
2. **Unknown span length:** exact word count and boundaries are not supplied to
   primary restoration systems.
3. **Development-only selection:** penalties, retrieval weights, proposal
   limits, and feature choices are selected without held-out restoration
   answers.
4. **Exact complete-span metrics:** partial words or correct individual slots do
   not count as a correct restored span.
5. **Manuscript clustering:** source-control accuracy is macro-averaged by
   manuscript, and permutation inference operates at manuscript or scroll
   level.
6. **Answer-removal stress:** retrieval experiments repeat analysis after
   documents containing the complete hidden answer are excluded.
7. **Negative results remain visible:** an expanded candidate pool is not called
   an accuracy improvement.
8. **Bounded historical claims:** similarity is not automatically interpreted
   as authorship, borrowing, direction of influence, or a lost source.

## 4. Restoration baseline

The restoration work has two different evaluation tracks:

- synthetic damage, where preserved text is hidden and the exact answer is
  observable; and
- real lacunae, where the original text is unobservable and evaluation measures
  agreement with attributed scholarly proposals.

They must not be combined into one accuracy number.

## 4A. Original real-lacuna research

### 4A.1 Research question

At a genuinely damaged DSS location, does the preserved-only model rank one or
more physically compatible, attributed scholarly restorations highly?

This is the original lacuna-research track. It remains part of the project and
is complementary to both synthetic restoration and source detection.

Because the missing manuscript text is physically absent, no experiment can
observe the original answer directly. The correct endpoint is therefore:

> agreement with attributed scholarly proposals under explicit physical
> constraints

It is not “accuracy on the original wording.”

### 4A.2 Qumran Digital benchmark

The strongest real-lacuna benchmark uses:

- 74 held-out non-biblical single-word lacuna targets;
- 99 distinct target/proposed-reading pairs;
- attributed readings from Qumran Digital;
- visible surviving letters;
- approximate word length with tolerance of one character; and
- a preserved-only masked-language model.

The model ranks candidates compatible with the surviving physical evidence.

| Condition | Top-1 | Top-10 | Top-20 |
| :--- | ---: | ---: | ---: |
| Physically constrained candidate ranking | 40.5% | 63.5% | 67.6% |
| Same targets without physical constraints | — | 9.5% | — |

The manuscript-cluster bootstrap 95% interval for constrained Top-10 is
51.4%–74.3%.

This is a meaningful result: the model frequently places an attributed
scholarly proposal among its first ten physically compatible candidates.
However, several limitations remain:

- the sites are selected disputed or documented locations rather than a random
  sample of all DSS damage;
- the benchmark is single-word only;
- the target is literature agreement, not physical truth;
- multiple scholarly readings may be compatible; and
- the result does not establish that the model improves a scholar's final
  judgment.

### 4A.3 Train-only RAG on the same real lacunae

Retrieval indexes only physically preserved text from training scrolls.
Retrieval weights are selected on development scrolls, and the same 74 held-out
targets are compared pairwise.

| Condition | Exact target-level Top-10 |
| :--- | ---: |
| Preserved-only model | 63.5% |
| Preserved-only model plus train-only RAG | 63.5% |

RAG does not improve the Qumran Digital result. This is a neutral downstream
result, not evidence against the usefulness of displaying parallels to a
scholar.

### 4A.4 Earlier Text-Fabric lacuna diagnostics

A separate fixed sample uses anonymous Text-Fabric editorial reconstructions as
evaluation labels. These are useful diagnostics but have a weaker evidential
status than attributed Qumran Digital proposals.

| Unit | No RAG Top-10 | Train-only RAG Top-10 |
| :--- | ---: | ---: |
| Single-word spans, 25 targets | 60.0% | 64.0% |
| Multiword slots, 440 slots in 100 spans | 41.4% | 41.8% |
| Exact multiword sequence, 100 spans | 7.0% | 9.0% |

The single-word and slot-level movements are small. The exact multiword result
is the relevant complete-span metric, but the observed 2-point increase is not
presented as a general RAG improvement because:

- the sample is small;
- word-slot count is known in this older diagnostic;
- the references are editorial labels rather than observable originals;
- compatible alternative readings are not exhaustively represented; and
- no positive inferential claim was established.

Older 48.0% and 36.8% RAG figures belong to legacy exploratory runs with
different units or protocols and are not current evidence.

### 4A.5 Role in the paper

The original real-lacuna research should remain a separate paper track:

1. **Known-answer synthetic recovery:** test model behavior where complete
   preserved answers are observable.
2. **Real-lacuna literature agreement:** test whether attributed proposals are
   ranked under physical constraints.
3. **Source and parallel assistance:** provide quote-aware external evidence
   without claiming that retrieval automatically restores the lacuna.
4. **Scholar study:** test whether candidates and parallels improve expert
   review, calibration, or efficiency.

The new quote-aware method strengthens the third component. It does not replace
the original lacuna benchmark. A useful interface can show:

- the damaged location and surviving letters;
- physically compatible restoration candidates;
- attributed scholarly readings;
- literal external parallels;
- residual source affinities after quotation masking; and
- explicit provenance for every item.

The final scholar study should compare at least:

- manuscript context only;
- candidates only;
- parallels only; and
- candidates plus parallels.

Primary artifacts for the original lacuna track:

- `comparison/reports/QD_RESEARCHER_BENCHMARK.md`
- `comparison/reports/qd_researcher_comparison.json`
- `comparison/reports/PRESERVED_RAG_LACUNA_LENGTHS.md`
- `comparison/reports/preserved_rag_lacuna_lengths.json`
- `docs/METHODOLOGY.md`, especially Track B

### 4.1 Task

One, two, or three contiguous preserved DSS words are hidden. Each target has
eight visible words on the left and eight on the right. The principal diagnostic
contains:

- 60 development spans;
- 300 held-out spans;
- 100 held-out targets at each word length; and
- no gold length or word-boundary information.

### 4.2 Best implemented baseline

The best current restoration baseline is a preserved-only masked-word model
that independently searches one-, two-, and three-word hypotheses and combines
them with a development-selected length penalty.

| System | Exact Top-10 |
| :--- | ---: |
| Preserved-only unknown-length word model | 15.0% |
| Fine-tuned TavBERT character model | 6.0% |
| Embible-style overlap ensemble | 4.7% |
| Development-fitted rank fusion | 15.0% |

The simpler word model is retained because rank fusion ties it rather than
improving it.

Word-model performance by hidden span length is:

| Hidden words | Exact Top-10 | Candidate-pool recall |
| ---: | ---: | ---: |
| 1 | 40.0% | 47.0% |
| 2 | 5.0% | 14.0% |
| 3 | 0.0% | 1.0% |
| Overall | 15.0% | 20.7% |

This exposes the central restoration bottleneck: the correct multiword sequence
usually never enters the candidate pool.

The one-token-per-word decoder can represent 92.3% of target words and 86.3% of
complete spans. Three-word complete-span representability is 77.0%. All
unrepresentable spans remain in the metric denominator.

## 5. External retrieval for restoration

### 5.1 Retrieval shelves

The frozen candidate generator was evaluated with:

1. no retrieval;
2. DSS training text only;
3. Hebrew Bible;
4. early Hebrew inscriptions;
5. later rabbinic comparators;
6. all external sources; and
7. DSS training text plus all external sources.

Visible left and right context form the retrieval query. Retrieval never sees
the hidden answer. Each shelf retrieves its Top-20 passages.

### 5.2 Reranking result

Candidate support from retrieval was added to the existing model ranking. The
retrieval weight was selected on the 60 development targets.

Every shelf selected weight zero. All held-out conditions therefore remained:

- exact Top-1: 4.3%;
- exact Top-10: 15.0%; and
- candidate-pool recall: 20.7%.

The combined DSS-plus-external shelf retrieved a Top-20 document containing the
complete answer for 28.7% of targets. This did not help when the answer was
absent from the model's candidate pool or scored below the Top-10 boundary.

Conclusion: the retriever locates relevant material, but reranking a weak fixed
pool cannot solve candidate-generation failure.

## 6. Retrieval-conditioned candidate injection

### 6.1 Method

For each shelf:

- retrieve using visible context only;
- extract equal numbers of one-, two-, and three-word phrases;
- score every proposed phrase with the same sequential masked-token likelihood
  used by the baseline decoder;
- choose the proposal limit on development data; and
- repeat with all documents containing the complete gold phrase removed.

An implementation audit compared scores for phrases appearing in both the
original beam and injected set. Maximum disagreement stayed below the declared
0.001 tolerance.

### 6.2 Development-selected result

Every shelf selected zero proposals. Held-out exact Top-10 remained 15.0%.

### 6.3 Forced maximum-injection diagnostic

The following diagnostic forces 50 proposals per word length. It was not
selected by development data and is used only to locate the bottleneck.

| Shelf | Candidate recall | Gold phrases proposed | Exact Top-10 |
| :--- | ---: | ---: | ---: |
| DSS training only | 26.7% | 49/300 | 15.0% |
| Hebrew Bible | 22.0% | 24/300 | 15.0% |
| Early inscriptions | 21.0% | 12/300 | 15.0% |
| Later rabbinic | 21.7% | 16/300 | 15.0% |
| All external | 22.0% | 23/300 | 15.0% |
| DSS plus external | 26.0% | 46/300 | 15.0% |

When documents containing the complete answer are removed, candidate recall
returns to the 20.7% baseline in every condition. All new gold proposals depend
on literal occurrence of the answer in a retrieved passage.

Conclusion: retrieval can supply missing literal phrases, but the current word
model does not rank them usefully.

## 7. Copy-aware fusion

A follow-up combined:

- word-model likelihood;
- proposal rank; and
- normalized retrieval support.

Only development data were used. The best configuration recovered one
additional Top-10 case out of 60 but lost both baseline Top-1 hits. External
shelves did not improve the development result.

This was judged too unstable to promote. No fresh restoration sample was spent
on it.

## 8. Tokenization-free ByT5

### 8.1 Evaluator correction

The original ByT5 evaluator had an invalid fallback: it placed the synthetic
target inside the visible left context. That leakage was removed. The corrected
evaluator:

- uses the same deterministic development sample as the word baseline;
- hides the complete span;
- supplies no length or boundary information;
- checks exact complete sequences; and
- records the model and sample hashes.

### 8.2 Result

The available checkpoint is a complete 1.1 GB one-epoch ByT5 pilot.

On 60 leakage-safe development targets:

- exact Top-1: 1.7%;
- exact Top-5: 1.7%;
- exact Top-10: 1.7%;
- one-word Top-10: 5.0%;
- two-word Top-10: 0.0%; and
- three-word Top-10: 0.0%.

Outputs collapse toward frequent forms such as `כול` and `ו`. The checkpoint is
not promoted and was not evaluated on held-out restoration data.

## 9. Cross-corpus source retrieval

### 9.1 Signals

The source screen keeps these representations distinguishable:

- word TF-IDF over unigrams and bigrams;
- character TF-IDF over three- to five-character n-grams;
- length-normalized surface-style features; and
- MiqraBERT embeddings as a separately reported sensitivity.

Results are diversified to at most one passage per external book before
source-book Top-k is computed.

### 9.2 Known-source positive control

The control consists of 35 reconstruction-free passages from eight mapped
Pesher manuscripts:

- 1QpHab → Habakkuk;
- 4Q161–4Q164 → Isaiah;
- 4Q166 → Hosea;
- 4Q169 → Nahum; and
- 4Q171 → Psalms.

The mapping is used only for evaluation, never as a retrieval feature.

With 100-word external windows, the 80% word / 20% character TF-IDF ranker
achieves:

- manuscript-macro Top-1: 86.6%;
- manuscript-macro Top-3: 99.1%;
- Top-1 manuscript-label permutation p: 0.0005; and
- Top-3 manuscript-label permutation p: 0.0005.

This demonstrates that the transparent retriever can recover independently
known source books before it is applied to unknown cases.

## 10. Embedding comparison

MiqraBERT embeddings were computed using the locally cached revision
`1ab168a2306a28652bd86afa47f77a04c51781f8`, mean pooling, local-only loading,
and disabled remote code.

### 10.1 One-hundred-word windows

| Method | Pesher Top-1 | Pesher Top-3 |
| :--- | ---: | ---: |
| TF-IDF | 86.6% | 99.1% |
| MiqraBERT embeddings | 25.9% | 32.1% |
| Fixed 50/50 hybrid | 42.6% | 53.6% |

### 10.2 Forty-word sensitivity

This run admits a different query set and is not directly comparable to the
100-word run.

| Method | Pesher Top-1 | Pesher Top-3 |
| :--- | ---: | ---: |
| TF-IDF | 50.7% | 69.9% |
| MiqraBERT embeddings | 13.4% | 21.0% |
| Fixed 50/50 hybrid | 23.3% | 33.0% |

Embeddings underperform the transparent TF-IDF baseline in both matched
comparisons. They remain a secondary discovery channel, not the primary source
ranker.

This does not establish that embeddings are generally unsuitable for ancient
Hebrew. The result may reflect passage length, DSS/Bible domain shift,
orthographic variation, pooling, or MiqraBERT's training objective.

## 11. Final method: quote-aware source connection

### 11.1 Motivation

High source-book recovery may be trivial if a Pesher passage contains a long
quotation from the source. The final method therefore asks whether source
recovery survives after literal shared phrases are removed.

### 11.2 Phrase-removal algorithm

For a threshold \(n\):

1. construct an inventory of every external \(n\)-gram;
2. locate every DSS \(n\)-gram appearing in that inventory;
3. mark every DSS token participating in any matching window;
4. replace each complete marked run with `__QUOTE_GAP__`;
5. retain the marker in retrieval text so words on opposite sides cannot form
   an artificial new phrase;
6. retain passages with at least 20 unmasked words; and
7. fail the run if any matching external \(n\)-gram remains inside an unmasked
   segment.

The primary analysis removes exact trigrams. Exact-bigram removal is a stricter
sensitivity.

### 11.3 Feature-family sensitivity

The literal and residual analyses report:

- the fixed 80/20 word-character combination;
- word TF-IDF alone; and
- character n-grams alone.

After trigram removal:

| Residual signal | Top-1 | Top-3 | Top-3 permutation p |
| :--- | ---: | ---: | ---: |
| Fixed combined | 41.1% | 76.5% | 0.0005 |
| Word only | 33.9% | 74.4% | 0.0005 |
| Character only | 52.4% | 83.6% | 0.0005 |

After bigram removal:

| Residual signal | Top-1 | Top-3 | Top-3 permutation p |
| :--- | ---: | ---: | ---: |
| Fixed combined | 23.1% | 49.0% | 0.0311 |
| Word only | 23.1% | 42.2% | 0.1284 |
| Character only | 45.3% | 82.2% | 0.0019 |

The strict mask damages word features much more than character features. That
pattern is expected if residual source affinity is distributed across
orthographic and morphological fragments rather than surviving complete
phrases.

### 11.4 Nested manuscript validation

Reporting the best feature on the same manuscripts used to choose it would
inflate performance. The final validation therefore uses nested
leave-one-manuscript-out selection:

1. hold out one Pesher manuscript;
2. evaluate combined, word-only, and character-only rankings on the other
   manuscripts;
3. select by macro Top-3, then Top-1, with a fixed tie-break;
4. score the untouched manuscript;
5. repeat for every manuscript; and
6. permutation-test the complete nested procedure by shuffling expected source
   books at manuscript level and repeating feature selection.

Every residual fold selects character n-grams using only the other
manuscripts.

| Mask | Manuscripts | Passages | Mean masked | Nested Top-1 | Nested Top-3 | Top-3 p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trigram | 8 | 35 | 27.0% | 52.4% | 83.6% | 0.0008 |
| Bigram | 7 | 30 | 58.4% | 45.3% | 82.2% | 0.0012 |

One manuscript loses too much text to meet the 20-word threshold under the
bigram mask. This is why the strict sensitivity has seven rather than eight
manuscripts.

### 11.5 Supported methodological claim

The current evidence supports:

> A quote-aware character n-gram retriever can recover independently known
> biblical source books from preserved Pesher passages even after all exact
> external bigrams are removed, under leave-one-manuscript-out feature
> selection.

This is a surface-affinity claim. Character n-grams can capture:

- orthography;
- morphology;
- recurring subword forms;
- dialect;
- genre; and
- short formulaic language.

The method does not by itself prove semantic paraphrase.

## 12. Unknown source candidates

The screen produces plausible candidates such as:

- Hodayot ↔ Psalms and Isaiah;
- Instruction ↔ Proverbs;
- rewritten Pentateuch ↔ Exodus and Numbers;
- Temple Scroll ↔ Leviticus, Numbers, and Deuteronomy;
- Jubilees ↔ Genesis and other Pentateuchal books; and
- 4QMMT ↔ legal/scriptural and later-reception comparators.

These patterns are philologically plausible, but the corrected corpus-wide
tests do not confirm a new relationship.

### 12.1 Multiplicity control

For residual rankings:

- composition labels are shuffled at DSS scroll level;
- all observed composition/source pairs are tested;
- zero-support hypotheses remain part of the correction family; and
- Benjamini-Hochberg correction is applied across the full family.

Results:

| Mask | Hypotheses corrected | Smallest adjusted q | Significant at q ≤ 0.05 |
| :--- | ---: | ---: | ---: |
| Trigram | 287 | 0.5739 | 0 |
| Bigram | 266 | 0.7582 | 0 |

Therefore, none of the unknown connections should be presented as a discovery.
They are ranked candidates for expert adjudication and future confirmatory
testing.

## 13. What failed and what it taught us

| Attempt | Result | Lesson |
| :--- | :--- | :--- |
| Retrieval reranking | Every shelf selected weight zero | Relevant documents do not help when the correct candidate is absent |
| Phrase injection | Candidate recall increased; Top-10 did not | Literal availability is insufficient without a better scoring model |
| Answer-removal injection | Recall returned to baseline | New gold candidates depended on verbatim answer occurrence |
| Copy-aware fusion | One extra dev Top-10, both Top-1 hits lost | Too unstable for confirmation |
| MiqraBERT embeddings | Far below TF-IDF source recovery | Current semantic representation is poorly calibrated to this task |
| One-epoch ByT5 | 1.7% dev Top-10, no multiword hits | Current sequence checkpoint is mode-collapsed |
| Unknown source enrichment | No corrected pair significant | Candidate lists require philology and confirmatory data |

These failures support the final paper design. They show why the contribution
should be source detection with explicit quotation controls, not a claim that
RAG solves restoration.

## 14. Paper framing

### 14.1 Recommended central claim

The paper should center on:

> Quote-aware source attribution for preserved ancient Hebrew passages, using
> known-source calibration, exact-phrase ablation, nested manuscript validation,
> and explicit multiplicity control.

### 14.2 Restoration's role

Restoration remains valuable as:

- a downstream stress test;
- evidence that source retrieval and restoration are different tasks;
- a negative result demonstrating the candidate-generation bottleneck; and
- a future expert-assistance application.

The paper should not claim:

- improved DSS restoration from external retrieval;
- state-of-the-art restoration;
- recovery of original text at real lacunae;
- semantic paraphrase detection;
- authorship identification;
- direct borrowing;
- direction of influence;
- a lost source; or
- a statistically confirmed new relationship for an unmapped composition.

## 15. Recommended follow-up

### 15.1 Expert adjudication

Ask DSS scholars to review a blinded sample containing:

- known literal source matches;
- known-source residual matches;
- top unknown residual candidates; and
- matched negative controls.

The annotation form should distinguish:

- direct quotation;
- close textual parallel;
- likely shared source;
- thematic or generic similarity;
- orthographic/dialectal affinity; and
- no meaningful relationship.

### 15.2 New external corpora

Highest-priority additions are:

1. Hebrew Ben Sira;
2. a broader, versioned Mishnah and early midrash shelf;
3. Samaritan Pentateuch;
4. more dated Hebrew inscriptions; and
5. separately modeled Aramaic comparators.

### 15.3 Confirmatory source benchmark

The present nested control is strong method validation but was developed after
the initial Pesharim screen. A final paper should, if possible, add a new
confirmatory collection of independently mapped passages or freeze the current
pipeline before adding a new corpus.

### 15.4 Restoration

Do not run more retrieval-weight grids over the current candidate pool. A future
restoration attempt needs:

- a better preserved-only character or byte sequence model;
- multi-seed training;
- development-only checkpoint and decoding selection;
- a frozen, non-overlapping confirmatory sample; and
- exact multiword recovery as a promotion requirement.

## 16. Reproduction

Run from the repository root.

### 16.1 Validate the corpus

```bash
PYTHONPATH=. .venv/bin/python data/validate_preserved_nonbib_corpus.py
```

### 16.2 Primary quote-aware analysis

```bash
PYTHONPATH=. .venv/bin/python analysis/cross_corpus_quote_ablation.py
```

### 16.3 Strict bigram sensitivity

```bash
PYTHONPATH=. .venv/bin/python analysis/cross_corpus_quote_ablation.py \
  --quote-ngram 2 \
  --residual-min-words 20 \
  --output-json comparison/reports/cross_corpus_quote_ablation_bigram.json \
  --output-markdown comparison/reports/CROSS_CORPUS_QUOTE_ABLATION_BIGRAM.md
```

### 16.4 Tests and protocol validation

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_cross_corpus_quote_ablation.py \
  tests/test_cross_corpus_connections.py \
  tests/test_cross_corpus_retrieval_ablation.py \
  tests/test_cross_corpus_retrieval_injection.py \
  tests/test_tokenization_free_benchmark.py \
  tests/test_paper_protocol.py

PYTHONPATH=. .venv/bin/python eval/validate_paper_protocol.py
```

## 17. Artifact index

### Main reading documents

- `docs/QUOTE_AWARE_SOURCE_CONNECTION_METHOD.md`
- `docs/CROSS_CORPUS_RESEARCH_AGENDA.md`
- `docs/RESULTS.md`
- `docs/BEST_METHOD.md`

### Main generated reports

- `comparison/reports/CROSS_CORPUS_CONNECTIONS.md`
- `comparison/reports/cross_corpus_connections.json`
- `comparison/reports/CROSS_CORPUS_QUOTE_ABLATION.md`
- `comparison/reports/cross_corpus_quote_ablation.json`
- `comparison/reports/CROSS_CORPUS_QUOTE_ABLATION_BIGRAM.md`
- `comparison/reports/cross_corpus_quote_ablation_bigram.json`
- `comparison/reports/CROSS_CORPUS_EMBEDDING_BENCHMARK.md`
- `comparison/reports/cross_corpus_embedding_benchmark.json`
- `comparison/reports/CROSS_CORPUS_RETRIEVAL_ABLATION.md`
- `comparison/reports/cross_corpus_retrieval_ablation.json`
- `comparison/reports/CROSS_CORPUS_RETRIEVAL_INJECTION.md`
- `comparison/reports/cross_corpus_retrieval_injection.json`
- `comparison/reports/byt5_dev_benchmark.json`

### Implementations

- `analysis/cross_corpus_connections.py`
- `analysis/cross_corpus_embedding_benchmark.py`
- `analysis/cross_corpus_quote_ablation.py`
- `eval/cross_corpus_retrieval_ablation.py`
- `eval/cross_corpus_retrieval_injection.py`
- `eval/tf_tokenization_free_benchmark.py`

## 18. Final decision

The useful paper method is the quote-aware source-connection pipeline.

The strongest result is not that AI reconstructs the Dead Sea Scrolls. It is
that a controlled computational method can:

- recover known source books;
- measure how much recovery depends on literal quotation;
- preserve a significant source signal after severe phrase removal;
- select its representation without using the manuscript under evaluation; and
- refuse to promote unknown historical connections when corrected evidence is
  insufficient.

That is both technically reproducible and historically responsible.
