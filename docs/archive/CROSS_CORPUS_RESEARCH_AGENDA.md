# Cross-corpus source and restoration research agenda

## Research question

Can preserved Dead Sea Scroll passages be connected to plausible scriptural
sources, earlier linguistic comparators, or later reception traditions—and do
those external connections improve restoration—without treating similarity as
proof of authorship or direct borrowing?

The current screen is a discovery instrument. It searches the Hebrew Bible,
early Hebrew inscriptions, Pirqe Avot, and Mekhilta Shirata with independent
lexical, orthographic, style, and semantic-embedding channels. Its known-source
control asks whether a method recovers the scriptural books underlying mapped
Pesharim before that method is trusted on unknown cases.

## Current calibration evidence

On 100-word external windows and 35 preserved passages from eight mapped
Pesharim manuscripts, the lexical/orthographic TF-IDF baseline recovered the
known source book at 86.6% macro Top-1 and 99.1% macro Top-3. MiqraBERT
embeddings reached 25.9% and 32.1%; a fixed 50/50 hybrid reached 42.6% and
53.6%.

A passage-length sensitivity run used 40-word external windows and admitted 55
passages from twelve mapped manuscripts. TF-IDF reached 50.7% Top-1 and 69.9%
Top-3, MiqraBERT reached 13.4% and 21.0%, and the hybrid reached 23.3% and
33.0%. Because the query set changes with the minimum passage length, these two
runs should not be compared as a window-size treatment effect. Within each run,
however, embeddings clearly underperform the identical TF-IDF baseline.

The present result therefore keeps embeddings as an independent sensitivity
and candidate-discovery channel. It does not justify blending them into the
primary source ranking. MiqraBERT was trained on Biblical Hebrew verse
parallels, so its failure here may reflect domain, vocalization, passage-length,
or DSS/Bible distribution mismatch rather than a general failure of semantic
representations.

## Completed restoration consequence test: external-retrieval ablation

Run the existing frozen lacuna-restoration benchmark under the following
retrieval shelves:

1. no retrieval;
2. DSS training passages only;
3. Hebrew Bible only;
4. early Hebrew inscriptions only;
5. later rabbinic comparators only;
6. all external sources;
7. DSS plus all external sources.

Keep the test spans, candidate generator, and evaluator fixed. Remove the held
out answer string and overlapping fragments from every retrieval index. Report
exact-span Top-1 and Top-10, reciprocal rank, and a paired bootstrap confidence
interval clustered by manuscript.

This is the most useful paper experiment because it tests a consequence:
whether external evidence recovers missing text better. A visually interesting
nearest neighbour is not enough.

### Completed first ablation

The single-checkpoint ablation has now been run on the frozen 60-development /
300-heldout expanded-pilot sample. Development selection chose retrieval weight
zero for every shelf. Consequently, held-out exact Top-10 remained 15.0% for
DSS-only, Bible, Hebrew inscriptions, rabbinic, all-external, and combined
retrieval.

This is not evidence that external texts contain no useful signal. The combined
DSS-plus-external shelf retrieved a Top-20 document containing the complete
answer for 28.7% of held-out spans, and 46/300 cases had both a model-generated
gold candidate and positive retrieval support. The reranker still could not
improve development accuracy.

The more important ceiling is candidate generation: the complete answer
appeared anywhere in the frozen model pool for only 20.7% of spans—47% for one
hidden word, 14% for two, and 1% for three. A reranker cannot recover an absent
candidate. The next restoration experiment should therefore test
retrieval-conditioned candidate generation or carefully controlled candidate
injection, rather than another weighting scheme over the same small pool. Its
answer-removal stress test must remain mandatory.

### Completed candidate-injection follow-up

Retrieval-conditioned candidate generation was tested with the same frozen
sample and checkpoint. Each shelf proposed equal numbers of one-, two-, and
three-word phrases from visible-context retrieval. The preserved-only word
model rescored every proposed phrase with the same sequential masked-token
likelihood used by its beam search. Development selected zero proposals for
every shelf, so held-out Top-10 remained 15.0%.

The unselected maximum-injection diagnostic clarifies the failure. Fifty DSS
training proposals per length raised held-out candidate recall from 20.7% to
26.7%, and combined DSS/external proposals raised it to 26.0%, but Top-10
remained 15.0%. Once documents containing the complete answer were removed,
candidate recall returned to 20.7%: every newly supplied gold phrase depended
on a retrieved verbatim occurrence. Thus this literal phrase-injection method
can expand the pool, but the current word model does not rank those additions
usefully. It does not test a decoder capable of generating a paraphrase from a
source that lacks the missing answer string.

## Source-connection experiments

### 1. Completed quote-aware residual-affinity test

The test now masks every word participating in an exact external three-word
overlap and reruns retrieval. It reports three layers:

- explicit or near-explicit quotation;
- residual thematic or paraphrastic affinity after masking;
- no stable connection.

This separates straightforward quotation detection from residual affinity. On
the known-source Pesher control, the literal ranker recovers the expected book
at 99.1% macro Top-3. After masking 27.0% of words on average and auditing that
zero external trigrams survive, nested leave-one-manuscript-out feature
selection reaches 83.6% Top-3 (permutation p=0.0008).

A strict exact-bigram sensitivity masks 58.4% of words, retains 226 passages,
and leaves zero external bigrams. Nested residual Top-3 remains 82.2%
(p=0.0012). Every fold selects character n-grams, while word-only performance
weakens under the bigram mask. This supports non-verbatim surface affinity, not
semantic paraphrase.

No unknown composition/source pair survives the scroll-cluster,
Benjamini-Hochberg-corrected screen at q <= 0.05. Candidate relationships must
therefore remain exploratory and receive philological adjudication.

### 2. Stability grid

Repeat candidate discovery across:

- 50-, 100-, and 150-word windows;
- word TF-IDF, character TF-IDF, MiqraBERT embeddings, and fixed hybrids;
- consonantal and normalized orthographic representations;
- all sources and date/genre-matched hard-negative shelves.

Promote only connections stable across multiple reasonable specifications.
Do not optimize weights on the same Pesharim control used for reporting.

### 3. Shared-source and reception network

Represent a proposed Qumran-to-rabbinic connection as a three-node motif:

`biblical source -> Qumran passage` and `biblical source -> later passage`.

After masking the shared biblical wording, test whether a residual Qumran/later
relationship remains. Unless additional historical evidence exists, label the
direction as unidentifiable. A Mekhilta or Pirqe match may show later reception,
a common biblical source, generic legal language, or chance; text similarity
alone cannot select among them.

### 4. Diachronic language axis

Build a separately licensed and versioned comparison shelf spanning:

- Iron Age and Persian-period Hebrew inscriptions;
- the Hebrew Bible;
- Ben Sira in Hebrew;
- Qumran Hebrew;
- the Mishnah and early midrash.

Match genres before inferring chronology. Use this axis to estimate linguistic
placement, not authorship. Ben Sira is the highest-priority missing comparator
because it is much closer to Qumran than broad rabbinic corpora while remaining
external to the scroll collection.

### 5. Source-conditioned restoration

For a passage with a stable source-family prediction, retrieve only passages
from that family and rerank restoration candidates. Evaluate on artificially
hidden preserved spans and on readings with explicit scholarly attribution.
Never present a model completion of a genuinely lost span as ground truth.

### 6. Textual layer replication

Replicate the published Hodayot clustering result on the reconstruction-free
corpus and test whether detected boundaries persist after controlling for
passage length, manuscript, and preservation density. This is a useful
validation track, but not the primary novelty because authorial clustering of
Qumran material has already been published.

### 7. Text and physical-scribe evidence

If manuscript images and palaeographic labels become available, compare textual
style clusters with image-based scribal clusters as independent evidence.
Agreement would be informative; disagreement would also be meaningful. Textual
style must not be relabelled as physical scribal identification.

## Corpus expansion priorities

1. Hebrew Ben Sira manuscripts, with explicit edition and reuse permission.
2. Broader Mishnah and early midrash, tracked per edition because licenses vary.
3. Samaritan Pentateuch for Pentateuchal textual-family comparisons.
4. More dated Hebrew and Aramaic inscriptions.
5. Elephantine and other Aramaic corpora in a separate Aramaic model.
6. Greek and Syriac witnesses only in an explicitly aligned multilingual track.

Languages and transmission traditions should not be mixed into one similarity
index without language-specific normalization and controls.

## Required claim controls

- Use preserved DSS text only; exclude modern reconstructions from queries.
- Record corpus versions and hashes.
- Flag exact quotations and near-duplicates rather than counting them as novel.
- Split and resample by manuscript, not overlapping passage window.
- Control for source-book size and diversify retrieved results by book.
- Compare within genre and plausible date before making diachronic claims.
- Report stability across window sizes and feature families.
- Keep lexical, orthographic, style, and embedding evidence visible separately.
- Do not infer direction, authorship, or a lost source from similarity alone.
- Require expert passage adjudication before promoting any candidate as a
  historical argument.

## Practical decision rule

Use TF-IDF as the transparent baseline. Add an embedding or hybrid method only
if it improves the manuscript-level known-source control or finds additional
candidates that survive masking and stability tests. Use the best calibrated
retriever in the external-restoration ablation, where improvement on held-out
preserved text is the decisive result.
