# Paper method: quote-aware cross-corpus source connection

Status: **validated method candidate; unknown historical connections remain
exploratory**.

## Contribution

The useful paper method is not another restoration reranker. It is a
quote-aware source-connection test for preserved Dead Sea Scroll text:

1. rank external source books using transparent word and character TF-IDF;
2. validate source-book recovery on Pesharim whose biblical source is already
   known;
3. mask every DSS word participating in an exact external phrase;
4. rerun retrieval on the remaining text, with boundary markers preventing
   artificial phrases across removed spans;
5. distinguish literal-source recovery from residual orthographic/lexical
   affinity;
6. select the residual feature family without using the manuscript being
   scored; and
7. cluster inference by manuscript and correct exploratory composition/source
   tests for multiple comparisons.

This operationalizes the important distinction in the Josephus-style research
question: a system should first show that it can recover named or otherwise
known sources, then ask whether a source relationship remains after the words
that could be direct quotation are removed.

## Data and ranking

- Queries: reconstruction-free, physically preserved non-biblical DSS chunks.
- External shelf: BHSA Hebrew Bible plus ETCBC early inscriptions, Pirqe Avot,
  and Mekhilta Shirata. Qumran books in the external ETCBC package are excluded.
- External windows: 100 words with 15-word overlap.
- Retrieval unit: passage, diversified to at most one result per external book.
- Literal rank: 80% within-query word TF-IDF percentile plus 20% character
  n-gram percentile.
- Residual candidates: combined, word-only, and character-only rankings.
- Primary control endpoint: macro source-book Top-3 by Pesher manuscript.

Modern reconstructions do not enter the query text. Corpus paths, selected-file
hashes, and repository revisions are recorded in the generated JSON.

## Quote ablation

For an ablation threshold \(n\), create an inventory of every external
\(n\)-gram. In each DSS passage, replace every complete run of tokens that
participates in a matching \(n\)-gram with a boundary marker. The marker is not
in the external vocabulary and prevents the words formerly surrounding a
quotation from becoming a new artificial bigram or trigram.

The implementation audits the resulting runs and fails if any external
\(n\)-gram survives. A passage is retained only when at least 20 unmasked words
remain. The three-word mask is the primary analysis; the two-word mask is a
strict sensitivity because it removes substantially more text.

## Validation result

| Condition | Manuscripts | Mean text masked | Literal Top-3 | Fixed combined residual Top-3 | Nested residual Top-3 | Nested permutation p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact 3-word mask | 8 | 27.0% | 99.1% | 76.5% | 83.6% | 0.0008 |
| Exact 2-word mask | 7 | 58.4% | 99.0% | 49.0% | 82.2% | 0.0012 |

Both ablations leave zero matching runs at their respective thresholds. In the
three-word condition, word-only residual Top-3 is 74.4% and character-only
Top-3 is 83.6% (both manuscript permutation p=0.0005). In the stricter
two-word condition, word-only recovery weakens to 42.2% and is not significant
(p=0.1284), while character-only remains 82.2% (p=0.0019).

The nested result is the paper-facing estimate. For each held-out Pesher
manuscript, the feature family is selected using macro Top-3 and Top-1 on the
other manuscripts. The permutation test shuffles expected source books at the
manuscript level and repeats the complete nested selection procedure. Every
fold selected character n-grams for the residual channel; no held-out
manuscript participated in its own feature choice.

## Interpretation

The result supports a bounded methodological claim:

> A quote-aware character n-gram retriever can recover known biblical source
> books from preserved Pesher passages even after all exact external bigrams
> are removed, under leave-one-manuscript-out feature selection.

It does **not** prove that the residual signal is semantic paraphrase. Character
n-grams can capture morphology, orthography, recurring short forms, dialect,
and genre. It also does not identify authorship, direction of influence, or a
lost source.

No unknown composition/source pair survives Benjamini-Hochberg correction at
q <= 0.05 after scroll-cluster permutation. Hodayot/Psalms,
Instruction/Proverbs, rewritten Pentateuch/Exodus, Temple Scroll/Pentateuch,
and later-reception matches remain philological candidates, not discoveries.
This negative multiplicity result must remain in the paper.

## Relationship to restoration

The restoration experiments motivate this pivot:

- external retrieval reranking selected weight zero on development data;
- literal candidate injection expanded candidate recall but did not improve
  exact Top-10;
- copy-aware fusion produced only one extra development Top-10 hit while losing
  both Top-1 hits;
- the one-epoch ByT5 checkpoint scored 1.7% exact Top-10 on the leakage-safe
  development sample and recovered no two- or three-word spans.

The paper should not claim that external books improve DSS restoration. It can
instead contribute a validated source-connection method and retain restoration
as a negative downstream ablation or future expert-assistance track.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python analysis/cross_corpus_quote_ablation.py

PYTHONPATH=. .venv/bin/python analysis/cross_corpus_quote_ablation.py \
  --quote-ngram 2 \
  --residual-min-words 20 \
  --output-json analysis/reports/cross_corpus_quote_ablation_bigram.json \
  --output-markdown analysis/reports/CROSS_CORPUS_QUOTE_ABLATION_BIGRAM.md

PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_cross_corpus_quote_ablation.py \
  tests/test_cross_corpus_connections.py
```

Primary generated evidence:

- `analysis/reports/CROSS_CORPUS_QUOTE_ABLATION.md`
- `analysis/reports/cross_corpus_quote_ablation.json`
- `analysis/reports/CROSS_CORPUS_QUOTE_ABLATION_BIGRAM.md`
- `analysis/reports/cross_corpus_quote_ablation_bigram.json`
