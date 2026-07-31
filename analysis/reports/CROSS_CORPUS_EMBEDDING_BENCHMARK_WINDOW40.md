# Cross-corpus embedding retrieval benchmark

Status: **exploratory calibration**. The embedding model is used as a
retriever, not as proof of borrowing, authorship, or source direction.

## Fair comparison

- All methods use the same reconstruction-free DSS passages.
- All methods search the same external passage windows.
- Rankings keep at most one result per external book.
- No retrieval weights were fitted on the positive control.
- The control is macro-averaged and permuted at DSS manuscript level.

## Known-source Pesher control

| Retrieval method | Manuscripts | Passages | Macro Top-1 | Macro Top-3 | Top-1 p | Top-3 p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf_80_word_20_char | 12 | 55 | 50.7% | 69.9% | 0.0001 | 0.0001 |
| miqrabert_embedding | 12 | 55 | 13.4% | 21.0% | 0.2438 | 0.06869 |
| fixed_50_tfidf_50_embedding | 12 | 55 | 23.3% | 33.0% | 0.0013 | 0.009399 |

## Decision rule

TF-IDF remains the primary interpretable screen unless embeddings or the
pre-specified 50/50 hybrid improve known-source recovery. Even if they do,
the channels remain visible separately during passage adjudication.

MiqraBERT was trained for Biblical Hebrew parallel retrieval. Its result
is a useful semantic sensitivity test, but its Bible-domain training can
favor biblical sources and does not validate rabbinic or epigraphic links.
