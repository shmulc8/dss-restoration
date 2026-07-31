# Quote-aware cross-corpus connection analysis

Status: **paper-method candidate; substantive connections remain exploratory**.

This analysis separates literal source recovery from residual affinity.
Every DSS token participating in an exact external 3-word
match is replaced by a boundary marker before the residual ranking.

## Validation and ablation

- DSS passages before masking: 290
- DSS passages retaining at least 20 words: 289
- Mean words masked: 27.0%
- Median words masked: 24.7%
- Surviving exact 3-word matches inside residual runs: 0
- Known-source Pesher Top-1 / Top-3 before masking: 86.6% / 99.1%
- Known-source Pesher Top-1 / Top-3 after masking: 41.1% / 76.5%

A large control drop is expected and useful: it demonstrates that the
literal channel detects known quotations. Connections that persist in the
residual channel cannot be explained by an exact external 3-word run left in the query, although genre, date,
dialect, and formulaic language remain alternative explanations.

### Feature-family sensitivity

| Ranking signal | Literal Top-3 | Residual Top-3 | Residual p |
| :--- | ---: | ---: | ---: |
| word_only | 99.1% | 74.4% | 0.0005 |
| character_only | 99.1% | 83.6% | 0.0005 |

Feature choice was also repeated in nested leave-one-manuscript-out
validation, selecting among combined, word-only, and character-only
rankings on the other manuscripts before scoring each held-out Pesher.
Nested residual Top-1 / Top-3: 52.4% / 83.6% (Top-3 permutation p=0.0007999).

## Composition-level decomposition

### 4QMMT (N=9)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| extrabiblical:Shirata | 5 | 5 | 55.6% |
| bhsa:Numeri | 2 | 3 | 33.3% |
| bhsa:Ecclesiastes | 3 | 2 | 22.2% |
| bhsa:Leviticus | 3 | 2 | 22.2% |
| bhsa:Deuteronomium | 2 | 2 | 22.2% |
| bhsa:Reges_II | 1 | 2 | 22.2% |
| bhsa:Nehemia | 0 | 2 | 22.2% |
| bhsa:Chronica_I | 2 | 1 | 11.1% |
| bhsa:Ezechiel | 2 | 1 | 11.1% |
| bhsa:Daniel | 1 | 1 | 11.1% |

### Book_of_Jubilees (N=8)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Numeri | 2 | 3 | 37.5% |
| extrabiblical:Pirqe | 2 | 3 | 37.5% |
| bhsa:Jeremia | 2 | 2 | 25.0% |
| extrabiblical:Shirata | 2 | 2 | 25.0% |
| bhsa:Chronica_I | 1 | 2 | 25.0% |
| bhsa:Ezechiel | 1 | 2 | 25.0% |
| bhsa:Nehemia | 0 | 2 | 25.0% |
| bhsa:Genesis | 3 | 1 | 12.5% |
| bhsa:Exodus | 2 | 1 | 12.5% |
| bhsa:Judices | 2 | 1 | 12.5% |

### Hodayot (N=83)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Psalmi | 42 | 40 | 48.2% |
| bhsa:Numeri | 36 | 39 | 47.0% |
| extrabiblical:Shirata | 28 | 37 | 44.6% |
| bhsa:Jesaia | 24 | 29 | 34.9% |
| bhsa:Iob | 9 | 12 | 14.5% |
| bhsa:Jeremia | 11 | 11 | 13.3% |
| bhsa:Proverbia | 12 | 9 | 10.8% |
| bhsa:Ezechiel | 9 | 8 | 9.6% |
| bhsa:Reges_I | 7 | 7 | 8.4% |
| bhsa:Ecclesiastes | 10 | 6 | 7.2% |

### Instruction (N=45)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Numeri | 23 | 26 | 57.8% |
| extrabiblical:Shirata | 21 | 23 | 51.1% |
| bhsa:Proverbia | 14 | 14 | 31.1% |
| bhsa:Jesaia | 11 | 9 | 20.0% |
| bhsa:Psalmi | 10 | 8 | 17.8% |
| bhsa:Iob | 7 | 6 | 13.3% |
| bhsa:Jeremia | 2 | 6 | 13.3% |
| extrabiblical:Pirqe | 3 | 5 | 11.1% |
| bhsa:Judices | 3 | 4 | 8.9% |
| bhsa:Ecclesiastes | 3 | 3 | 6.7% |

### Pesharim (N=35)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Numeri | 8 | 12 | 34.3% |
| bhsa:Habakuk | 13 | 11 | 31.4% |
| bhsa:Jesaia | 10 | 10 | 28.6% |
| bhsa:Psalmi | 10 | 8 | 22.9% |
| bhsa:Jeremia | 8 | 7 | 20.0% |
| extrabiblical:Shirata | 4 | 7 | 20.0% |
| bhsa:Ezechiel | 9 | 5 | 14.3% |
| bhsa:Samuel_I | 3 | 5 | 14.3% |
| bhsa:Chronica_II | 4 | 4 | 11.4% |
| bhsa:Nahum | 6 | 3 | 8.6% |

### Temple Scroll (N=86)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Numeri | 41 | 49 | 57.0% |
| bhsa:Leviticus | 39 | 28 | 32.6% |
| bhsa:Deuteronomium | 30 | 26 | 30.2% |
| bhsa:Ezechiel | 29 | 25 | 29.1% |
| extrabiblical:Shirata | 2 | 24 | 27.9% |
| bhsa:Exodus | 24 | 18 | 20.9% |
| bhsa:Reges_I | 15 | 17 | 19.8% |
| bhsa:Chronica_II | 13 | 12 | 14.0% |
| bhsa:Jeremia | 9 | 10 | 11.6% |
| bhsa:Samuel_I | 3 | 6 | 7.0% |

### rewritten pentateuch (N=23)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Numeri | 11 | 13 | 56.5% |
| bhsa:Exodus | 13 | 8 | 34.8% |
| extrabiblical:Shirata | 5 | 7 | 30.4% |
| bhsa:Ezechiel | 6 | 5 | 21.7% |
| bhsa:Reges_I | 6 | 4 | 17.4% |
| bhsa:Leviticus | 2 | 4 | 17.4% |
| bhsa:Jesaia | 0 | 4 | 17.4% |
| bhsa:Chronica_II | 2 | 3 | 13.0% |
| bhsa:Josua | 1 | 3 | 13.0% |
| bhsa:Samuel_II | 3 | 2 | 8.7% |

## Scroll-cluster residual enrichment

Composition labels are shuffled at the scroll level. Benjamini-Hochberg
correction covers every observed composition/source pair.

No residual pair survives correction at q <= 0.05.

## Claim boundary

Residual enrichment is evidence of a reproducible corpus affinity, not proof of authorship, direct borrowing, a lost source, or direction of influence. Genre, date, dialect, shared tradition, and formulaic language require philological adjudication.
