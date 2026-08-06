# Quote-aware cross-corpus connection analysis

Status: **paper-method candidate; substantive connections remain exploratory**.

This analysis separates literal source recovery from residual affinity.
Every DSS token participating in an exact external 2-word
match is replaced by a boundary marker before the residual ranking.

## Validation and ablation

- DSS passages before masking: 290
- DSS passages retaining at least 20 words: 226
- Mean words masked: 58.4%
- Median words masked: 58.4%
- Surviving exact 2-word matches inside residual runs: 0
- Known-source Pesher Top-1 / Top-3 before masking: 84.7% / 99.0%
- Known-source Pesher Top-1 / Top-3 after masking: 23.1% / 49.0%

A large control drop is expected and useful: it demonstrates that the
literal channel detects known quotations. Connections that persist in the
residual channel cannot be explained by an exact external 2-word run left in the query, although genre, date,
dialect, and formulaic language remain alternative explanations.

### Feature-family sensitivity

| Ranking signal | Literal Top-3 | Residual Top-3 | Residual p |
| :--- | ---: | ---: | ---: |
| word_only | 99.0% | 42.2% | 0.1284 |
| character_only | 99.0% | 82.2% | 0.0019 |

Feature choice was also repeated in nested leave-one-manuscript-out
validation, selecting among combined, word-only, and character-only
rankings on the other manuscripts before scoring each held-out Pesher.
Nested residual Top-1 / Top-3: 45.3% / 82.2% (Top-3 permutation p=0.0012).

## Composition-level decomposition

### 4QMMT (N=3)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Ezechiel | 1 | 1 | 33.3% |
| bhsa:Jesaia | 1 | 1 | 33.3% |
| bhsa:Sacharia | 1 | 1 | 33.3% |
| bhsa:Ecclesiastes | 0 | 1 | 33.3% |
| bhsa:Genesis | 0 | 1 | 33.3% |
| bhsa:Psalmi | 0 | 1 | 33.3% |
| bhsa:Reges_II | 0 | 1 | 33.3% |
| bhsa:Samuel_II | 0 | 1 | 33.3% |
| extrabiblical:Pirqe | 0 | 1 | 33.3% |
| bhsa:Chronica_I | 2 | 0 | 0.0% |

### Book_of_Jubilees (N=1)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Numeri | 1 | 1 | 100.0% |
| extrabiblical:Pirqe | 1 | 1 | 100.0% |
| extrabiblical:Shirata | 1 | 1 | 100.0% |

### Hodayot (N=77)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| extrabiblical:Shirata | 26 | 40 | 51.9% |
| bhsa:Numeri | 34 | 39 | 50.6% |
| bhsa:Psalmi | 39 | 26 | 33.8% |
| bhsa:Jesaia | 23 | 22 | 28.6% |
| bhsa:Jeremia | 11 | 10 | 13.0% |
| bhsa:Iob | 9 | 9 | 11.7% |
| bhsa:Reges_I | 4 | 8 | 10.4% |
| bhsa:Proverbia | 11 | 7 | 9.1% |
| bhsa:Ezechiel | 8 | 7 | 9.1% |
| bhsa:Genesis | 1 | 7 | 9.1% |

### Instruction (N=32)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Numeri | 20 | 19 | 59.4% |
| extrabiblical:Shirata | 14 | 15 | 46.9% |
| bhsa:Jesaia | 7 | 11 | 34.4% |
| bhsa:Proverbia | 12 | 7 | 21.9% |
| bhsa:Psalmi | 6 | 5 | 15.6% |
| bhsa:Jeremia | 1 | 5 | 15.6% |
| bhsa:Ecclesiastes | 3 | 4 | 12.5% |
| bhsa:Leviticus | 1 | 3 | 9.4% |
| bhsa:Iob | 5 | 2 | 6.2% |
| bhsa:Judices | 3 | 2 | 6.2% |

### Pesharim (N=30)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Ezechiel | 9 | 11 | 36.7% |
| extrabiblical:Shirata | 4 | 9 | 30.0% |
| bhsa:Psalmi | 9 | 8 | 26.7% |
| bhsa:Habakuk | 13 | 7 | 23.3% |
| bhsa:Jeremia | 8 | 7 | 23.3% |
| bhsa:Jesaia | 7 | 7 | 23.3% |
| bhsa:Numeri | 7 | 7 | 23.3% |
| bhsa:Samuel_I | 3 | 4 | 13.3% |
| bhsa:Daniel | 1 | 3 | 10.0% |
| bhsa:Leviticus | 1 | 3 | 10.0% |

### Temple Scroll (N=73)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Numeri | 36 | 43 | 58.9% |
| extrabiblical:Shirata | 2 | 32 | 43.8% |
| bhsa:Leviticus | 35 | 18 | 24.7% |
| bhsa:Jesaia | 4 | 16 | 21.9% |
| bhsa:Deuteronomium | 28 | 12 | 16.4% |
| bhsa:Reges_I | 13 | 11 | 15.1% |
| bhsa:Ezechiel | 24 | 10 | 13.7% |
| bhsa:Samuel_I | 3 | 9 | 12.3% |
| bhsa:Chronica_II | 10 | 8 | 11.0% |
| bhsa:Reges_II | 6 | 8 | 11.0% |

### rewritten pentateuch (N=10)

| Source | Literal Top-3 | Residual Top-3 | Residual support |
| :--- | ---: | ---: | ---: |
| bhsa:Numeri | 3 | 5 | 50.0% |
| bhsa:Exodus | 7 | 4 | 40.0% |
| bhsa:Ezechiel | 3 | 3 | 30.0% |
| extrabiblical:Shirata | 2 | 3 | 30.0% |
| bhsa:Josua | 0 | 3 | 30.0% |
| bhsa:Deuteronomium | 2 | 2 | 20.0% |
| bhsa:Jeremia | 0 | 2 | 20.0% |
| bhsa:Samuel_II | 3 | 1 | 10.0% |
| bhsa:Leviticus | 1 | 1 | 10.0% |
| extrabiblical:Pirqe | 1 | 1 | 10.0% |

## Scroll-cluster residual enrichment

Composition labels are shuffled at the scroll level. Benjamini-Hochberg
correction covers every observed composition/source pair.

No residual pair survives correction at q <= 0.05.

## Claim boundary

Residual enrichment is evidence of a reproducible corpus affinity, not proof of authorship, direct borrowing, a lost source, or direction of influence. Genre, date, dialect, shared tradition, and formulaic language require philological adjudication.
