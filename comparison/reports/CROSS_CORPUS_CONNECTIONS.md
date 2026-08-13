# Cross-corpus DSS connection screen

Status: **exploratory candidate screen**. This report does not establish
authorship, direct borrowing, or the existence of a lost source.

## Scope

- DSS query passages: 290
- Query compositions: 7
- External passages: 5775
- External books/sources: 49
- Signals kept separate: lexical, character/orthographic, and surface style.
- Top matches are diversified to at most one passage per external book.

## External shelf

| Corpus | Material | Interpretive role |
| :--- | :--- | :--- |
| BHSA | Full consonantal Hebrew Bible | Scriptural source/parallel search |
| ETCBC/extrabiblical | Early inscriptions | Diachronic language controls |
| ETCBC/extrabiblical | Pirqe Avot | Later rabbinic comparator |
| ETCBC/extrabiblical | Mekhilta Shirata | Later reception comparator; contains biblical material |

## Known-source positive control

Before interpreting unknown connections, the same unsupervised ranking
was tested on Pesharim whose scriptural source book is already known.

- Manuscripts: 8
- Passages: 35
- Macro Top-1 source-book recovery: 86.6%
- Macro Top-3 source-book recovery: 99.1%
- Manuscript-level permutation p-values: Top-1=0.0005, Top-3=0.0005

## Composition-level candidates

### Temple Scroll

| External source | Top-3 support | Strong parallels | Query passages | Support | Mean lexical |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bhsa:Leviticus | 41 | 28 | 86 | 47.7% | 0.142 |
| bhsa:Numeri | 41 | 23 | 86 | 47.7% | 0.149 |
| bhsa:Ezechiel | 29 | 14 | 86 | 33.7% | 0.127 |
| bhsa:Deuteronomium | 28 | 26 | 86 | 32.6% | 0.294 |
| bhsa:Exodus | 24 | 16 | 86 | 27.9% | 0.114 |
| bhsa:Reges_I | 15 | 3 | 86 | 17.4% | 0.110 |
| bhsa:Chronica_II | 13 | 6 | 86 | 15.1% | 0.113 |
| bhsa:Jeremia | 9 | 6 | 86 | 10.5% | 0.102 |
| bhsa:Judices | 8 | 4 | 86 | 9.3% | 0.104 |
| bhsa:Reges_II | 7 | 5 | 86 | 8.1% | 0.116 |

### Instruction

| External source | Top-3 support | Strong parallels | Query passages | Support | Mean lexical |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bhsa:Numeri | 23 | 0 | 45 | 51.1% | 0.102 |
| extrabiblical:Shirata | 21 | 0 | 45 | 46.7% | 0.086 |
| bhsa:Proverbia | 14 | 1 | 45 | 31.1% | 0.076 |
| bhsa:Jesaia | 11 | 0 | 45 | 24.4% | 0.082 |
| bhsa:Psalmi | 9 | 0 | 45 | 20.0% | 0.073 |
| bhsa:Iob | 7 | 0 | 45 | 15.6% | 0.081 |
| bhsa:Exodus | 5 | 1 | 45 | 11.1% | 0.076 |
| bhsa:Samuel_I | 3 | 0 | 45 | 6.7% | 0.071 |
| bhsa:Nehemia | 3 | 1 | 45 | 6.7% | 0.061 |
| extrabiblical:Pirqe | 3 | 0 | 45 | 6.7% | 0.098 |

### Hodayot

| External source | Top-3 support | Strong parallels | Query passages | Support | Mean lexical |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bhsa:Psalmi | 43 | 5 | 83 | 51.8% | 0.075 |
| bhsa:Numeri | 36 | 0 | 83 | 43.4% | 0.094 |
| extrabiblical:Shirata | 28 | 1 | 83 | 33.7% | 0.086 |
| bhsa:Jesaia | 24 | 5 | 83 | 28.9% | 0.085 |
| bhsa:Proverbia | 12 | 1 | 83 | 14.5% | 0.072 |
| bhsa:Jeremia | 11 | 0 | 83 | 13.2% | 0.075 |
| bhsa:Ecclesiastes | 10 | 1 | 83 | 12.0% | 0.070 |
| bhsa:Ezechiel | 9 | 0 | 83 | 10.8% | 0.076 |
| bhsa:Daniel | 8 | 2 | 83 | 9.6% | 0.080 |
| bhsa:Iob | 8 | 2 | 83 | 9.6% | 0.080 |

### Pesharim

| External source | Top-3 support | Strong parallels | Query passages | Support | Mean lexical |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bhsa:Habakuk | 13 | 13 | 35 | 37.1% | 0.247 |
| bhsa:Psalmi | 10 | 6 | 35 | 28.6% | 0.145 |
| bhsa:Jesaia | 10 | 5 | 35 | 28.6% | 0.192 |
| bhsa:Ezechiel | 9 | 2 | 35 | 25.7% | 0.075 |
| bhsa:Numeri | 8 | 0 | 35 | 22.9% | 0.095 |
| bhsa:Jeremia | 8 | 3 | 35 | 22.9% | 0.086 |
| bhsa:Nahum | 6 | 6 | 35 | 17.1% | 0.211 |
| bhsa:Hosea | 5 | 3 | 35 | 14.3% | 0.136 |
| bhsa:Deuteronomium | 4 | 3 | 35 | 11.4% | 0.084 |
| bhsa:Micha | 4 | 2 | 35 | 11.4% | 0.083 |

### rewritten pentateuch

| External source | Top-3 support | Strong parallels | Query passages | Support | Mean lexical |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bhsa:Exodus | 13 | 12 | 23 | 56.5% | 0.217 |
| bhsa:Numeri | 11 | 6 | 23 | 47.8% | 0.174 |
| bhsa:Deuteronomium | 7 | 6 | 23 | 30.4% | 0.115 |
| bhsa:Reges_I | 6 | 2 | 23 | 26.1% | 0.135 |
| bhsa:Ezechiel | 6 | 3 | 23 | 26.1% | 0.136 |
| extrabiblical:Shirata | 5 | 1 | 23 | 21.7% | 0.113 |
| bhsa:Genesis | 4 | 2 | 23 | 17.4% | 0.139 |
| bhsa:Samuel_II | 3 | 2 | 23 | 13.0% | 0.088 |
| bhsa:Nehemia | 2 | 0 | 23 | 8.7% | 0.080 |
| bhsa:Leviticus | 2 | 1 | 23 | 8.7% | 0.144 |

### Book_of_Jubilees

| External source | Top-3 support | Strong parallels | Query passages | Support | Mean lexical |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bhsa:Genesis | 4 | 2 | 9 | 44.4% | 0.133 |
| bhsa:Numeri | 2 | 0 | 9 | 22.2% | 0.138 |
| extrabiblical:Shirata | 2 | 0 | 9 | 22.2% | 0.119 |
| extrabiblical:Pirqe | 2 | 0 | 9 | 22.2% | 0.137 |
| bhsa:Leviticus | 2 | 2 | 9 | 22.2% | 0.146 |
| bhsa:Jeremia | 2 | 0 | 9 | 22.2% | 0.087 |
| bhsa:Exodus | 2 | 2 | 9 | 22.2% | 0.088 |
| bhsa:Ecclesiastes | 2 | 1 | 9 | 22.2% | 0.070 |
| bhsa:Judices | 2 | 1 | 9 | 22.2% | 0.081 |
| bhsa:Reges_II | 1 | 1 | 9 | 11.1% | 0.147 |

### 4QMMT

| External source | Top-3 support | Strong parallels | Query passages | Support | Mean lexical |
| :--- | ---: | ---: | ---: | ---: | ---: |
| extrabiblical:Shirata | 5 | 2 | 9 | 55.6% | 0.104 |
| bhsa:Leviticus | 3 | 1 | 9 | 33.3% | 0.080 |
| bhsa:Ecclesiastes | 3 | 0 | 9 | 33.3% | 0.072 |
| bhsa:Ezechiel | 2 | 0 | 9 | 22.2% | 0.082 |
| bhsa:Numeri | 2 | 1 | 9 | 22.2% | 0.125 |
| bhsa:Chronica_I | 2 | 1 | 9 | 22.2% | 0.080 |
| bhsa:Esther | 2 | 1 | 9 | 22.2% | 0.072 |
| bhsa:Deuteronomium | 2 | 1 | 9 | 22.2% | 0.086 |
| bhsa:Josua | 1 | 0 | 9 | 11.1% | 0.082 |
| bhsa:Sacharia | 1 | 0 | 9 | 11.1% | 0.091 |

## Interpretation boundary

Connections are ranked hypotheses, not evidence of authorship, direct borrowing, or a lost source. Genre, date, dialect, formulaic language, and transmission can produce the same signal.

The next adjudication step should inspect exact passages, remove known
quotations and near-duplicates, repeat the analysis within matched genres,
and test whether each connection is stable across feature families and
window sizes.
