# Guide: `heldout_failures_refined_hebrew_only_fast.csv`

This file is a spreadsheet-friendly view of **real failure cases** from the current best local model:

- `model = ft_msbert_span_refined`
- `split = heldout-scrolls`
- `book_filter = excluding-conservative-aramaic-books`

In other words, these are misses from a **held-out benchmark**, after excluding a conservative list of clearly Aramaic books.

## What each column means

| Column | Meaning |
| --- | --- |
| `model` | The checkpoint that produced the predictions. |
| `split` | The evaluation split. Here it is the held-out book-level split, meaning these scrolls were not used in training. |
| `book_filter` | Additional corpus filter applied during evaluation. Here it means a conservative Hebrew-only control. |
| `gold` | The true reconstructed word in the fully preserved source text. |
| `gold_fit_freq` | How many times the exact gold word appears in the `fit` partition used during development/training. This is a quick proxy for lexical rarity. |
| `category` | A rough automatic label for the type of failure. |
| `why` | A one-line explanation of why the row was assigned to that category. |
| `context` | Short local context around the target word in the raw Text-Fabric token stream. The missing position is marked as `⬚⬚⬚`. This can look unnatural because clitics are often split into separate one-letter tokens. |
| `context_joined` | A more readable version of the same context after conservative re-joining of likely one-letter proclitics with the following word. This is the better column for human inspection. |
| `top1` ... `top5` | The model’s first five candidate predictions for the gap. |
| `top5_joined` | The same five candidates in one cell, useful for quick reading/filtering in spreadsheet tools. |

## How to read the categories

| Category | Interpretation |
| --- | --- |
| `same_length_semantic` | The model is not just guessing a short particle. It is preferring another plausible content word of similar length. This usually points to a genuine contextual or semantic ambiguity. |
| `particle_or_too_short` | The model’s top guess is much shorter than the gold word. This often reflects the clitic / one-letter-prefix problem we observed in the training data. |
| `rare_or_unseen` | The gold form is very rare or absent in the development/training partition. This points to a data sparsity problem more than a pure ranking problem. |
| `other` | A miss that does not fit the simple heuristics above. These cases usually need manual inspection. |

## Practical ways to inspect the sheet

Recommended sorts / filters:

1. Sort by `category` to see which failure types cluster together.
2. Then sort by `gold_fit_freq` ascending to isolate rare-word problems.
3. Filter to `particle_or_too_short` to inspect the clitic issue directly.
4. Filter to `same_length_semantic` to find places where broader context or better domain conditioning may help.

## Important caveat

This CSV is meant for **qualitative inspection**, not for reporting final percentages by itself.

- It contains real held-out failures.
- But it is still a **sampled diagnostic view**, not the full benchmark table.
- Its value is mainly to help decide what kind of improvements are still plausible.
