# Guide: `heldout_failures_refined_hebrew_only_reader_friendly.csv`

This is a reader-facing version of the held-out failure sheet.

What changed relative to the raw diagnostic CSV:

- `context_for_reading` is the main context column. It conservatively re-joins likely one-letter clitics with the following word, so the text is easier to read.
- The prediction columns were renamed to `model_top1` ... `model_top5`.
- `likely_issue` translates the technical failure category into a more human-readable label.
- `reader_note` gives a one-sentence explanation of why the row is interesting.
- The raw diagnostic fields are still preserved as `raw_context` and `raw_category` for traceability.

Recommended reading order:

1. Read `context_for_reading` and `target_word` first.
2. Compare them to `model_top1` and then to `all_top5`.
3. Use `likely_issue` to group rows into improvement themes.
4. Use `target_fit_frequency` to spot sparsity problems.

Rows in this file: 86
