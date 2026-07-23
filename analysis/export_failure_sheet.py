"""Build a reader-friendly CSV/MD pair from the raw failure export."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.clitic_join import join_likely_clitics

INFILE = ROOT / "analysis" / "reports" / "heldout_failures_refined_hebrew_only_fast.csv"
OUT_CSV = ROOT / "analysis" / "reports" / "heldout_failures_refined_hebrew_only_reader_friendly.csv"
OUT_MD = ROOT / "analysis" / "reports" / "heldout_failures_refined_hebrew_only_reader_friendly.md"


def likely_issue(row):
    category = row["category"]
    if category == "particle_or_too_short":
        return "Split-clitic noise / short-word bias"
    if category == "rare_or_unseen":
        return "Rare or unseen target word"
    if category == "same_length_semantic":
        return "Plausible but wrong content-word choice"
    return "Needs manual inspection"


def reader_note(row):
    category = row["category"]
    gold = row["gold"]
    top1 = row["top1"]
    if category == "particle_or_too_short":
        return f"The model chose a much shorter word ('{top1}') instead of the target '{gold}'."
    if category == "rare_or_unseen":
        return f"The target '{gold}' is very rare in the fit data, so the model backs off to safer guesses."
    if category == "same_length_semantic":
        return f"The model stays in the right semantic area, but chooses the wrong word instead of '{gold}'."
    return f"The model misses '{gold}' for a reason not captured by the simple heuristics."


def joined_context(raw_context):
    words = raw_context.split()
    if "⬚⬚⬚" not in words:
        return raw_context
    idx = words.index("⬚⬚⬚")
    placeholder = "__LACUNA__"
    words[idx] = placeholder
    joined, _ = join_likely_clitics(" ".join(words))
    return joined.replace(placeholder, "⬚⬚⬚")


def build_csv():
    with INFILE.open() as fh:
        rows = list(csv.DictReader(fh))

    fieldnames = [
        "target_word",
        "context_for_reading",
        "model_top1",
        "model_top2",
        "model_top3",
        "model_top4",
        "model_top5",
        "all_top5",
        "likely_issue",
        "reader_note",
        "target_fit_frequency",
        "raw_context",
        "raw_category",
    ]

    out_rows = []
    for row in rows:
        out_rows.append({
            "target_word": row["gold"],
            "context_for_reading": joined_context(row["context"]),
            "model_top1": row["top1"],
            "model_top2": row["top2"],
            "model_top3": row["top3"],
            "model_top4": row["top4"],
            "model_top5": row["top5"],
            "all_top5": row["top5_joined"],
            "likely_issue": likely_issue(row),
            "reader_note": reader_note(row),
            "target_fit_frequency": row["gold_fit_freq"],
            "raw_context": row["context"],
            "raw_category": row["category"],
        })

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    return len(out_rows)


def build_md(nrows):
    OUT_MD.write_text(
        "\n".join([
            "# Guide: `heldout_failures_refined_hebrew_only_reader_friendly.csv`",
            "",
            "This is a reader-facing version of the held-out failure sheet.",
            "",
            "What changed relative to the raw diagnostic CSV:",
            "",
            "- `context_for_reading` is the main context column. It conservatively re-joins likely one-letter clitics with the following word, so the text is easier to read.",
            "- The prediction columns were renamed to `model_top1` ... `model_top5`.",
            "- `likely_issue` translates the technical failure category into a more human-readable label.",
            "- `reader_note` gives a one-sentence explanation of why the row is interesting.",
            "- The raw diagnostic fields are still preserved as `raw_context` and `raw_category` for traceability.",
            "",
            "Recommended reading order:",
            "",
            "1. Read `context_for_reading` and `target_word` first.",
            "2. Compare them to `model_top1` and then to `all_top5`.",
            "3. Use `likely_issue` to group rows into improvement themes.",
            "4. Use `target_fit_frequency` to spot sparsity problems.",
            "",
            f"Rows in this file: {nrows}",
            "",
        ]),
        encoding="utf-8",
    )


def main():
    nrows = build_csv()
    build_md(nrows)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print(f"rows={nrows}")


if __name__ == "__main__":
    main()
