#!/usr/bin/env python3
"""Generate every numerical/synthesis table used by docs/paper.tex."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "paper_results_manifest.json"
SNAPSHOT = ROOT / "experiments" / "results" / "paper" / "paper_results_snapshot.json"
PROFILE = ROOT / "experiments" / "results" / "paper" / "paper_data_profile.json"
OUTPUT = ROOT / "docs" / "paper_tables.tex"


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}\\%"


def integer(value: int | float) -> str:
    return f"{int(value):,}"


def hydrate_data_profile(data: dict[str, object]) -> None:
    """Populate descriptive tables from the score-free data profile."""
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if profile["status"] != "descriptive_statistics_only_no_model_scores":
        raise ValueError("Data profile must not contain model results.")
    tables = {table["command"]: table for table in data["tables"]}
    corpus = profile["corpus"]
    heldout = corpus["splits"]["heldout"]
    heldout_shape = corpus["checkpoint_associated_heldout_shape"]
    tables["CorpusOverviewTable"]["rows"] = [
        ["Scroll identifiers", integer(corpus["scrolls"]), integer(heldout["scrolls"])],
        ["Training/evaluation chunks", integer(corpus["chunks"]), integer(heldout["chunks"])],
        ["Preserved word tokens", integer(corpus["preserved_words"]), integer(heldout["preserved_words"])],
        ["Lacuna records", integer(corpus["lacunae"]), integer(heldout["lacunae"])],
        ["Damaged word positions", integer(corpus["all_lacunae_shape"]["damaged_word_positions"]), integer(heldout_shape["damaged_word_positions"])],
    ]
    tables["CorpusOverviewTable"]["source"] = str(PROFILE.relative_to(ROOT))

    eligibility = corpus["scroll_eligibility"]
    tables["ScrollEligibilityTable"]["rows"] = [
        ["Archival registry", integer(eligibility["archival_registry_scrolls"]), "retain for provenance"],
        ["Primary MLM contributors", integer(eligibility["primary_mlm_scrolls"]), "at least one eligible chunk"],
        ["No primary MLM chunk", integer(eligibility["scrolls_without_primary_mlm_chunks"]), "exclude from MLM text"],
        ["Zero preserved words", integer(eligibility["zero_preserved_word_scrolls"]), "exclude from linguistic modeling"],
        ["Extreme fragmentation", integer(eligibility["extreme_fragmentation_scrolls"]), "flag; exclude from primary context task"],
    ]
    tables["ScrollEligibilityTable"]["source"] = str(PROFILE.relative_to(ROOT))

    audit = corpus["external_pipeline_audit"]
    strict = audit["strict_reconstruction_free_pipeline"]
    contiguous = audit["contiguous_run_pipeline"]
    tables["PipelineAuditTable"]["rows"] = [
        ["Model-contributing scrolls", integer(strict["model_contributing_scrolls"]), integer(contiguous["model_contributing_scrolls"])],
        ["Retained word tokens", integer(strict["retained_words"]), integer(contiguous["retained_words"])],
        ["Editorially reconstructed words", "0", f'{integer(contiguous["partially_reconstructed_words"])} ({pct(contiguous["partially_reconstructed_word_percent"])})'],
        ["Damage in model context", r"inline \texttt{<GAP>}", "breaks contiguous run"],
        ["Minimum local evidence", "20 preserved / 128 units", "10 contiguous words"],
    ]
    tables["PipelineAuditTable"]["source"] = str(PROFILE.relative_to(ROOT))

    all_buckets = corpus["all_lacunae_shape"]["gap_word_count"]["buckets"]
    heldout_buckets = heldout_shape["gap_word_count"]["buckets"]
    tables["LacunaShapeTable"]["rows"] = [
        [
            display,
            f'{integer(all_buckets[key]["n"])} ({pct(all_buckets[key]["percent"])})',
            f'{integer(heldout_buckets[key]["n"])} ({pct(heldout_buckets[key]["percent"])})',
        ]
        for display, key in (
            ("1", "1"),
            ("2", "2"),
            ("3", "3"),
            ("4--5", "4-5"),
            ("6+", "6+"),
        )
    ]
    tables["LacunaShapeTable"]["source"] = str(PROFILE.relative_to(ROOT))

    unknown = profile["evaluation_sets"]["unknown_length_synthetic"]
    qd = profile["evaluation_sets"]["qd_literature_agreement"]
    tables["EvaluationSetsTable"]["rows"] = [
        ["Unknown-length synthetic", integer(unknown["targets"]), integer(unknown["scrolls"]), "preserved complete span"],
        ["QD natural lacunae", integer(qd["targets"]), integer(qd["scrolls"]), f'{integer(qd["unique_target_readings"])} attributed readings'],
    ]
    tables["EvaluationSetsTable"]["source"] = str(PROFILE.relative_to(ROOT))


def hydrate_reproduced_results(data: dict[str, object]) -> None:
    """Replace paper-facing core metrics with the frozen reproduced snapshot."""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    tables = {table["command"]: table for table in data["tables"]}

    span_names = [
        "Preserved-only word span",
        "Preserved-only TavBERT",
        "Embible-style overlap",
        "Dev-fitted rank fusion",
    ]
    span_rows = []
    for name in span_names:
        result = snapshot["span_systems"][name]
        span_rows.append([
            name,
            pct(result["top1"]),
            pct(result["top5"]),
            pct(result["top10"]),
            f'{result["top10_scroll_cluster_ci"]["ci_low"]:.1f}--'
            f'{result["top10_scroll_cluster_ci"]["ci_high"]:.1f}\\%',
            f'{result["cer"]:.3f}',
        ])
    # Oracle length is a diagnostic from the same freshly rerun artifact.
    source = json.loads(
        (ROOT / snapshot["artifacts"]["span"]["path"]).read_text(encoding="utf-8")
    )
    oracle = source["results"]["cwc_word_oracle"]
    span_rows.append([
        "Oracle word-length filter",
        pct(oracle["top1"]), pct(oracle["top5"]), pct(oracle["top10"]),
        "--", f'{oracle["mean_top1_cer"]:.3f}',
    ])
    tables["SpanPilotTable"]["rows"] = span_rows
    tables["SpanPilotTable"]["source"] = snapshot["artifacts"]["span"]["path"]

    strata = source["results"]["by_word_count"]
    tables["SpanStrataTable"]["rows"][0] = [
        "Word-span exact Top-10",
        *[pct(strata[str(words)]["uwc_word"]["top10"]) for words in (1, 2, 3)],
    ]
    coverage = source["protocol"]["tokenizer_coverage"]["by_word_count"]
    tables["SpanStrataTable"]["rows"][1] = [
        "Complete-span coverage",
        *[
            pct(100 * coverage[str(words)]["fully_representable_span_rate"])
            for words in (1, 2, 3)
        ],
    ]
    tables["SpanStrataTable"]["source"] = snapshot["artifacts"]["span"]["path"]

    qd = snapshot["qd"]
    tables["QDTable"]["rows"] = [
        ["MLM: context only", *[pct(qd["context_only"][key]) for key in ("top1", "top5", "top10", "top20")]],
        ["Frequency: context only", *[pct(qd["frequency_context_only"][key]) for key in ("top1", "top5", "top10", "top20")]],
        ["MLM: visible traces", *[pct(qd["visible_only"][key]) for key in ("top1", "top5", "top10", "top20")]],
        ["Frequency: visible traces", *[pct(qd["frequency_visible_only"][key]) for key in ("top1", "top5", "top10", "top20")]],
        ["MLM: editor length (oracle)", *[pct(qd["editor_length_only"][key]) for key in ("top1", "top5", "top10", "top20")]],
        ["MLM: traces + length (oracle)", *[pct(qd["visible_plus_editor_length"][key]) for key in ("top1", "top5", "top10", "top20")]],
    ]
    low, high = qd["visible_only"]["top10_scroll_cluster_ci"]
    tables["QDTable"]["caption"] = (
        "Agreement with any attributed reading at 93 Qumran Digital targets. "
        f"The visible-trace MLM Top-10 scroll-cluster bootstrap 95\\% CI is {low:.1f}--{high:.1f}. "
        "All rows retain the same targets and references. Editor-derived length is "
        "reported only as an oracle-assisted condition."
    )
    tables["QDTable"]["source"] = snapshot["artifacts"]["qd"]["path"]

    byt5 = snapshot["byt5_checkpoint_replications"]
    tables["ByTCheckpointTable"]["rows"] = [
        [
            str(row["seed"]), str(row["epochs"]), str(row["batch_size"]),
            f'{row["learning_rate"]:.0e}', pct(row["top1"]), pct(row["top10"]),
        ]
        for row in byt5["checkpoints"]
    ]
    tables["ByTCheckpointTable"]["source"] = ", ".join(
        item["path"] for item in snapshot["artifacts"]["byt5"]
    )


def render_table(table: dict[str, object]) -> str:
    environment = str(table["environment"])
    width = r"\textwidth" if environment == "table*" else r"\linewidth"
    command = str(table["command"])
    header = " & ".join(table["header"])
    rows = "\n".join("  " + " & ".join(row) + r" \\" for row in table["rows"])
    return rf"""\newcommand{{\{command}}}{{%
\begin{{{environment}}}[t]
\centering
\caption{{{table['caption']}}}
\label{{{table['label']}}}
\resizebox{{{width}}}{{!}}{{%
\begin{{tabular}}{{{table['columns']}}}
\toprule
{header} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}%
}}
\end{{{environment}}}%
}}
"""


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data["status"] != "team_review_evidence_snapshot":
        raise ValueError("Paper manifest must be the team-review evidence snapshot.")
    hydrate_data_profile(data)
    hydrate_reproduced_results(data)
    commands: set[str] = set()
    labels: set[str] = set()
    rendered = ["% Generated by docs/generate_paper_tables.py; do not edit manually.\n"]
    for table in data["tables"]:
        command = table["command"]
        label = table["label"]
        if command in commands or label in labels:
            raise ValueError(f"Duplicate command or label: {command} / {label}")
        commands.add(command)
        labels.add(label)
        rendered.append(render_table(table))
    OUTPUT.write_text("\n".join(rendered), encoding="utf-8")


if __name__ == "__main__":
    main()
