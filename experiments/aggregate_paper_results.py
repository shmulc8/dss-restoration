#!/usr/bin/env python3
"""Aggregate only freshly generated, target-level paper evidence.

This script never runs a model and never simulates outcomes. It validates that
all input artifacts use the same frozen target set, computes scroll-clustered
uncertainty and paired exact tests, audits split registries, and writes a single
shareable snapshot for the paper table generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPAN = ROOT / "experiments/results/paper/span_baselines_rerun_20260810.json"
DEFAULT_QD = ROOT / "experiments/results/paper/qd_msbert_rerun_20260810.json"
DEFAULT_BYT5 = tuple(
    ROOT / f"experiments/results/paper/byt5_unknown_length_seed{seed}.json"
    for seed in (41, 42, 43)
)
DEFAULT_OUTPUT = ROOT / "experiments/results/paper/paper_results_snapshot.json"
DEFAULT_MARKDOWN = ROOT / "experiments/results/paper/PAPER_RESULTS_SNAPSHOT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def topk_hit(case: dict[str, Any], system: str, k: int) -> int:
    gold = case["gold"]
    if isinstance(gold, list):
        gold = " ".join(gold)
    return int(gold in case["top10"][system][:k])


def clustered_ci(
    cases: list[dict[str, Any]],
    hit: Callable[[dict[str, Any]], int],
    *,
    samples: int = 10_000,
    seed: int = 20260810,
) -> dict[str, float | int]:
    by_scroll: dict[str, list[int]] = defaultdict(list)
    for case in cases:
        by_scroll[str(case["scroll"])].append(hit(case))
    scrolls = sorted(by_scroll)
    observed = [value for values in by_scroll.values() for value in values]
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled = [scrolls[generator.randrange(len(scrolls))] for _ in scrolls]
        values = [value for scroll in sampled for value in by_scroll[scroll]]
        estimates.append(100 * sum(values) / len(values))
    estimates.sort()
    return {
        "n": len(observed),
        "scroll_clusters": len(scrolls),
        "estimate": 100 * sum(observed) / len(observed),
        "ci_low": estimates[int(0.025 * samples)],
        "ci_high": estimates[int(0.975 * samples) - 1],
        "bootstrap_samples": samples,
    }


def exact_mcnemar(first: list[int], second: list[int]) -> dict[str, float | int]:
    if len(first) != len(second):
        raise ValueError("paired predictions have different lengths")
    first_only = sum(a == 1 and b == 0 for a, b in zip(first, second))
    second_only = sum(a == 0 and b == 1 for a, b in zip(first, second))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index) for index in range(min(first_only, second_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "p_exact_two_sided": p_value,
    }


def holm_adjust(comparisons: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(comparisons, key=lambda key: comparisons[key]["p_exact_two_sided"])
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        adjusted = min(1.0, (total - index) * comparisons[key]["p_exact_two_sided"])
        running = max(running, adjusted)
        comparisons[key]["p_holm"] = running


def split_audit(qd: dict[str, Any]) -> dict[str, Any]:
    preserved = load(ROOT / "curation/derived/preserved_nonbib_manifest.json")
    canonical = load(ROOT / "data_preparation/dss_scroll_splits_v1.json")
    legacy = {
        scroll: split
        for split, scrolls in preserved["scroll_splits"].items()
        for scroll in scrolls
    }
    current = canonical["scroll_assignment"]
    target_sigla = [str(target["siglum"]) for target in qd["targets"]]
    return {
        "targets": len(target_sigla),
        "unique_scrolls": len(set(target_sigla)),
        "model_associated_split": {
            split: sum(legacy.get(siglum) == split for siglum in target_sigla)
            for split in ("train", "dev", "heldout")
        },
        "later_canonical_registry": {
            split: sum(current.get(siglum) == split for siglum in target_sigla)
            for split in ("train", "val", "test")
        },
        "interpretation": (
            "All targets are held out under the split used to train the checkpoint. "
            "A later registry assigns some targets differently; the two registries "
            "must not be mixed in one experiment."
        ),
    }


def training_metadata(model_dir: str) -> dict[str, Any]:
    directory = Path(model_dir)
    metadata = load(directory / "byt5_training_metadata.json")
    return {key: metadata[key] for key in ("seed", "epochs", "batch_size", "learning_rate")}


def render_markdown(snapshot: dict[str, Any]) -> str:
    span = snapshot["span_systems"]
    qd = snapshot["qd"]
    lines = [
        "# Frozen paper-results snapshot",
        "",
        f"Status: **{snapshot['status']}**.",
        "",
        "## Unknown-length exact complete-span recovery",
        "",
        "| System | Top-1 | Top-5 | Top-10 | 95% scroll-cluster CI (Top-10) |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    for name, values in span.items():
        ci = values["top10_scroll_cluster_ci"]
        lines.append(
            f"| {name} | {values['top1']:.1f}% | {values['top5']:.1f}% | "
            f"{values['top10']:.1f}% | {ci['ci_low']:.1f}%--{ci['ci_high']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Qumran Digital literature agreement",
            "",
            f"- P0 Top-10: {qd['p0']['top10']:.1f}% "
            f"({qd['p0']['top10_scroll_cluster_ci'][0]:.1f}%--"
            f"{qd['p0']['top10_scroll_cluster_ci'][1]:.1f}%).",
            f"- U0 Top-10: {qd['u0']['top10']:.1f}%.",
            f"- P0 + retrieval Top-10: {qd['rag']['top10']:.1f}%.",
            f"- Split audit: {qd['split_audit']['model_associated_split']} under the "
            "model-associated registry; "
            f"{qd['split_audit']['later_canonical_registry']} under the later registry.",
            "",
            "## Promotion status",
            "",
            *[f"- {item}" for item in snapshot["remaining_promotion_gates"]],
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(span_path: Path, qd_path: Path, byt5_paths: tuple[Path, ...]) -> dict[str, Any]:
    span = load(span_path)
    qd = load(qd_path)
    byt5 = [load(path) for path in byt5_paths]
    sample_hash = span["protocol"]["heldout_sample_sha256"]
    if any(report["protocol"]["sample_sha256"] != sample_hash for report in byt5):
        raise ValueError("ByT5 and baseline reports do not use the same target manifest")
    cases = span["cases"]
    if len({case["item_id"] for case in cases}) != len(cases):
        raise ValueError("duplicate target IDs in span report")

    labels = {
        "Preserved-only word span": "uwc_word",
        "Preserved-only TavBERT": "char_unknown",
        "Embible-style overlap": "embible_overlap_ensemble",
        "Dev-fitted rank fusion": "rank_ensemble",
    }
    span_systems = {}
    hit_vectors = {}
    for label, key in labels.items():
        metrics = span["results"][key]
        vector = [topk_hit(case, key, 10) for case in cases]
        hit_vectors[key] = vector
        span_systems[label] = {
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "top10": metrics["top10"],
            "cer": metrics["mean_top1_cer"],
            "boundary_f1": metrics["mean_boundary_f1"],
            "word_count_mae": metrics["mean_word_count_error"],
            "top10_scroll_cluster_ci": clustered_ci(
                cases, lambda case, system=key: topk_hit(case, system, 10)
            ),
            "by_word_count": span["results"]["by_word_count"],
        }

    comparisons = {
        "word_vs_character": exact_mcnemar(hit_vectors["uwc_word"], hit_vectors["char_unknown"]),
        "word_vs_embible_overlap": exact_mcnemar(
            hit_vectors["uwc_word"], hit_vectors["embible_overlap_ensemble"]
        ),
        "word_vs_rank_fusion": exact_mcnemar(
            hit_vectors["uwc_word"], hit_vectors["rank_ensemble"]
        ),
    }
    holm_adjust(comparisons)

    checkpoint_rows = []
    training_configs = []
    for path, report in zip(byt5_paths, byt5):
        config = training_metadata(report["protocol"]["model_dir"])
        training_configs.append(config)
        result = report["results"]
        checkpoint_rows.append(
            {
                **config,
                "top1": result["exact_top1"],
                "top5": result["exact_top5"],
                "top10": result["exact_top10"],
                "by_word_count": result["by_word_count"],
                "model_sha256": report["protocol"]["model_sha256"],
                "artifact": str(path.relative_to(ROOT)),
            }
        )
    controlled_keys = {(row["epochs"], row["batch_size"], row["learning_rate"]) for row in training_configs}

    qd_ci = qd["target_level_any_attributed_restoration"][
        "top10_scroll_cluster_bootstrap_95ci"
    ]
    rag_ci = qd["rag_target_level_any_attributed_restoration"][
        "top10_scroll_cluster_bootstrap_95ci"
    ]
    snapshot = {
        "status": "shareable_reproduced_snapshot_not_final_promotion",
        "primary_task": "synthetic_damage_unknown_length_exact_complete_span_top10",
        "sample_sha256": sample_hash,
        "span_systems": span_systems,
        "paired_top10_mcnemar": comparisons,
        "byt5_checkpoint_replications": {
            "controlled_seed_set": len(controlled_keys) == 1,
            "reason": (
                "seed 41 used different training hyperparameters"
                if len(controlled_keys) != 1
                else "training hyperparameters are identical"
            ),
            "checkpoints": checkpoint_rows,
            "top10_mean_descriptive": statistics.mean(row["top10"] for row in checkpoint_rows),
            "top10_sd_descriptive": statistics.stdev(row["top10"] for row in checkpoint_rows),
        },
        "qd": {
            "p0": {
                **qd["target_level_any_attributed_restoration"],
                "top10_scroll_cluster_ci": qd_ci,
            },
            "rag": {
                **qd["rag_target_level_any_attributed_restoration"],
                "top10_scroll_cluster_ci": rag_ci,
            },
            "u0": qd["diagnostics"]["unconstrained_target_level"],
            "initial_reading": qd["qd_initial_control"],
            "split_audit": split_audit(qd),
        },
        "artifacts": {
            "span": {"path": str(span_path.relative_to(ROOT)), "sha256": sha256(span_path)},
            "qd": {"path": str(qd_path.relative_to(ROOT)), "sha256": sha256(qd_path)},
            "byt5": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in byt5_paths
            ],
        },
        "remaining_promotion_gates": [
            "train and evaluate three checkpoints with identical hyperparameters",
            "freeze one authoritative split registry before new training",
            "complete a formulaic and near-duplicate cross-split audit",
        ],
    }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span", type=Path, default=DEFAULT_SPAN)
    parser.add_argument("--qd", type=Path, default=DEFAULT_QD)
    parser.add_argument("--byt5", type=Path, nargs="+", default=list(DEFAULT_BYT5))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    snapshot = aggregate(args.span.resolve(), args.qd.resolve(), tuple(path.resolve() for path in args.byt5))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(snapshot), encoding="utf-8")
    print(json.dumps({
        "status": snapshot["status"],
        "sample_sha256": snapshot["sample_sha256"],
        "span_systems": {
            name: {key: value for key, value in metrics.items() if key != "by_word_count"}
            for name, metrics in snapshot["span_systems"].items()
        },
        "paired_top10_mcnemar": snapshot["paired_top10_mcnemar"],
        "byt5_checkpoint_replications": snapshot["byt5_checkpoint_replications"],
        "qd": snapshot["qd"],
        "remaining_promotion_gates": snapshot["remaining_promotion_gates"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
