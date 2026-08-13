"""Merge shard/top-up predictions per model, restrict to the fixed
100-sentence sample, and rescore with the external runner's score_run_dir.

Dedup rule: a sentence may appear in an interrupted shard run and again in the
top-up run; keep the copy with the most span records (complete beats truncated).
Only sentences present for ALL four models are scored (paired sample).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
from tuning.metrics_runner import score_run_dir

TAGS = ["tavbert-base", "tavbert-finetuned", "msbert-base", "msbert-finetuned"]
SAMPLE = set(json.loads((HERE / "sample_uids.json").read_text()))

groups_by_tag = {}
for tag in TAGS:
    groups = {}  # uid -> {"sentence": rec, "spans": [...]}
    for pf in sorted((HERE / "shard_eval_results" / tag).glob("*/*/predictions.jsonl")):
        file_groups = defaultdict(lambda: {"sentence": None, "spans": []})
        for line in pf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = rec["sentence_uid"]
            if rec.get("record") == "sentence":
                file_groups[uid]["sentence"] = rec
            else:
                file_groups[uid]["spans"].append(rec)
        for uid, g in file_groups.items():
            if uid not in SAMPLE or g["sentence"] is None:
                continue
            if uid not in groups or len(g["spans"]) > len(groups[uid]["spans"]):
                groups[uid] = g
    groups_by_tag[tag] = groups
    print(f"{tag}: {len(groups)} sample sentences with predictions")

common = set.intersection(*(set(g) for g in groups_by_tag.values()))
print(f"paired across all 4 models: {len(common)} sentences")

summary = {}
for tag in TAGS:
    merged = HERE / "merged_results" / tag
    merged.mkdir(parents=True, exist_ok=True)
    with open(merged / "predictions.jsonl", "w", encoding="utf-8") as out:
        for uid in sorted(common):
            g = groups_by_tag[tag][uid]
            out.write(json.dumps(g["sentence"], ensure_ascii=False) + "\n")
            for span in g["spans"]:
                out.write(json.dumps(span, ensure_ascii=False) + "\n")
    src_manifest = next((HERE / "shard_eval_results" / tag).glob("*/*/manifest.json"))
    manifest = json.loads(src_manifest.read_text())
    manifest["paired_sample"] = {
        "n_sentences": len(common),
        "source": "sample_uids.json",
    }
    (merged / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    word_df, metrics, counts = score_run_dir(merged)
    o = metrics["overall"]
    summary[tag] = {
        "sentences": counts["sentences"],
        "masked_words": counts["masked_words"],
        "unaligned": counts["unaligned_words"],
        "hit@1": o["hit_at_1"]["mean"],
        "hit@3": o["hit_at_3"]["mean"],
        "hit@5": o["hit_at_5"]["mean"],
        "hit@10": o["hit_at_10"]["mean"],
        "hit@10_lo": o["hit_at_10"]["ci_low"],
        "hit@10_hi": o["hit_at_10"]["ci_high"],
        "char_sim": o["char_sim_top1"]["mean"],
        "mrr": o["mrr"]["mean"],
    }

print("\n=== PAIRED SAMPLE SUMMARY (aligned-only scoring, their protocol) ===")
for tag, s in summary.items():
    print(
        f"{tag}\tsent={s['sentences']}\twords={s['masked_words']}\tunaligned={s['unaligned']}\t"
        f"hit@1={s['hit@1']:.4f}\thit@3={s['hit@3']:.4f}\thit@5={s['hit@5']:.4f}\t"
        f"hit@10={s['hit@10']:.4f} [{s['hit@10_lo']:.4f},{s['hit@10_hi']:.4f}]\t"
        f"char_sim={s['char_sim']:.4f}\tmrr={s['mrr']:.4f}"
    )

print("\n=== UNALIGNED-AS-MISS (headline per decision #8) ===")
for tag, s in summary.items():
    total = s["masked_words"]
    scored = total - s["unaligned"]
    for k in ("hit@1", "hit@10"):
        adj = s[k] * scored / total if total else 0.0
        print(
            f"{tag} {k}_all_words: {adj:.4f} (aligned-only {s[k]:.4f}, {s['unaligned']}/{total} unaligned)"
        )

(HERE / "merged_results" / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nwrote merged_results/summary.json")
