"""Score all five merged runs, overall and on the ByT5-clean scroll subset.

Clean subset = sample scrolls that are heldout/dev in Shmulik's split
(11Q17, 1Q27, 11Q5); the rest (1Q16, 1Q25, 1QM) are in ByT5's training data.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(TUNING))
from eval.metrics_runner import score_run_dir

TAGS = [
    "tavbert-base",
    "tavbert-finetuned",
    "msbert-base",
    "msbert-finetuned",
    "byt5-preserved",
    "byt5-olen",
]
CLEAN_SCROLLS = {"11Q17", "1Q27", "11Q5"}


def row(tag, metrics, counts):
    o = metrics["overall"]
    return {
        "sentences": counts["sentences"],
        "masked_words": counts["masked_words"],
        "unaligned": counts["unaligned_words"],
        "hit@1": o["hit_at_1"]["mean"],
        "hit@10": o["hit_at_10"]["mean"],
        "hit@10_ci": [o["hit_at_10"]["ci_low"], o["hit_at_10"]["ci_high"]],
        "char_sim": o["char_sim_top1"]["mean"],
        "mrr": o["mrr"]["mean"],
    }


results = {"full": {}, "clean": {}}
for tag in TAGS:
    src = HERE / "merged_results" / tag
    _, metrics, counts = score_run_dir(src)
    results["full"][tag] = row(tag, metrics, counts)

    sub = HERE / "merged_results" / f"{tag}__clean"
    sub.mkdir(exist_ok=True)
    with open(sub / "predictions.jsonl", "w", encoding="utf-8") as out:
        keep = set()
        for line in (src / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("record") == "sentence":
                if rec.get("scroll") in CLEAN_SCROLLS:
                    keep.add(rec["sentence_uid"])
                    out.write(line + "\n")
            elif rec["sentence_uid"] in keep:
                out.write(line + "\n")
    shutil.copy(src / "manifest.json", sub / "manifest.json")
    _, metrics, counts = score_run_dir(sub)
    results["clean"][tag] = row(tag, metrics, counts)

for scope in ("full", "clean"):
    print(f"\n=== {scope.upper()} ({results[scope][TAGS[0]]['sentences']} sentences) — aligned-only ===")
    for tag, s in results[scope].items():
        print(
            f"{tag}\twords={s['masked_words']}\tunal={s['unaligned']}\t"
            f"hit@1={s['hit@1']:.4f}\thit@10={s['hit@10']:.4f} "
            f"[{s['hit@10_ci'][0]:.3f},{s['hit@10_ci'][1]:.3f}]\t"
            f"char_sim={s['char_sim']:.4f}\tmrr={s['mrr']:.4f}"
        )
    print(f"--- {scope}: unaligned-as-miss headline ---")
    for tag, s in results[scope].items():
        tot = s["masked_words"]
        sc = tot - s["unaligned"]
        print(
            f"{tag}\thit@1_all={s['hit@1']*sc/tot:.4f}\thit@10_all={s['hit@10']*sc/tot:.4f}"
        )

(HERE / "merged_results" / "summary_with_subsets.json").write_text(json.dumps(results, indent=2))
print("\nwrote merged_results/summary_with_subsets.json")
