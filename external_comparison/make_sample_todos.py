"""Build a fixed 100-sentence paired sample and per-model TODO shard files.

Sample = every test sentence any model already completed, topped up with the
next untouched test rows to exactly 100. For each model, the TODO file holds
only the sample sentences that model hasn't finished yet (last sentence of
each interrupted predictions.jsonl is treated as incomplete and redone).

Writes: sample_uids.json, sample_todo_<tag>.xlsx (only if non-empty).
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = (
    HERE.parent
    / "external_impl"
    / "new_dead_sea_scrolls"
    / "data_preparation"
    / "dss_sentences_min7_splits_ppp_nonbib.xlsx"
)
TAGS = ["tavbert-base", "tavbert-finetuned", "msbert-base", "msbert-finetuned"]
SAMPLE_SIZE = 100


def uid(row):
    key = f"{row['scroll']}|{row['fragment']}|{row['line_start']}|{row['line_end']}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


df = pd.read_excel(DATA).drop(columns=["Unnamed: 0"], errors="ignore")
test = df[df["split"] == "test"].reset_index(drop=True)
test["uid"] = test.apply(uid, axis=1)

completed = {}
for tag in TAGS:
    done = set()
    for pf in sorted((HERE / "shard_eval_results" / tag).glob("*/*/predictions.jsonl")):
        uids_in_order = []
        for line in pf.read_text(encoding="utf-8").splitlines():
            if '"record": "sentence"' in line:
                uids_in_order.append(json.loads(line)["sentence_uid"])
        if uids_in_order:
            done.update(uids_in_order[:-1])  # last one may be mid-write
    completed[tag] = done
    print(f"{tag}: {len(done)} completed sentences reusable")

touched = set().union(*completed.values())
sample = [u for u in test["uid"] if u in touched]
for u in test["uid"]:
    if len(sample) >= SAMPLE_SIZE:
        break
    if u not in touched:
        sample.append(u)
sample = sample[:SAMPLE_SIZE]
(HERE / "sample_uids.json").write_text(json.dumps(sample, indent=0))
print(f"sample size: {len(sample)} (reusing {len(touched & set(sample))} touched)")

for tag in TAGS:
    todo_uids = [u for u in sample if u not in completed[tag]]
    todo = test[test["uid"].isin(todo_uids)].drop(columns=["uid"])
    out = HERE / f"sample_todo_{tag}.xlsx"
    todo.to_excel(out)
    print(f"{tag}: {len(todo)} sentences to run -> {out.name}")
