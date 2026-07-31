#!/bin/sh
# Finish the 100-sentence sample with at most 2 concurrent eval processes.
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python

printf '%s\n' \
  "tau/tavbert-he tavbert-base" \
  "models/tavbert_full_ppp_nonbib_local tavbert-finetuned" \
  "dicta-il/MsBERT msbert-base" \
  "models/msbert_full_ppp_nonbib_local msbert-finetuned" |
while read -r model tag; do
  echo "$PY run_shard_eval.py $model $tag sample_todo_${tag}.xlsx > shard_logs/sample_${tag}.log 2>&1"
done | xargs -P 2 -I {} sh -c '{}'

echo "SAMPLE RUN DONE"
grep -h "EVAL OK\|Traceback" shard_logs/sample_*.log
