#!/bin/sh
# Launch 4 models x 3 shards = 12 parallel shard evals; wait for all.
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
mkdir -p shard_logs

for model_tag in "tau/tavbert-he tavbert-base" \
                 "models/tavbert_full_ppp_nonbib_local tavbert-finetuned" \
                 "dicta-il/MsBERT msbert-base" \
                 "models/msbert_full_ppp_nonbib_local msbert-finetuned"; do
  model=${model_tag% *}
  tag=${model_tag#* }
  for shard in shards/test_shard_*.xlsx; do
    name=$(basename "$shard" .xlsx)
    "$PY" run_shard_eval.py "$model" "$tag" "$shard" \
      > "shard_logs/${tag}_${name}.log" 2>&1 &
  done
done

wait
echo "ALL SHARD EVALS DONE"
grep -h "EVAL OK\|Traceback\|Error" shard_logs/*.log | tail -20
