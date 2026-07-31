# External comparison: unified-protocol runs (2026-07-31)

Code and result artifacts behind §R1 of `docs/UNIFICATION_DECISION_POINTS.md`:
fine-tuning TavBERT/MsBERT with the external (`new_dead_sea_scrolls`) recipe and
evaluating all model families — including ByT5 via an adapter — under the
external `eval_runner` protocol on a paired 100-sentence test sample.

## Layout

- `run_finetune_headless.py` — executes the external `tuning/finetune_msbert.ipynb`
  code cells headlessly (hub cell stubbed, `EXPORT_EPOCH=None` → probe-best,
  Agg matplotlib). Usage: `python run_finetune_headless.py <run_name> <model_id>`.
- `run_pipeline.py`, `run_one_eval.py`, `make_shards.py`, `run_shard_eval.py`,
  `run_all_shards.sh`, `run_sample_quiet.sh`, `make_sample_todos.py` —
  orchestration for full-split / sharded / paired-sample evaluation.
  Sharding is valid because the external masking is a pure function of
  (seed, sentence identity), not dataset order.
- `byt5_predictor.py` — the ByT5 → `predictions.jsonl` adapter (decision #4):
  reuses the sentence/span structure from a completed MLM run, generates
  beam candidates with ByT5, writes the external schema. `byt5_predictor_olen.py`
  is the O-len variant (30 beams, gold-length ±2 filter).
- `merge_and_score.py`, `score_with_subsets.py` — merge shard predictions,
  restrict to the fixed paired sample (`sample_uids.json`), rescore with the
  external `score_run_dir` (bit-identical scoring layer), report full sample
  and the ByT5-clean scroll subset (11Q17, 1Q27, 11Q5).
- `results/<model>/{metrics.json,manifest.json}` + `results/summary_with_subsets.json` —
  the scored outputs quoted in §R1. `predictions.jsonl` files are kept out of
  git per the external repo's convention; available on request.

## Original run layout

Scripts were executed from `scratch/external_finetune/` (gitignored) with the
external codebase unpacked at `scratch/external_impl/new_dead_sea_scrolls/`
and its dataset at `data_preparation/dss_sentences_min7_splits_ppp_nonbib.xlsx`.
Paths in the scripts resolve relative to that layout; adjust `EXT`/`TUNING`/`DATA`
constants to reproduce elsewhere.
