"""Run the external eval_runner on the full test split for one model.

Usage: python run_one_eval.py <model_id_or_path> <tag>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TUNING = HERE.parent / "external_impl" / "new_dead_sea_scrolls" / "tuning"
DATA = (
    HERE.parent
    / "external_impl"
    / "new_dead_sea_scrolls"
    / "data_preparation"
    / "dss_sentences_min7_splits_ppp_nonbib.xlsx"
)
OUT = HERE / "full_test_eval_results"

sys.path.insert(0, str(TUNING))
from eval_runner import run_eval

model_id, tag = sys.argv[1], sys.argv[2]
print(f"EVAL START: {tag} ({model_id})", flush=True)
run_dir = run_eval(
    model_id=model_id,
    dataset_path=str(DATA),
    out_root=str(OUT),
    split="test",
    limit=None,
    push_to_hub=False,
)
print(f"EVAL OK: {tag} -> {run_dir}", flush=True)
