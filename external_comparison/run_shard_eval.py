"""Run the external eval_runner on one shard file.

Usage: python run_shard_eval.py <model_id_or_path> <tag> <shard_xlsx>
Output goes to shard_eval_results/<tag>/shard_<name>/.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(TUNING))
from eval.metrics_runner import run_eval

model_id, tag, shard = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out_root = HERE / "shard_eval_results" / tag / shard.stem
out_root.mkdir(parents=True, exist_ok=True)
print(f"EVAL START: {tag} {shard.stem}", flush=True)
run_dir = run_eval(
    model_id=model_id,
    dataset_path=str(shard),
    out_root=str(out_root),
    split="test",
    limit=None,
    push_to_hub=False,
)
print(f"EVAL OK: {tag} {shard.stem} -> {run_dir}", flush=True)
