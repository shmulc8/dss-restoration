"""Headless driver for the external team's tuning/finetune_msbert.ipynb.

Executes the notebook's code cells verbatim except:
  - cell 2 (Colab/HF-Hub setup) is replaced with a no-hub stub
  - TuningConfig defaults for run_name / model_name / data_path are patched
  - EXPORT_EPOCH is forced to None (probe-selected "best" checkpoint)
  - matplotlib runs on the Agg backend (no display)

Usage:
  python run_finetune_headless.py <run_name> <model_id>
e.g.
  python run_finetune_headless.py tavbert_full_ppp_nonbib_local tau/tavbert-he
  python run_finetune_headless.py msbert_full_ppp_nonbib_local dicta-il/MsBERT

cwd should be this directory; runs/ and models/ are created here.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
TUNING = EXT / "tuning"
DATA = EXT / "data_preparation" / "dss_sentences_min7_splits_ppp_nonbib.xlsx"
NOTEBOOK = TUNING / "finetune_msbert.ipynb"

run_name, model_id = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(TUNING))

STUB_CELL_2 = f"""
PUSH_TO_HUB = False
HF_REPO_OWNER = None
HF_PRIVATE = True
HF_PUSH_METRICS_EACH_EPOCH = False
IN_COLAB = False
hf_api, HF_REPO_ID = None, None
HUB_ENABLED = False
print("headless run: hub disabled, run_name={run_name}, model={model_id}")
"""

nb = json.load(open(NOTEBOOK))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
# code cell 0 in the code-only list corresponds to notebook cell 2
cells[0] = STUB_CELL_2

patched = []
for src in cells:
    src, n_run = re.subn(
        r'run_name: str = "[^"]*"', f'run_name: str = "{run_name}"', src
    )
    src, n_model = re.subn(
        r'model_name: str = "[^"]*"', f'model_name: str = "{model_id}"', src
    )
    src, _ = re.subn(r'data_path: str = "[^"]*"', f'data_path: str = "{DATA}"', src)
    src, _ = re.subn(r"^EXPORT_EPOCH = .*$", "EXPORT_EPOCH = None", src, flags=re.M)
    src = src.replace("plt.show()", "plt.close(fig)")
    patched.append(src)

ns = {"__name__": "__main__"}
for i, src in enumerate(patched):
    print(f"\n----- executing code cell {i}/{len(patched) - 1} -----", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), ns)

print("\nFINETUNE DONE:", ns["export_path"])
