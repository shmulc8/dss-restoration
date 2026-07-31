"""Split the test rows of the ppp_nonbib dataset into N shard xlsx files.

Masking is a pure function of (seed, sentence identity) in the external runner,
so evaluating shards independently yields identical predictions to one run.
"""
import sys
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3

df = pd.read_excel(DATA).drop(columns=["Unnamed: 0"], errors="ignore")
test = df[df["split"] == "test"].reset_index(drop=True)
out_dir = HERE / "shards"
out_dir.mkdir(exist_ok=True)
for i in range(N):
    shard = test.iloc[i::N]
    out = out_dir / f"test_shard_{i}_of_{N}.xlsx"
    shard.to_excel(out)
    print(out, len(shard))
