#!/usr/bin/env python3
"""Build the paper snapshot from real target-level evaluation artifacts.

This compatibility entry point replaces the previous demonstration runner,
which simulated hits and was never valid evidence. Model inference is performed
by ``tf_embible_dss_benchmark.py``, ``tf_tokenization_free_benchmark.py``, and
``run_qd_benchmark.py``; this command validates and aggregates their outputs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aggregate_paper_results import main


if __name__ == "__main__":
    main()
