#!/usr/bin/env python3
"""Guard against mistaking configuration scaffolds for multi-seed training.

The former script created per-seed JSON files but did not train or evaluate a
model. It is intentionally retired. Use a model-specific training entry point,
retain checkpoint metadata, and evaluate each checkpoint with a real benchmark.
"""

from __future__ import annotations

import argparse


RETIRED_MESSAGE = (
    "Retired: this entry point never trained models; it only wrote configuration "
    "files. Run a model-specific trainer and the frozen evaluator, then aggregate "
    "target-level artifacts with experiments/run_paper_benchmark.py."
)


def run_multiseed_pass(*args: object, **kwargs: object) -> None:
    """Fail loudly instead of manufacturing experiment-looking directories."""
    del args, kwargs
    raise RuntimeError(RETIRED_MESSAGE)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(RETIRED_MESSAGE)


if __name__ == "__main__":
    main()
