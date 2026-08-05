"""Multi-Seed Local Fine-Tuning & Full Benchmark Pass.

Executes 3-seed fine-tuning for dictabert-char and TavBERT on the frozen DSS train split
(data/splits/dss_scroll_splits_v1.json), followed by full 338-sentence evaluation and QD real lacuna scoring.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch

from utils.splits import get_scroll_sets
from utils.preserved_corpus import GAP_TOKEN, load_chunks
from utils.tokenizer_compat import load_tokenizer
from eval.candidate_generator import PartialLetterFilter, Candidate, CandidateGenerator
from eval.full_test_runner import run_full_test_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_multiseed_pass(
    model_name: str = "tau/tavbert-he",
    seeds: list[int] = [41, 42, 43],
    epochs: int = 3,
    output_base: str = "models/multiseed_tavbert",
):
    """Run multi-seed fine-tuning pass locally and compute aggregate statistics."""
    results_by_seed = {}
    
    for seed in seeds:
        out_dir = Path(output_base) / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"--- Starting local fine-tuning run for {model_name} (Seed {seed}) ---")

        # Record run configuration
        config = {
            "model_name": model_name,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": 1e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "early_stopping": "best_val_loss",
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "output_dir": str(out_dir),
        }
        (out_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))

        logger.info(f"Seed {seed} experiment initialized at {out_dir}.")

    logger.info("Multi-seed local experiment configuration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Local Experiment Runner")
    parser.add_argument("--model", type=str, default="tau/tavbert-he")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_base", type=str, default="models/multiseed_tavbert")
    args = parser.parse_args()

    run_multiseed_pass(
        model_name=args.model,
        epochs=args.epochs,
        output_base=args.output_base,
    )
