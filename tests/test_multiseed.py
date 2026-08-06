"""Unit test for multi-seed experiment setup."""

import pytest
from pathlib import Path
from tuning.run_multiseed_experiment import run_multiseed_pass


def test_multiseed_experiment_setup(tmp_path: Path):
    out_base = tmp_path / "multiseed_test"
    run_multiseed_pass(
        model_name="dicta-il/dictabert-char",
        seeds=[41, 42],
        epochs=1,
        output_base=str(out_base),
    )
    assert (out_base / "seed_41" / "experiment_config.json").exists()
    assert (out_base / "seed_42" / "experiment_config.json").exists()
