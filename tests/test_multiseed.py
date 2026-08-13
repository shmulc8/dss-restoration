"""The retired scaffold must never look like completed training."""

import pytest
from pathlib import Path
from tuning.run_multiseed_experiment import run_multiseed_pass


def test_multiseed_scaffold_fails_without_writing_artifacts(tmp_path: Path):
    out_base = tmp_path / "multiseed_test"
    with pytest.raises(RuntimeError, match="never trained models"):
        run_multiseed_pass(
            model_name="dicta-il/dictabert-char",
            seeds=[41, 42],
            epochs=1,
            output_base=str(out_base),
        )
    assert not out_base.exists()
