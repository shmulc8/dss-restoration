import json
from pathlib import Path

from experiments.build_qd_composition_exclusion import validate


def test_qd_composition_exclusion_has_no_label_overlap() -> None:
    result = json.loads(
        Path("experiments/results/paper/qd_composition_exclusion.json").read_text(
            encoding="utf-8"
        )
    )
    validate(result)
    assert result["verified_composition_overlap"] == []
    assert result["retained_train_chunks"] < result["original_train_chunks"]
