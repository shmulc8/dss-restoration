from pathlib import Path

from experiments.build_qd_composition_exclusion import build


def test_qd_composition_exclusion_has_no_label_overlap() -> None:
    result = build(
        qd_result=Path("experiments/results/paper/qd_evidence_conditions_20260811.json"),
        source_csv=Path("dss_chunks.csv"),
    )
    assert result["verified_composition_overlap"] == []
    assert result["retained_train_chunks"] < result["original_train_chunks"]
