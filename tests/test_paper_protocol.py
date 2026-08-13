import json
from pathlib import Path

from experiments.validate_paper_protocol import validate


ROOT = Path(__file__).resolve().parents[1]


def test_locked_protocol_passes_validation() -> None:
    protocol = json.loads(
        (ROOT / "experiments" / "paper_protocol_v1.json").read_text(encoding="utf-8")
    )
    assert validate(protocol) == []


def test_validator_rejects_oracle_primary_condition() -> None:
    protocol = json.loads(
        (ROOT / "experiments" / "paper_protocol_v1.json").read_text(encoding="utf-8")
    )
    protocol["primary_condition"]["word_count"] = "gold"
    assert "primary word_count must be unknown" in validate(protocol)
