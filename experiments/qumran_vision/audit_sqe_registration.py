#!/usr/bin/env python3
"""Audit whether frozen QD targets have sign-level image registration in SQE."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QD = ROOT / "experiments/results/paper/qd_methods_seed42_20260811.json"
DEFAULT_OUTPUT = ROOT / "experiments/results/exploratory/qd_sqe_image_registration_20260812.json"
IMAGE_REF = "qumranica/sqe-database:0.33.0"
EXPECTED_DIGEST = "sha256:450a994563b79524de5a7cd3cdb9289a09bca9bfc6b64bacefe852678e1545dd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_ids(path: Path) -> list[int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return sorted({int(row["word_id"]) for row in report["targets"]})


def target_id_sha256(ids: list[int]) -> str:
    return hashlib.sha256(("\n".join(map(str, ids)) + "\n").encode()).hexdigest()


def query(ids: list[int]) -> str:
    values = ",".join(f"({word_id})" for word_id in ids)
    return f"""
CREATE TEMPORARY TABLE target_word (qwb_word_id INT UNSIGNED PRIMARY KEY);
INSERT INTO target_word VALUES {values};
SELECT 'target_count', COUNT(*) FROM target_word;
SELECT 'qwb_word_overlap', COUNT(DISTINCT t.qwb_word_id)
  FROM target_word t JOIN qwb_word q USING(qwb_word_id);
SELECT 'sign_section_overlap', COUNT(DISTINCT t.qwb_word_id)
  FROM target_word t JOIN sign_stream_section_to_qwb_word m USING(qwb_word_id);
SELECT 'sign_roi_overlap', COUNT(DISTINCT t.qwb_word_id)
  FROM target_word t
  JOIN sign_stream_section_to_qwb_word m USING(qwb_word_id)
  JOIN position_in_stream_to_section_rel rel USING(sign_stream_section_id)
  JOIN position_in_stream p USING(position_in_stream_id)
  JOIN sign_interpretation_roi r
    ON r.sign_interpretation_id = p.sign_interpretation_id;
SELECT 'database_qwb_words', COUNT(*) FROM qwb_word;
SELECT 'database_sign_rois', COUNT(*) FROM sign_interpretation_roi;
SELECT 'database_qwb_words_with_roi', COUNT(DISTINCT m.qwb_word_id)
  FROM sign_stream_section_to_qwb_word m
  JOIN position_in_stream_to_section_rel rel USING(sign_stream_section_id)
  JOIN position_in_stream p USING(position_in_stream_id)
  JOIN sign_interpretation_roi r
    ON r.sign_interpretation_id = p.sign_interpretation_id;
SELECT 'database_sqe_images', COUNT(*) FROM SQE_image;
"""


def run_live(container: str, ids: list[int]) -> dict[str, int]:
    image_id = subprocess.run(
        ["docker", "inspect", container, "--format", "{{.Image}}"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if image_id != EXPECTED_DIGEST:
        raise ValueError(f"SQE container uses {image_id}, expected {EXPECTED_DIGEST}")
    result = subprocess.run(
        ["docker", "exec", "-i", container, "mysql", "-uroot", "-pnone", "SQE", "-N"],
        input=query(ids),
        text=True,
        capture_output=True,
        check=True,
    )
    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        key, value = line.split("\t", 1)
        counts[key] = int(value)
    return counts


def validate(report: dict[str, object], qd_path: Path) -> None:
    ids = target_ids(qd_path)
    expected = {
        "target_count": len(ids),
        "target_id_sha256": target_id_sha256(ids),
        "qd_artifact_sha256": sha256(qd_path),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"stale registration audit: {key}={report.get(key)!r}, expected {value!r}")
    counts = report["coverage"]
    if not isinstance(counts, dict) or counts.get("sign_roi_overlap") != 0:
        raise ValueError("paper audit must not claim a blocked fusion gate when registered targets exist")
    if report.get("status") != "blocked_no_registered_qd_targets":
        raise ValueError("unexpected registration-audit status")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qd", type=Path, default=DEFAULT_QD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--container", help="running SQE database container for a live audit")
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    qd_path = args.qd.resolve()
    output = args.output.resolve()
    if args.validate_existing:
        validate(json.loads(output.read_text(encoding="utf-8")), qd_path)
        print(f"validated {output}")
        return
    if not args.container:
        parser.error("--container is required unless --validate-existing is used")

    ids = target_ids(qd_path)
    counts = run_live(args.container, ids)
    if counts.get("target_count") != len(ids):
        raise ValueError("SQE audit did not retain the complete frozen target set")
    registered = counts.get("sign_roi_overlap", 0)
    report = {
        "status": "ready_for_fusion" if registered else "blocked_no_registered_qd_targets",
        "audit_date": date.today().isoformat(),
        "target_count": len(ids),
        "target_id_sha256": target_id_sha256(ids),
        "qd_artifact": str(qd_path.relative_to(ROOT)),
        "qd_artifact_sha256": sha256(qd_path),
        "sqe_database_image": IMAGE_REF,
        "sqe_database_digest": EXPECTED_DIGEST,
        "coverage": counts,
        "join_path": [
            "qwb_word",
            "sign_stream_section_to_qwb_word",
            "position_in_stream_to_section_rel",
            "position_in_stream",
            "sign_interpretation_roi",
        ],
        "interpretation": (
            "No image-only or fused QD score is estimable from this public snapshot; "
            "new sign-level registration is required."
            if not registered
            else "At least one frozen QD target has sign-level registration."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
