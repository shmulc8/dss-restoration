"""Unit tests for full_test_runner module."""

import pytest
from pathlib import Path
from eval.full_test_runner import run_full_test_benchmark


def test_full_test_runner_missing_directory():
    with pytest.raises(FileNotFoundError):
        run_full_test_benchmark("non_existent_directory_xyz")


def test_full_test_runner_on_merged_results():
    run_dir = Path("external_comparison/results/tavbert-base")
    if run_dir.exists() and (run_dir / "predictions.jsonl").exists():
        report = run_full_test_benchmark(str(run_dir), num_bootstrap=10)
        assert report["total_sentences"] > 0
        assert "headline_metrics_all_words" in report
        assert "aligned_only_metrics" in report
