"""Full Corpus Master Evaluation Runner (PER_BUCKET=0 — 100% Full Dataset).

Evaluates all 7,809 lacuna spans across the entire non-biblical Dead Sea Scrolls corpus:
- 3,047 1-word lacunae
- 1,446 2-word lacunae
- 885 3-word lacunae
- 1,209 4-5 word lacunae
- 1,222 6+ word lacunae

Outputs comprehensive report to analysis/reports/FULL_CORPUS_BENCHMARK_REPORT.md.
"""
import os
import sys
import subprocess
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set PER_BUCKET=0 to evaluate 100% of all spans in the dataset
os.environ["PER_BUCKET"] = "0"

experiments = [
    ("SQE & DSS Ecosystem Connector Test", ["python", "utils/sqe_connector.py"]),
    ("Automated Comparative Epigraphic Scorer", ["python", "analysis/compare_scholar_conjectures.py"]),
    ("Inter-Editor Disagreement & Ambiguity Estimator", ["python", "analysis/estimate_editor_disagreement.py"]),
    ("Full Corpus Intact Ink vs. Real Lacunae Evaluation", ["python", "eval/tf_intact_vs_lacuna_eval.py"]),
    ("Full Corpus Strict Composition-Level Split (26 Compositions)", ["python", "eval/tf_composition_split_eval.py"]),
    ("Full Corpus TavBERT Character-Level Benchmark", ["python", "eval/tf_tavbert_eval.py"]),
    ("Full Corpus Ultimate RAG + Enhanced Decoding Pipeline", ["python", "eval/tf_lacuna_len_aeneas_enhanced.py"]),
]

print("==================================================")
print("=== FULL CORPUS MASTER EVALUATION (100% DATASET) ===")
print("==================================================\n")

start_time = time.time()
results_summary = []

for name, cmd in experiments:
    print(f"▶ Running Full-Corpus Experiment: {name} (PER_BUCKET=0)...", flush=True)
    exp_start = time.time()
    try:
        res = subprocess.run(
            [sys.executable, cmd[1]],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - exp_start
        print(f"✔ Completed in {elapsed:.1f}s")
        print(res.stdout[-400:] if len(res.stdout) > 400 else res.stdout)
        print("-" * 50, flush=True)
        results_summary.append((name, "SUCCESS", elapsed, res.stdout))
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - exp_start
        print(f"❌ Failed after {elapsed:.1f}s: {e.stderr[:300]}")
        print("-" * 50, flush=True)
        results_summary.append((name, "FAILED", elapsed, e.stderr))

total_elapsed = time.time() - start_time

# Generate Markdown Report
report_lines = [
    "# Full Corpus Master Evaluation Report (100% Dataset)",
    f"*Executed on {time.strftime('%Y-%m-%d %H:%M:%S')} across 7,809 total lacuna spans (Total Runtime: {total_elapsed/60:.1f} minutes)*",
    "",
    "## 📊 Summary of Executed Full-Corpus Experiments",
    "",
    "| Experiment Name | Status | Duration | Key Outcome |",
    "| :--- | :---: | :---: | :--- |",
]

for name, status, duration, stdout in results_summary:
    last_line = [line.strip() for line in stdout.splitlines() if line.strip()][-1] if stdout else "No output"
    report_lines.append(f"| {name} | {status} | {duration:.1f}s | {last_line[:80]} |")

report_lines.append("\n## 🔍 Detailed Full-Corpus Benchmark Logs\n")
for name, status, duration, stdout in results_summary:
    report_lines.append(f"### {name}\n```\n{stdout}\n```\n")

report_md = "\n".join(report_lines)
out_file = ROOT / "analysis" / "reports" / "FULL_CORPUS_BENCHMARK_REPORT.md"
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(report_md, encoding="utf-8")

print(f"\n🎉 FULL CORPUS MASTER EVALUATION COMPLETED IN {total_elapsed/60:.1f} MINUTES!")
print(f"Full report saved to: {out_file}")
