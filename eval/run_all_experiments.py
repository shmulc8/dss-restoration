"""Unified Master Experiment Suite Runner.

Executes all 5 core research evaluations:
1. Ultimate Parallel-Witness RAG + Enhanced Decoding Pipeline
2. Intact Text (Ground-Truth Ink) vs. Real Physical Lacunae (Editor Concordance)
3. Strict Composition-Level Split Validation (26 Held-Out Compositions)
4. TavBERT Character-Level Benchmark Comparison
5. Cross-Epoch Historical Hebrew Generalization Benchmark

Outputs comprehensive report to analysis/reports/full_experiment_suite_report.md.
"""
import sys
import subprocess
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

experiments = [
    ("SQE & DSS Ecosystem Connector Test", ["python", "utils/sqe_connector.py"]),
    ("Cross-Epoch Historical Hebrew Benchmark", ["python", "eval/tf_historical_hebrew_eval.py"]),
    ("Intact Text vs Real Lacunae Benchmark", ["python", "eval/tf_intact_vs_lacuna_eval.py"]),
    ("Strict Composition-Level Split Benchmark", ["python", "eval/tf_composition_split_eval.py"]),
    ("TavBERT Character-Level Benchmark", ["python", "eval/tf_tavbert_eval.py"]),
    ("Ultimate RAG + Enhanced Decoding Pipeline", ["python", "eval/tf_lacuna_len_aeneas_enhanced.py"]),
]

print("==================================================")
print("=== STARTING UNIFIED MASTER EXPERIMENT SUITE ===")
print("==================================================\n")

start_time = time.time()
results_summary = []

for name, cmd in experiments:
    print(f"▶ Running Experiment: {name}...", flush=True)
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
    "# Master Experiment Suite Report",
    f"*Executed on {time.strftime('%Y-%m-%d %H:%M:%S')} (Total Runtime: {total_elapsed/60:.1f} minutes)*",
    "",
    "## 📊 Summary of Executed Experiments",
    "",
    "| Experiment Name | Status | Duration | Key Outcome |",
    "| :--- | :---: | :---: | :--- |",
]

for name, status, duration, stdout in results_summary:
    # Extract last line summary from stdout
    last_line = [line.strip() for line in stdout.splitlines() if line.strip()][-1] if stdout else "No output"
    report_lines.append(f"| {name} | {status} | {duration:.1f}s | {last_line[:80]} |")

report_lines.append("\n## 🔍 Detailed Logs\n")
for name, status, duration, stdout in results_summary:
    report_lines.append(f"### {name}\n```\n{stdout}\n```\n")

report_md = "\n".join(report_lines)
out_file = ROOT / "analysis" / "reports" / "full_experiment_suite_report.md"
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(report_md, encoding="utf-8")

print(f"\n🎉 ALL EXPERIMENTS COMPLETED IN {total_elapsed/60:.1f} MINUTES!")
print(f"Report saved to: {out_file}")
