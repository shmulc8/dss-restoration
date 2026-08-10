# Reproducibility snapshot

This repository is a reproduced research snapshot for team review. The
authoritative status is `team_review_evidence_snapshot` in
`experiments/results/paper/paper_results_snapshot.json`.

## Environment

- Python 3.14
- dependencies pinned in `requirements.txt`
- TeX Live with pdfLaTeX or XeLaTeX and the packages imported by
  `docs/paper.tex`

Create a clean Python environment:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Model checkpoints are not redistributed in the repository. Frozen paper
artifacts record checkpoint identifiers and SHA-256 hashes; evaluations that
need local checkpoints must resolve the paths documented in their manifests.

## Validate the checked-in snapshot

These commands do not retrain models. They reproduce the paper-facing
aggregation from checked-in target-level results and validate the data and
protocol boundaries:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python curation/validate_preserved_nonbib_corpus.py --derived-only
.venv/bin/python experiments/validate_leakage.py
.venv/bin/python experiments/validate_paper_protocol.py
.venv/bin/python experiments/audit_split_similarity.py
.venv/bin/python experiments/run_paper_benchmark.py
.venv/bin/python experiments/build_paper_data_profile.py
.venv/bin/python docs/generate_paper_tables.py
git diff --exit-code -- \
  docs/paper_results_manifest.json \
  docs/paper_tables.tex \
  experiments/results/paper/paper_data_profile.json \
  experiments/results/paper/paper_results_snapshot.json \
  experiments/results/paper/PAPER_RESULTS_SNAPSHOT.md
```

For the stronger source-backed corpus check, obtain ETCBC DSS Text-Fabric 2.0
under its license and run:

```bash
DSS_TF_DIR=/absolute/path/to/ETCBC/dss/tf/2.0 \
  .venv/bin/python curation/validate_preserved_nonbib_corpus.py
```

## Compile the paper

From the repository root:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../output/pdflatex docs/paper.tex
latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=../output/xelatex docs/paper.tex
```

The tracked `docs/paper.pdf` is the team-review PDF. CI independently compiles
the source and uploads its PDF as a workflow artifact.

## Evidence boundary

The headline natural-lacuna result is agreement with at least one attributed
modern proposal, not recovery of historical truth. The 300-span task is a
synthetic unknown-length benchmark with known answers. Corpus counts are not
model evaluation counts. ByT5 has a matched three-seed replication; the MLM
comparisons remain single-checkpoint evidence. A future model-comparison claim
requires matched MLM seeds and a composition-grouped training split.

No human-subjects study is part of this snapshot.
