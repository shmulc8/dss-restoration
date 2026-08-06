# CLAUDE.md

## Project overview

Research project on restoring lost text in the Dead Sea Scrolls, built on the `ETCBC/dss` Text-Fabric
corpus. It is the counterpart to [`new_dead_sea_scrolls`](../new_dead_sea_scrolls) and shares that
project's masking/scoring machinery; the merge plan for the two codebases is
[UNIFICATION_DECISION_POINTS.md](UNIFICATION_DECISION_POINTS.md).

The distinguishing question here is **how much the surviving physical evidence at a lacuna — the ink
still on the parchment, and the size of the hole — contributes to restoration.** That is measured by
contrasting two information regimes on the same targets, so the project runs two evaluation tracks
rather than one.

The repository is a linear pipeline of five stages, each consuming the previous stage's output:

1. **`curation/`** — reads `ETCBC/dss` via Text-Fabric and emits the corpus. A word is dropped if
   *any* of its signs is editor-supplied (`rec == 1`), damage-marked (`rem == 1`) or `#`, so the model
   only ever sees physically surviving ink; `validate_preserved_nonbib_corpus.py` hard-fails if an
   editorial character survives into the output. Writes `curation/derived/`:
   `preserved_nonbib_chunks.jsonl` (1,647 chunks), `nonbib_lacunae.jsonl` (27,814 real lacunae),
   `qd_researcher_variants.jsonl` (1,811 published reading proposals), and a manifest carrying
   SHA-256s and counts.
2. **`data_preparation/`** — assigns every scroll to a split by a deterministic hash of its name
   (`sha1(scroll) % 100`, cut points 73/88), identical to the sibling project's scheme. The realized
   assignment is frozen in `dss_scroll_splits_v1.json` (531 train / 108 val / 93 test, 0 straddling)
   and every code path loads that file through `splits.py`.
3. **`tuning/`** — fine-tuning plus the shared masking/prediction/scoring machinery: `eval_utils.py`
   (the classes), `metrics_runner.py` (scoring, hit@k, cluster bootstrap, `mcnemar_test`),
   `candidate_generator.py` (physical-ink filtering, length ensembling), `tokenizer_compat.py`.
4. **`experiments/`** — one driver per benchmark, each writing a durable run directory under
   `experiments/results/runs/<run_id>/` containing `predictions.jsonl`, `metrics.json`,
   `word_scores.csv` and `manifest.json`.
5. **`comparison/`** — aggregation and reporting over those run directories; generated reports live
   in `comparison/reports/` and are the only citable source of numbers.

`utils/` holds cross-cutting helpers (paths, Hebrew morphology, clitic joining, book filters).
`tests/` is a conventional top-level suite. `demo/` is a separate web app and is not part of the
pipeline.

## The two evaluation tracks

Track A and Track B measure different things on different data. **Their numbers are not comparable
and must never be chained into one progression.**

- **Track A — synthetic cloze (`scatter-30`).** Hides 30% of the content words in *preserved* text.
  Driver: `experiments/run_cloze_benchmark.py`. This is synthetic damage, not a real lacuna.
- **Track B — real lacunae.** Physical damage cuts *through* words, leaving partial ink. Scored as
  agreement with published scholarly restorations from Qumran-Digital. Driver:
  `experiments/run_qd_benchmark.py`.

Three information regimes are named throughout and must be stated with every number:

| Regime | What the model gets |
| :--- | :--- |
| `U0` | context only |
| `O-len` | context + gold character length as a proxy |
| `P0` | context + surviving ink + estimated gap length (±1) |

## Evidence discipline

- **[docs/RESULTS.md](docs/RESULTS.md) is the evidence register** — the single index of paper-facing
  numbers. A figure that is not there is exploratory, superseded, or not yet measured.
- **Every number must trace to a generated artifact** under `comparison/reports/` or
  `experiments/results/runs/`. Numbers written directly into a document, with no run behind them,
  are the failure mode this project has already had once.
- **`tests/test_public_claims.py` enforces that**: it fails if a figure with no artifact appears on a
  public surface, if the QD headline drifts from the scorer's own report, or if a significance claim
  is made without a test behind it. Documents covered are README, `docs/RESULTS.md`,
  `docs/PAPER_PRESENTATION.html`, `docs/PAPER_MASTER_SUMMARY.md` and the two demo pages.
- **Statistics.** Confidence intervals come from a sentence-level percentile cluster bootstrap
  (B=1000) — clustering on sentences, because words inside a sentence share a context and a masking
  draw. Masking is seeded per sentence (`sha1(seed|uid)`), so runs over the same protocol are
  **paired**, which is what makes `mcnemar_test` valid between two models.

## Known issues

- **Two split registries coexist.** `data_preparation/dss_scroll_splits_v1.json` (canonical, 732
  scrolls) and the corpus manifest's own `scroll_splits` (736 scrolls, train/dev/heldout). The QD
  benchmark ran against the manifest registry, so 40 of its 74 targets fall in the canonical
  registry's *train* split. Consolidating these is decision #3 and blocks every model comparison.
- **`transformers` is at 5.13.0 here, while the sibling project pins 4.48.0.** On 5.x, a checkpoint
  declaring `"tokenizer_class": "BertTokenizer"` is honoured literally, so plain WordPiece is built
  from `vocab.txt` and the `Split` pre-tokenizer in `tokenizer.json` is discarded. For a
  character-vocab checkpoint that collapses **every Hebrew word to one `[UNK]`**, and nothing raises:
  training and validation run happily on a corpus of `[UNK]`s while val loss goes to ~0.0001.
  This is live on this version — `AutoTokenizer` returns 100% `[UNK]` on Hebrew for
  `dicta-il/dictabert-large-char`. `tokenizer_compat.load_tokenizer` detects it, reloads from
  `tokenizer.json`, and hard-asserts the result encodes Hebrew without `[UNK]`. **Do not bypass
  `load_tokenizer`**, and sanity-check the `[UNK]` fraction when adding a checkpoint.
- **`tuning/eval_utils.py` has drifted** from the sibling project's copy of the same classes (624
  lines against 719). They are supposed to stay in lockstep.

## Environment

```bash
uv run pytest                       # full suite

PYTHONPATH=. uv run python experiments/run_cloze_benchmark.py --run-dir experiments/results/runs/tavbert-base
PYTHONPATH=. uv run python experiments/run_qd_benchmark.py    # offline, cached QD snapshot
PYTHONPATH=. uv run python experiments/lacuna_corpus_stats.py
```

`models/` (checkpoints) and `scratch/` (exploratory runs) are gitignored but **must not be deleted** —
they hold the raw outputs behind the reports in `comparison/reports/`.

## Data licensing

`ETCBC/dss` is **CC BY-NC 4.0**. Attribution must name Martin G. Abegg Jr., James E. Bowley and
Edward M. Cook. The non-commercial term applies to any redistributed derived text, including examples
printed in a paper or shipped in the demo.
