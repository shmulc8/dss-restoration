# Dead Sea Scrolls lacuna restoration

This repository studies whether language models and retrieved textual parallels
can help scholars assess missing text in the non-biblical Dead Sea Scrolls.
The project is deliberately reconstruction-free during fine-tuning: modern
editorial restorations are removed from training targets and from the retrieval
index.

- [Live research site](https://dss-restoration-demo.pages.dev/)
- [Hebrew research deck](https://dss-restoration-demo.pages.dev/slides_he.html)
- [Locked paper methodology](docs/METHODOLOGY.md)
- [Current evidence register](docs/RESULTS.md)

## Research claim

The intended contribution is a leakage-controlled evaluation framework for DSS
restoration, not a claim that a model has recovered the original wording of a
damaged manuscript. The framework separates four questions:

1. Can the model recover genuinely preserved DSS language hidden for testing?
2. Does it rank attested modern scholarly proposals highly at real lacunae?
3. Do train-only textual parallels improve predictions on the same frozen test
   cases?
4. Do candidates and parallels help scholars work more accurately or quickly?

Only the first three have pilot results. The scholar-assistance study has not
yet been run.

## Current results

These numbers are retained as pilot or diagnostic evidence. They are not
interchangeable and must not be collapsed into one accuracy headline.

| Track | Evaluation unit | Result | Interpretation |
| :--- | :--- | :--- | :--- |
| Preserved-language sanity check | 300 intact words from held-out non-biblical scrolls | Preserved-only model: 43.7% Top-20 | Grounded language-recovery check; not a real-lacuna benchmark |
| Attributed real-lacuna comparison | 74 single-word Qumran Digital targets | 63.5% Top-10 | Agreement with any physically compatible attributed reading; not ground truth |
| RAG ablation on those targets | Same 74 targets | 63.5% to 63.5% Top-10 | No measured improvement |
| Multiword RAG pilot | 100 held-out Text-Fabric spans | 7.0% to 9.0% exact-sequence Top-10 | Descriptive pilot; word-slot count is supplied and uncertainty is too large for a paper claim |

The physical-constraint ablation on the 74 Qumran Digital targets scores 9.5%
Top-10 without visible-letter and approximate-length constraints versus 63.5%
with them. This measures the complete constrained decoder, not an improvement
in the language model.

The exact provenance, limitations, and status of every retained number are in
[`docs/RESULTS.md`](docs/RESULTS.md). Superseded master reports and the earlier
RAG evaluation that leaked gold length have been removed from the current
repository. Git history preserves them for audit.

## Clean data and training path

The derived corpus is built from `ETCBC/dss` Text-Fabric 2.0:

1. keep only non-biblical scroll material;
2. replace reconstructed or unknown material with anonymous `<GAP>` slots;
3. permit only physically preserved words to become fine-tuning labels;
4. split by scroll before training, tuning, retrieval, or evaluation.

The checked-in manifest records 736 scrolls, 1,647 chunks, and 27,814 lacuna
records. Rebuild and validate the corpus with:

```bash
.venv/bin/python data/build_preserved_nonbib_corpus.py
.venv/bin/python data/validate_preserved_nonbib_corpus.py
.venv/bin/python eval/validate_leakage.py
```

Fine-tune the current preserved-only baseline with:

```bash
.venv/bin/python training/finetune_span_preserved_nonbib.py
```

This checkpoint is a baseline, not the final paper model. The paper protocol
requires multiple seeds, dev-only model selection, composition-level stress
tests, realistic damage generation, and an unknown-length decoder. The baseline
matrix also adopts Embible's Hebrew character/word comparison: word-only,
TavBERT-style character-only, constrained word completion, and a calibrated
character-word ensemble under known, predicted, and unknown whitespace.

## Current evaluation entry points

```bash
# List the paper-facing evaluations without running them
.venv/bin/python eval/run_all_experiments.py --list

# Run validation only
.venv/bin/python eval/run_all_experiments.py --checks

# Run the retained pilot evaluations
.venv/bin/python eval/run_all_experiments.py --pilots
```

The runner includes only the supported pipeline. Older experimental scripts may
remain for exploratory diagnosis, but they are not registered paper results.

## Result terminology

- **Preserved recovery:** the hidden answer was physically present in the
  manuscript transcription.
- **Literature agreement:** a prediction matches at least one compatible,
  attributed modern proposal.
- **Slot score:** an individual missing word is evaluated independently.
- **Exact-sequence score:** every word in the proposed span must match in order.
- **RAG:** retrieval uses preserved text from training scrolls only; its weight
  is selected on development scrolls.

The primary metric for the next paper benchmark is exact complete-span Top-10
under unknown length. Top-1/5/20, character error rate, reciprocal rank, slot
scores, calibration, and abstention are secondary diagnostics.

## Source and licensing boundary

Text-Fabric attributes its transcription to Martin G. Abegg Jr., James E.
Bowley, and Edward M. Cook, based on Martin Abegg's data files. Its `rec`
feature identifies modern reconstruction but does not attribute each reading to
an editor. Publication-level comparisons therefore use the separately cached,
attributed Qumran Digital snapshot. Those comparisons measure agreement with
the literature, not manuscript truth.

The data manifest and cached-source notes contain the applicable source and
license information. Do not redistribute external editions beyond their
licenses.
